"""Direct forecasts anchored to the last observation, with learned trend corrections."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def trend(history, horizon, target_idx=2):
    """Last observed level plus last observed per-sample slope, never future data."""
    last, previous = history[:, -1, target_idx], history[:, -2, target_idx]
    leads = np.arange(1, horizon + 1)
    return last[:, None] + (last-previous)[:, None] * leads


def polynomial_basis(horizon):
    lead = np.arange(1, horizon + 1) / horizon
    return np.stack((lead, lead**2, lead**3), axis=0)


def baseline(history, horizon, name, target_idx=2):
    if name == "persistence":
        return np.repeat(history[:, -1, target_idx, None], horizon, axis=1)
    forecast = trend(history, horizon, target_idx)
    if name == "linear_drift":
        return forecast
    if name == "quadratic":
        # Quadratic continuation through the last three observations.
        second = history[:, -1, target_idx] - 2*history[:, -2, target_idx] + history[:, -3, target_idx]
        leads = np.arange(1, horizon + 1)
        return forecast + .5*second[:, None]*leads*(leads+1)
    raise ValueError(f"Unknown baseline {name}.")


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray
    residual_scale: float

    @classmethod
    def fit(cls, x, y, target_idx=2):
        flattened = x.reshape(-1, x.shape[-1])
        mean, std = flattened.mean(axis=0), flattened.std(axis=0)
        std[std < 1e-8] = 1
        residual_scale = max(float(np.sqrt(np.mean((y-trend(x, y.shape[1], target_idx))**2))), 1e-6)
        return cls(mean, std, residual_scale)

    def transform(self, x):
        return ((x-self.mean)/self.std).astype(np.float32)

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(),
                "residual_scale": self.residual_scale}

    @classmethod
    def from_dict(cls, data):
        return cls(np.asarray(data["mean"]), np.asarray(data["std"]), float(data["residual_scale"]))


def summary_features(x):
    """Observable current state and changes over the history window."""
    return np.concatenate((x[:, -1], x[:, -1]-x[:, 0],
                           x[:, -1]-2*x[:, x.shape[1]//2]+x[:, 0]), axis=1)


class DirectNetwork(nn.Module):
    """LSTM or MLP predicts three coefficients of a smooth horizon correction.

    Additive level/trend anchoring removes the need to learn absolute mAb
    levels, and direct forecasts avoid autoregressive error accumulation.
    The LSTM also sees current-state features through a skip connection.
    """

    def __init__(self, kind, features=11, hidden=64):
        super().__init__()
        self.kind, self.features, self.hidden = kind, features, hidden
        if kind == "lstm":
            self.encoder = nn.LSTM(features, hidden, num_layers=2,
                                   batch_first=True, dropout=.1)
            inputs = hidden + features
        elif kind == "mlp":
            inputs = features * 3
        else:
            raise ValueError("Network kind must be lstm or mlp.")
        self.head = nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        if self.kind == "lstm":
            sequence, _ = self.encoder(x)
            encoded = torch.cat((sequence[:, -1], x[:, -1]), dim=1)
        else:
            middle = x.shape[1]//2
            encoded = torch.cat((x[:, -1], x[:, -1]-x[:, 0],
                                 x[:, -1]-2*x[:, middle]+x[:, 0]), dim=1)
        return self.head(encoded)


def fit_ridge(x, y, norm, alpha, target_idx=2):
    design = summary_features(norm.transform(x)).astype(float)
    design = np.column_stack((np.ones(len(design)), design))
    targets = (y-trend(x, y.shape[1], target_idx)) / norm.residual_scale
    coefficients = np.linalg.lstsq(polynomial_basis(y.shape[1]).T, targets.T, rcond=None)[0].T
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[0, 0] = 0
    return np.linalg.solve(design.T@design + regularizer, design.T@coefficients)


def predict_artifact(artifact, x):
    target_idx = artifact["features"].index(artifact["target"])
    """Inference requires histories and a saved model; it never runs the ODE."""
    x = np.asarray(x, dtype=float)
    window, features, horizon = artifact["window"], len(artifact["features"]), artifact["horizon"]
    if x.ndim != 3 or x.shape[1:] != (window, features) or not np.all(np.isfinite(x)):
        raise ValueError(f"Expected finite histories of shape (n, {window}, {features}).")
    kind = artifact["kind"]
    if kind == "ensemble":
        return np.mean([predict_artifact(member, x) for member in artifact["members"]], axis=0)
    if kind in {"persistence", "linear_drift", "quadratic"}:
        return baseline(x, horizon, kind, target_idx)
    norm = Normalizer.from_dict(artifact["normalizer"])
    normalized = norm.transform(x)
    if kind == "ridge":
        design = np.column_stack((np.ones(len(x)), summary_features(normalized)))
        coefficients = design @ np.asarray(artifact["weights"])
    elif kind in {"lstm", "mlp"}:
        model = DirectNetwork(kind, features, artifact["hidden"])
        model.load_state_dict(artifact["state_dict"])
        model.eval()
        with torch.no_grad():
            batches = torch.from_numpy(normalized).split(512)
            coefficients = torch.cat([model(batch) for batch in batches]).numpy()
    else:
        raise ValueError(f"Unknown artifact kind {kind}.")
    result = trend(x, horizon, target_idx) + norm.residual_scale * coefficients @ polynomial_basis(horizon)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Forecast contains nonfinite values.")
    return result
