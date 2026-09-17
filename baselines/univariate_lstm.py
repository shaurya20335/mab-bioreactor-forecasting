"""Simulation-to-LSTM experiment matching project_proposal.pdf, pages 7–11.

Univariate mAb history -> next mAb concentration. Splits precede scaling;
test data never select the checkpoint. Metrics use original concentration units.
"""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class Config:
    window: int = 20
    hidden_size: int = 50
    dropout: float = 0.2
    epochs: int = 50
    patience: int = 10
    batch_size: int = 64
    learning_rate: float = 1e-3
    train_fraction: float = 0.64
    validation_fraction: float = 0.16
    forecast_steps: int = 1000
    seed: int = 42

    def __post_init__(self):
        for name in ("window", "hidden_size", "epochs", "patience", "batch_size", "forecast_steps"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0, 1).")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive.")
        if not (0 < self.train_fraction < 1 and 0 < self.validation_fraction < 1
                and self.train_fraction + self.validation_fraction < 1):
            raise ValueError("Train and validation fractions must leave a nonempty test set.")
        if self.window < 2:
            raise ValueError("window must be at least 2 for the drift baseline.")


@dataclass(frozen=True)
class Scale:
    minimum: float
    span: float

    def transform(self, values):
        return (np.asarray(values) - self.minimum) / self.span

    def inverse(self, values):
        return np.asarray(values) * self.span + self.minimum


def prepare_data(times, values, config):
    """Split by target time. All windows contain strictly earlier observations.

    Validation/test windows can use observed history from preceding splits,
    as appropriate for rolling one-step prediction. Recursive forecasts are
    evaluated separately without feeding in held-out observations.
    """
    times, values = np.asarray(times, dtype=float), np.asarray(values, dtype=float)
    if (times.ndim != 1 or values.shape != times.shape or times.size < 3
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(values))
            or np.any(values < 0)):
        raise ValueError("Expected finite 1-D time and nonnegative mAb series of equal length.")
    steps = np.diff(times)
    if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-7, atol=1e-9):
        raise ValueError("LSTM input must be ordered and uniformly sampled in time.")
    train_end = int(len(values) * config.train_fraction)
    validation_end = int(len(values) * (config.train_fraction + config.validation_fraction))
    if train_end <= config.window or not train_end < validation_end < len(values):
        raise ValueError("Not enough samples for the window and three chronological splits.")
    training_values = values[:train_end]
    span = float(np.ptp(training_values))
    if span == 0:
        raise ValueError("Training concentrations are constant; use a constant baseline.")
    scale = Scale(float(training_values.min()), span)
    scaled = scale.transform(values).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(scaled, config.window + 1)
    x = torch.from_numpy(windows[:, :-1].copy()).unsqueeze(-1)
    y = torch.from_numpy(windows[:, -1].copy()).unsqueeze(-1)
    indices = np.arange(config.window, len(values))
    masks = {
        "train": indices < train_end,
        "validation": (indices >= train_end) & (indices < validation_end),
        "test": indices >= validation_end,
    }
    return x, y, indices, masks, scale, train_end, validation_end


class MAbLSTM(nn.Module):
    """Report architecture: LSTM(50) -> dropout -> LSTM(50) -> dropout -> dense."""

    def __init__(self, config=Config()):
        super().__init__()
        self.lstm1 = nn.LSTM(1, config.hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(config.dropout)
        self.lstm2 = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, 1)

    def forward(self, x):
        sequence, _ = self.lstm1(x)
        sequence, _ = self.lstm2(self.dropout1(sequence))
        return self.output(self.dropout2(sequence[:, -1, :]))


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(batch) for batch in x.split(512)]).numpy().ravel()


def recursive_forecast(model, history, scale, steps, window):
    """Feed predictions back as inputs; history ends at the forecast origin."""
    if steps < 1 or len(history) < window:
        raise ValueError("Forecast requires positive steps and at least one history window.")
    context = scale.transform(np.asarray(history)[-window:]).astype(np.float32).copy()
    result = np.empty(steps, dtype=float)
    model.eval()
    with torch.no_grad():
        for i in range(steps):
            value = model(torch.from_numpy(context.copy()).reshape(1, window, 1)).item()
            if not np.isfinite(value):
                raise RuntimeError("Recursive forecast became nonfinite.")
            result[i] = value
            context[:-1], context[-1] = context[1:].copy(), value
    return scale.inverse(result)


def error_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("Metric inputs must have identical, nonempty shapes.")
    error = predicted - actual
    if not np.all(np.isfinite(error)):
        raise ValueError("Cannot score nonfinite predictions.")
    return {"mse_mg2_L2": float(np.mean(error ** 2)),
            "mae_mg_L": float(np.mean(np.abs(error))),
            "rmse_mg_L": float(np.sqrt(np.mean(error ** 2)))}


def run_experiment(csv_path="data/example_batch.csv", output_dir="results/baselines/univariate_lstm", config=Config()):
    source, out = Path(csv_path), Path(output_dir)
    with source.open() as handle:
        columns = handle.readline().strip().split(",")
    if "Time (min)" not in columns or "mAb Concentration (mg/l)" not in columns:
        raise ValueError("CSV requires Time (min) and mAb Concentration (mg/l) columns.")
    data = np.loadtxt(source, delimiter=",", skiprows=1, ndmin=2)
    times = data[:, columns.index("Time (min)")]
    values = data[:, columns.index("mAb Concentration (mg/l)")]
    x, y, indices, masks, scale, train_end, validation_end = prepare_data(times, values, config)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = MAbLSTM(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_function = nn.MSELoss()
    loader = DataLoader(TensorDataset(x[masks["train"]], y[masks["train"]]),
                        batch_size=config.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(config.seed))
    history, best_loss, best_state, best_epoch, stale = [], float("inf"), None, 0, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, count = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_function(model(xb), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            count += len(xb)
        validation_prediction = predict(model, x[masks["validation"]])
        validation_loss = float(np.mean((validation_prediction - y[masks["validation"]].numpy().ravel()) ** 2))
        if not np.isfinite(validation_loss):
            raise RuntimeError("Nonfinite validation loss.")
        history.append([epoch, train_loss / count, validation_loss])
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:3d}: train MSE={train_loss/count:.6g}, "
                  f"validation MSE={validation_loss:.6g} (scaled)", flush=True)
        if stale >= config.patience:
            break
    model.load_state_dict(best_state)
    predicted = scale.inverse(predict(model, x))
    one_step = {name: error_metrics(values[indices[mask]], predicted[mask])
                for name, mask in masks.items()}
    test_indices = indices[masks["test"]]
    actual_test = values[test_indices]
    persistence = values[test_indices - 1]
    drift = persistence + (persistence - values[test_indices - 2])
    recursive_test = recursive_forecast(model, values[:validation_end], scale,
                                       len(actual_test), config.window)
    future = recursive_forecast(model, values, scale, config.forecast_steps, config.window)
    future_times = times[-1] + np.arange(1, config.forecast_steps + 1) * (times[1] - times[0])
    origin_value = values[validation_end - 1]
    origin_slope = origin_value - values[validation_end - 2]
    recursive_persistence = np.full(len(actual_test), origin_value)
    recursive_drift = origin_value + np.arange(1, len(actual_test) + 1) * origin_slope
    metrics = {
        "data_file": source.name,
        "data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "data_interpretation": "Synthetic trajectory for the supplied project; no experimental validation.",
        "config": asdict(config), "scale_fitted_on_training_only": asdict(scale),
        "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
        "best_epoch": best_epoch, "epochs_run": len(history),
        "sampling_interval_min": float(times[1] - times[0]),
        "split_target_times_min": {
            name: [float(times[indices[mask][0]]), float(times[indices[mask][-1]])]
            for name, mask in masks.items()
        },
        "one_step_lstm": one_step,
        "one_step_test_baselines": {
            "persistence": error_metrics(actual_test, persistence),
            "linear_drift": error_metrics(actual_test, drift),
        },
        "recursive_test": {
            "lstm": error_metrics(actual_test, recursive_test),
            "persistence": error_metrics(actual_test, recursive_persistence),
            "linear_drift": error_metrics(actual_test, recursive_drift),
        },
        "future_forecast": {"steps": config.forecast_steps,
                            "negative_predictions": int(np.sum(future < 0)),
                            "validation": "No observations beyond the CSV; not scored."},
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    torch.save({"model_state": best_state, "config": asdict(config), "scale": asdict(scale),
                "sampling_interval_min": metrics["sampling_interval_min"],
                "target": "mAb Concentration (mg/l)", "data_sha256": metrics["data_sha256"]},
               out / "lstm_checkpoint.pt")
    np.savetxt(out / "loss.csv", history, delimiter=",", comments="",
               header="epoch,training_scaled_mse,validation_scaled_mse")
    split_names = np.array(["train"] * len(indices), dtype=object)
    split_names[masks["validation"]], split_names[masks["test"]] = "validation", "test"
    import csv
    with (out / "one_step_predictions.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_min", "split", "actual_mg_L", "lstm_mg_L", "persistence_mg_L", "linear_drift_mg_L"])
        for idx, name, prediction in zip(indices, split_names, predicted):
            writer.writerow([times[idx], name, values[idx], prediction,
                             values[idx-1], 2 * values[idx-1] - values[idx-2]])
    np.savetxt(out / "recursive_test.csv",
               np.column_stack((times[test_indices], actual_test, recursive_test, recursive_persistence, recursive_drift)),
               delimiter=",", comments="", header="time_min,actual_mg_L,lstm_mg_L,persistence_mg_L,linear_drift_mg_L")
    np.savetxt(out / "future_forecast.csv", np.column_stack((future_times, future)),
               delimiter=",", comments="", header="time_min,lstm_mg_L")
    plot_results(out, times, values, indices, predicted, masks, history,
                 recursive_test, recursive_drift, future_times, future)
    print(f"Best epoch: {best_epoch}; test one-step RMSE: {one_step['test']['rmse_mg_L']:.6g} mg/L")
    print(f"Persistence RMSE: {metrics['one_step_test_baselines']['persistence']['rmse_mg_L']:.6g} mg/L")
    print(f"Saved experiment to {out}")
    return metrics


def plot_results(out, times, values, indices, predicted, masks, history,
                 recursive_test, recursive_drift, future_times, future):
    import matplotlib.pyplot as plt

    history = np.asarray(history)
    test_indices = indices[masks["test"]]
    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
        ax = axes[0, 0]
        ax.plot(history[:, 0], history[:, 1], label="Training")
        ax.plot(history[:, 0], history[:, 2], label="Validation")
        ax.set(xlabel="Epoch", ylabel="MSE (scaled)", title="Training and validation loss")
        ax.legend()
        ax = axes[0, 1]
        ax.plot(times, values, color="black", label="Simulated mAb")
        for name, mask in masks.items():
            ax.plot(times[indices[mask]], predicted[mask], label=f"LSTM {name}")
        ax.set(xlabel="Time (min)", ylabel="mAb (mg/L)", title="Rolling one-step predictions")
        ax.legend()
        ax = axes[1, 0]
        ax.plot(times[test_indices], values[test_indices], color="black", label="Simulated mAb")
        ax.plot(times[test_indices], recursive_test, label="Recursive LSTM")
        ax.plot(times[test_indices], recursive_drift, linestyle="--", label="Linear drift baseline")
        ax.set(xlabel="Time (min)", ylabel="mAb (mg/L)", title="Held-out forecast: no observed test inputs")
        ax.legend()
        ax = axes[1, 1]
        ax.hist(predicted[masks["test"]] - values[test_indices], bins=30, color="#725495")
        ax.set(xlabel="Prediction minus simulation (mg/L)", ylabel="Count", title="One-step test errors")
        for ax in axes.flat:
            ax.grid(alpha=.2)
        fig.savefig(out / "evaluation.png", dpi=180)
        fig.savefig(out / "evaluation.pdf")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        ax.plot(times, values, label="Simulated history")
        ax.plot(np.r_[times[-1], future_times], np.r_[values[-1], future], label="Recursive LSTM forecast")
        ax.set(xlabel="Time (min)", ylabel="mAb (mg/L)", title="Future forecast — extrapolation without observations")
        ax.grid(alpha=.2)
        ax.legend()
        fig.savefig(out / "future_forecast.png", dpi=180)
        fig.savefig(out / "future_forecast.pdf")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/example_batch.csv")
    parser.add_argument("--output-dir", default="results/baselines/univariate_lstm")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--forecast-steps", type=int, default=1000)
    args = parser.parse_args()
    run_experiment(args.csv, args.output_dir,
                   Config(epochs=args.epochs, forecast_steps=args.forecast_steps))
