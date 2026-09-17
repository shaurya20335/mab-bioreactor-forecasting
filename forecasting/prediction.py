"""Load the selected model and forecast from an observed CSV, without training."""

import json
from pathlib import Path

import numpy as np
import torch

from bioreactor import CSV_COLUMNS, Operation
from .models import predict_artifact
from .serialization import TARGET_UNITS


def read_history(path, artifact, operation=Operation(), origin=None):
    with Path(path).open() as handle:
        columns = handle.readline().strip().split(",")
    if any(name not in columns for name in CSV_COLUMNS):
        raise ValueError("History CSV must have the seven columns produced by bioreactor.py.")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    data = raw[:, [columns.index(name) for name in CSV_COLUMNS]]
    if len(data) < 3 or not np.all(np.isfinite(data[:, 0])) or np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError("History must contain finite, strictly increasing times.")
    if origin is not None:
        if not np.isfinite(origin):
            raise ValueError("Origin must be finite.")
        data = data[data[:, 0] <= origin]
        if not len(data) or not np.isclose(data[-1, 0], origin, rtol=0, atol=1e-8):
            raise ValueError("The requested forecast origin must be an observed timestamp.")
    if (not np.all(np.isfinite(data)) or np.any(data[:, 1] <= 0)
            or np.any(data[:, 2:] < 0) or np.any(data[:, 6] > 100)):
        raise ValueError("Observed history has nonfinite or invalid physical states or viability outside 0–100%.")
    origin = data[-1, 0]
    wanted = origin + np.arange(1-artifact["window"], 1)*artifact["step_min"]
    indices = np.searchsorted(data[:, 0], wanted)
    if np.any(indices >= len(data)) or not np.allclose(data[indices, 0], wanted, rtol=0, atol=1e-8):
        raise ValueError(f"Need {artifact['window']} observed rows at {artifact['step_min']}-minute spacing ending at the origin; no interpolation is performed.")
    states = data[indices, 1:]
    controls = [operation.temperature, operation.inlet_flow, operation.sample_flow,
                operation.perfusion_flow, operation.feed_glucose]
    x = np.column_stack((states, np.tile(controls, (len(states), 1))))[None]
    low, high = np.asarray(artifact["support_min"]), np.asarray(artifact["support_max"])
    tolerance = .05*np.maximum(high-low, 1e-6)
    outside = (x[0].min(axis=0) < low-tolerance) | (x[0].max(axis=0) > high+tolerance)
    return x, float(origin), [name for name, flag in zip(artifact["features"], outside) if flag]


def forecast_csv(history_path="data/example_batch.csv", model_path="results/final/best_model.pt",
                 output_path="results/final/project_forecast.csv", operation=Operation(),
                 origin=None, minutes=None):
    torch.set_num_threads(1)
    artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    if artifact.get("format_version") != 1:
        raise ValueError("Unsupported forecasting checkpoint format.")
    x, origin, outside = read_history(history_path, artifact, operation, origin)
    maximum = artifact["step_min"] * artifact["horizon"]
    minutes = maximum if minutes is None else minutes
    if not isinstance(minutes, int) or minutes <= 0 or minutes > maximum or minutes % artifact["step_min"]:
        raise ValueError(f"minutes must be a positive multiple of {artifact['step_min']}, at most {maximum}.")
    prediction = predict_artifact(artifact, x)[0, :minutes//artifact["step_min"]]
    radius = artifact.get("interval_radius", artifact.get("interval_radius_mg_L"))
    times = origin + np.arange(1, len(prediction)+1)*artifact["step_min"]
    table = np.column_stack((times, prediction, np.maximum(0, prediction-radius), prediction+radius))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, table, delimiter=",", comments="", fmt="%.10g",
               header=f"time_min,predicted_{artifact['target']},lower_90,upper_90")
    metadata = {"model": artifact["selected_name"], "origin_min": origin, "minutes": minutes,
                "target": artifact["target"], "prediction_unit": TARGET_UNITS[artifact["target"]],
                "step_min": artifact["step_min"], "history": str(history_path),
                "control_assumption": artifact["control_assumption"],
                "outside_training_support": outside, "negative_predictions": int(np.sum(prediction < 0)),
                "interval_scope": "90% nominal simultaneous batch coverage under the synthetic calibration generator; not an experimental confidence interval.",
                "operation": {name: getattr(operation, name) for name in operation.__dataclass_fields__}}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Forecast saved to {path}; model={artifact['selected_name']}; final {artifact['target']}={prediction[-1]:.4f}")
    if outside:
        print("Outside training support: " + ", ".join(outside))
    return table, metadata
