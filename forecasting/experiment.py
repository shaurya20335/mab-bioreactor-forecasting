"""Train on whole batches, select on validation, calibrate, then unlock testing."""

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import Design, FEATURES, generate_dataset, make_windows
from .models import (DirectNetwork, Normalizer, baseline, fit_ridge,
                     polynomial_basis, predict_artifact, trend)
from .serialization import TARGET_UNITS


def metrics(actual, prediction):
    actual, prediction = np.asarray(actual), np.asarray(prediction)
    if actual.shape != prediction.shape or not actual.size:
        raise ValueError("Scoring requires equally shaped, nonempty actual and predicted arrays.")
    error = prediction - actual
    if not np.all(np.isfinite(error)):
        raise ValueError("Nonfinite errors cannot be scored.")
    return {"rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "mse": float(np.mean(error**2)),
            "bias": float(np.mean(error))}


def score_windows(windows, prediction, step_min):
    result = metrics(windows["y"], prediction)
    result["by_lead"] = {str(lead): metrics(windows["y"][:, lead//step_min-1],
                                         prediction[:, lead//step_min-1])
                         for lead in (60, 240, 1000)
                         if lead % step_min == 0 and lead//step_min <= prediction.shape[1]}
    result["by_mode"] = {mode: metrics(windows["y"][windows["mode"] == mode],
                                       prediction[windows["mode"] == mode])
                         for mode in np.unique(windows["mode"])}
    per_batch = [metrics(windows["y"][windows["batch_id"] == name],
                         prediction[windows["batch_id"] == name])["rmse"]
                 for name in np.unique(windows["batch_id"])]
    rng = np.random.default_rng(2026)
    bootstrap = rng.choice(per_batch, (2000, len(per_batch)), replace=True).mean(axis=1)
    result["mean_batch_rmse"] = float(np.mean(per_batch))
    result["mean_batch_rmse_bootstrap_95pct"] = np.quantile(bootstrap, [.025, .975]).tolist()
    result["n_batches"], result["n_windows"] = len(per_batch), len(prediction)
    return result


def train_network(kind, train, validation, norm, base, seed, epochs=150, patience=25):
    torch.manual_seed(seed)
    target_idx = base["features"].index(base["target"])
    model = DirectNetwork(kind, hidden=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=.5)
    x = torch.from_numpy(norm.transform(train["x"]))
    target = torch.tensor((train["y"]-trend(train["x"], base["horizon"], target_idx)) / norm.residual_scale,
                          dtype=torch.float32)
    val_x = torch.from_numpy(norm.transform(validation["x"]))
    val_target = torch.tensor((validation["y"]-trend(validation["x"], base["horizon"], target_idx)) / norm.residual_scale,
                              dtype=torch.float32)
    basis = torch.tensor(polynomial_basis(base["horizon"]), dtype=torch.float32)
    loader = DataLoader(TensorDataset(x, target), batch_size=128, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    best, best_epoch, best_state, stale, history = float("inf"), 0, None, 0, []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(xb) @ basis, yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            val_prediction = torch.cat([model(batch) @ basis for batch in val_x.split(512)])
            val_loss = nn.functional.mse_loss(val_prediction, val_target).item()
        if not math.isfinite(val_loss):
            raise RuntimeError("Nonfinite validation loss.")
        history.append([epoch, total/len(x)*norm.residual_scale**2,
                        val_loss*norm.residual_scale**2])
        scheduler.step(val_loss)
        if val_loss < best:
            best, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 25 == 0:
            print(f"{kind} seed={seed} epoch={epoch}: validation RMSE={math.sqrt(val_loss)*norm.residual_scale:.4f}", flush=True)
        if stale >= patience:
            break
    artifact = dict(base, kind=kind, normalizer=norm.to_dict(), state_dict=best_state,
                    hidden=64, seed=seed, best_epoch=best_epoch, epochs_run=epoch)
    info = {"best_epoch": best_epoch, "epochs_run": epoch,
            "training_seconds": time.perf_counter()-started, "seed": seed}
    return artifact, np.asarray(history), info


def interval_radius(actual, prediction, batch_ids, coverage=.9):
    """Split-conformal constant radius from independent whole-batch max errors.

    Windows/horizons within a batch are dependent, so each batch contributes
    ONE maximum score. Coverage concerns an entire batch of forecast paths
    drawn under the synthetic generator, not independent individual rows.
    The standard guarantee assumes exchangeability; balanced-mode evaluation
    reports empirical coverage without claiming a guarantee for each mode.
    """
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1.")
    errors = np.abs(np.asarray(actual)-np.asarray(prediction))
    scores = np.array([errors[batch_ids == name].max() for name in np.unique(batch_ids)])
    rank = math.ceil((len(scores)+1)*coverage)
    if rank > len(scores):
        raise ValueError("Too few calibration batches for this finite-sample coverage level.")
    return float(np.sort(scores)[rank-1]), scores.tolist()


def write_prediction_table(path, windows, prediction, step_min, radius=None):
    import csv
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["batch_id", "mode", "origin_min", "lead_min", "target_min", "actual", "predicted"]
        if radius is not None:
            header += ["lower_90", "upper_90"]
        writer.writerow(header)
        for row in range(len(prediction)):
            for j, value in enumerate(prediction[row]):
                lead = (j+1)*step_min
                record = [windows["batch_id"][row], windows["mode"][row], windows["origin_min"][row],
                          lead, windows["origin_min"][row]+lead, windows["y"][row, j], value]
                if radius is not None:
                    record += [max(0., value-radius), value+radius]
                writer.writerow(record)


def run_experiment(output_dir="results/final", *, epochs=150, design=Design(), counts=None, seeds=(11, 29, 47), target_name="mab_mg_L"):
    if target_name not in TARGET_UNITS:
        raise ValueError("Target must be one of the six observed reactor states.")
    if not isinstance(epochs, int) or epochs < 1 or not seeds:
        raise ValueError("Positive epochs and at least one neural training seed are required.")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest, arrays, times = generate_dataset(out / "data", design, counts)
    train = make_windows(manifest, arrays, "train", target_name=target_name)
    validation = make_windows(manifest, arrays, "validation", target_name=target_name)
    target_idx = FEATURES.index(target_name)
    norm = Normalizer.fit(train["x"], train["y"], target_idx=target_idx)
    base = {"format_version": 1, "features": list(FEATURES), "window": design.window,
            "horizon": design.horizon, "step_min": design.step_min,
            "target": target_name, "target_unit": TARGET_UNITS[target_name],
            "dataset_sha256": manifest["batches_sha256"],
            "control_assumption": "The supplied controls remain constant throughout the forecast.",
            "support_min": train["x"].min(axis=(0, 1)).tolist(),
            "support_max": train["x"].max(axis=(0, 1)).tolist()}
    candidates, validation_scores, training_details = {}, {}, {}
    histories = {}
    for name in ("persistence", "linear_drift", "quadratic"):
        candidates[name] = dict(base, kind=name)
    ridge_scores = {}
    for alpha in (1e-6, 1e-4, .01, 1., 100.):
        candidate = dict(base, kind="ridge", alpha=alpha, normalizer=norm.to_dict(),
                         weights=fit_ridge(train["x"], train["y"], norm, alpha, target_idx=target_idx).tolist())
        score = metrics(validation["y"], predict_artifact(candidate, validation["x"]))
        ridge_scores[str(alpha)] = score
        if "ridge" not in candidates or score["rmse"] < best_ridge:
            candidates["ridge"], best_ridge = candidate, score["rmse"]
    print(f"Ridge validation RMSE: {best_ridge:.4f}", flush=True)
    for kind in ("mlp", "lstm"):
        members = []
        for seed in seeds:
            artifact, history, info = train_network(kind, train, validation, norm, base, seed, epochs=epochs)
            name = f"{kind}_seed{seed}"
            candidates[name], histories[name], training_details[name] = artifact, history, info
            members.append(artifact)
        candidates[f"{kind}_ensemble"] = dict(base, kind="ensemble", members=members, family=kind)
    for name, artifact in candidates.items():
        validation_scores[name] = score_windows(validation, predict_artifact(artifact, validation["x"]), design.step_min)
    selected_name = min(validation_scores, key=lambda name: validation_scores[name]["rmse"])
    selected = candidates[selected_name]
    selection = {"selected_model": selected_name, "target": target_name,
                 "target_unit": TARGET_UNITS[target_name],
                 "criterion": "Validation RMSE across all forecast leads and origins",
                 "validation_scores": validation_scores, "ridge_alpha_search": ridge_scores,
                 "training": training_details, "seeds": list(seeds), "max_epochs": epochs,
                 "test_used_for_selection": False}
    # This file is written BEFORE constructing or evaluating test windows.
    (out / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(f"FROZEN selection: {selected_name}; validation RMSE={validation_scores[selected_name]['rmse']:.4f}", flush=True)

    calibration = make_windows(manifest, arrays, "calibration", target_name=target_name)
    calibration_future = make_windows(manifest, arrays, "calibration", future=True, target_name=target_name)
    combined_cal = {key: np.concatenate((calibration[key], calibration_future[key])) for key in calibration}
    cal_prediction = predict_artifact(selected, combined_cal["x"])
    radius, cal_scores = interval_radius(combined_cal["y"], cal_prediction, combined_cal["batch_id"])
    selected = dict(selected, selected_name=selected_name, interval_radius=radius,
                    interval_coverage=.9, interval_unit="whole synthetic batch",
                    calibration_batch_scores=cal_scores)
    torch.save(selected, out / "best_model.pt")
    checkpoint_dir = out / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for name, artifact in candidates.items():
        torch.save(artifact, checkpoint_dir / f"{name}.pt")
    for name, history in histories.items():
        np.savetxt(out / f"{name}_loss.csv", history, delimiter=",", comments="",
                   header="epoch,train_mse,validation_mse")

    test = make_windows(manifest, arrays, "test", target_name=target_name)
    future_test = make_windows(manifest, arrays, "test", future=True, target_name=target_name)
    test_scores, future_scores = {}, {}
    for name, artifact in candidates.items():
        test_scores[name] = score_windows(test, predict_artifact(artifact, test["x"]), design.step_min)
        future_scores[name] = score_windows(future_test, predict_artifact(artifact, future_test["x"]), design.step_min)
    prediction = predict_artifact(selected, test["x"])
    future_prediction = predict_artifact(selected, future_test["x"])
    all_test = {key: np.concatenate((test[key], future_test[key])) for key in test}
    all_prediction = np.concatenate((prediction, future_prediction))
    covered = np.abs(all_prediction-all_test["y"]) <= radius
    batch_coverage = [bool(covered[all_test["batch_id"] == name].all()) for name in np.unique(all_test["batch_id"])]
    # Predeclared sensitivity test: perturb observed state channels by 0.5% of
    # TRAINING standard deviations. Controls remain known/exact. No retraining.
    noisy = test["x"].copy()
    rng = np.random.default_rng(design.seed+123)
    noisy[:, :, :6] += rng.normal(size=noisy[:, :, :6].shape) * norm.std[:6] * .005
    noisy[:, :, :6] = np.maximum(noisy[:, :, :6], 0)
    noisy[:, :, 5] = np.minimum(noisy[:, :, 5], 100)
    stress = {name: metrics(test["y"], predict_artifact(artifact, noisy)) for name, artifact in candidates.items()}
    results = {"selected_model": selected_name, "target": target_name,
               "target_unit": TARGET_UNITS[target_name],
               "metric_units": {"rmse": TARGET_UNITS[target_name], "mae": TARGET_UNITS[target_name],
                                "bias": TARGET_UNITS[target_name], "mse": f"({TARGET_UNITS[target_name]})^2"},
               "design": manifest["design"], "counts": manifest["counts"],
               "dataset_sha256": manifest["batches_sha256"], "data_kind": "synthetic",
               "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
               "validation": validation_scores, "test": test_scores, "future_time_test": future_scores,
               "measurement_noise_sensitivity": {"definition": "Independent Gaussian noise, SD=0.5% of training feature SD, applied to observed state channels only.", "scores": stress},
               "intervals": {"nominal_batch_coverage": .9, "radius": radius,
                             "test_batch_coverage": float(np.mean(batch_coverage)),
                             "test_point_coverage": float(covered.mean()), "calibration_batches": len(cal_scores)},
               "negative_selected_predictions": int(np.sum(all_prediction < 0)),
               "scope": "Fixed simulator kinetics, constant operating conditions, synthetic validation only."}
    (out / "metrics.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    write_prediction_table(out / "test_predictions.csv", test, prediction, design.step_min, radius)
    write_prediction_table(out / "future_time_test_predictions.csv", future_test, future_prediction, design.step_min, radius)
    np.savez_compressed(out / "plot_data.npz", test_actual=test["y"], test_prediction=prediction,
                        test_origins=test["origin_min"], test_modes=test["mode"], test_batch_ids=test["batch_id"],
                        future_actual=future_test["y"], future_prediction=future_prediction,
                        future_history=future_test["x"], future_modes=future_test["mode"])
    print(f"Selected test RMSE={test_scores[selected_name]['rmse']:.4f}; "
          f"future-time RMSE={future_scores[selected_name]['rmse']:.4f}", flush=True)
    return results
