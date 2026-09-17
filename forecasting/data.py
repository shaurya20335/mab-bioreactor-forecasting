"""Synthetic reactor batches and leakage-free window construction."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bioreactor import ModelDomainError, Operation, Parameters, observations, simulate


FEATURES = ("volume_ml", "viable_cells_1e6_ml", "mab_mg_L", "glucose_g_L",
            "lactate_g_L", "viability_pct", "temperature_K", "inlet_ml_min",
            "sample_ml_min", "perfusion_ml_min", "feed_glucose_g_L")
SPLIT_COUNTS = {"train": 48, "validation": 12, "calibration": 12, "test": 16}
MODES = ("batch", "fed_batch", "continuous", "retained_perfusion")


@dataclass(frozen=True)
class Design:
    seed: int = 20260916
    step_min: int = 20
    window: int = 20
    horizon: int = 50
    stride: int = 5
    development_end_min: int = 5000
    duration_min: int = 6000

    def __post_init__(self):
        for name in ("step_min", "window", "horizon", "stride", "development_end_min", "duration_min"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.window < 3:
            raise ValueError("At least three history points are required.")
        if self.duration_min % self.step_min or self.development_end_min % self.step_min:
            raise ValueError("Time bounds must be multiples of the sampling interval.")
        if self.development_end_min + self.horizon * self.step_min > self.duration_min:
            raise ValueError("Duration must include the entire future-time test horizon.")
        if (self.window + self.horizon) * self.step_min > self.development_end_min:
            raise ValueError("Insufficient development history for this window and horizon.")


def feature_matrix(states, operation):
    controls = [operation.temperature, operation.inlet_flow, operation.sample_flow,
                operation.perfusion_flow, operation.feed_glucose]
    return np.column_stack((observations(states), np.tile(controls, (len(states), 1))))


def generate_dataset(directory, design=Design(), counts=None):
    """Independent RNG stream per split; fixed kinetics, varied initial/operating conditions.

    Only physically valid complete runs are retained; all rejected attempts
    are logged. These are explicitly synthetic, not measured experimental runs.
    """
    counts = dict(SPLIT_COUNTS if counts is None else counts)
    if set(counts) != set(SPLIT_COUNTS) or any(not isinstance(n, int) or n < 1 for n in counts.values()):
        raise ValueError("Supply positive counts for train, validation, calibration, and test.")
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    times = np.arange(0, design.duration_min + 1, design.step_min, dtype=float)
    records, arrays, rejected = [], {}, []
    for split_index, (split, count) in enumerate(counts.items()):
        rng = np.random.default_rng(np.random.SeedSequence([design.seed, split_index]))
        for i in range(count):
            mode = MODES[i % len(MODES)]
            for attempt in range(100):
                flow = float(rng.uniform(.25, 1.5))
                operation = Operation(
                    inlet_flow=0 if mode == "batch" else flow,
                    sample_flow=flow if mode == "continuous" else 0,
                    perfusion_flow=flow if mode == "retained_perfusion" else 0,
                    temperature=float(rng.uniform(308, 310)),
                    feed_glucose=float(rng.uniform(12, 24)),
                )
                viable, viability = rng.uniform(.45, .85), rng.uniform(97, 100)
                initial = [rng.uniform(4800, 6500), viable, rng.uniform(0, 8),
                           rng.uniform(6, 9), rng.uniform(.06, .3), viable*(100/viability-1)]
                try:
                    states = simulate(times, initial, operation=operation)
                except ModelDomainError as exc:
                    rejected.append({"split": split, "batch": i, "attempt": attempt, "reason": str(exc)})
                    continue
                batch_id = f"{split}_{i:03d}"
                arrays[batch_id] = feature_matrix(states, operation)
                records.append({"id": batch_id, "split": split, "mode": mode,
                                "initial_state": list(map(float, initial)),
                                "operation": asdict(operation)})
                break
            else:
                raise RuntimeError(f"Could not generate a valid batch for {split}/{i}.")
        print(f"Generated {count} {split} batches", flush=True)
    np.savez_compressed(out / "batches.npz", times=times, **arrays)
    manifest = {"design": asdict(design), "features": list(FEATURES), "counts": counts,
                "kinetic_parameters": asdict(Parameters()), "batches": records,
                "rejected_attempts": rejected,
                "provenance": "Synthetic numerical solutions; constant controls and fixed kinetic parameters.",
                "batches_sha256": hashlib.sha256((out / "batches.npz").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, arrays, times


def make_windows(manifest, arrays, split, *, future=False, target_name="mab_mg_L"):
    """Each input and its future targets belong to exactly one whole-batch split."""
    design = Design(**manifest["design"])
    target_idx = FEATURES.index(target_name)
    last = design.development_end_min // design.step_min
    origins = ([last] if future else
               range(design.window, last - design.horizon + 1, design.stride))
    x, y, ids, origin_times, modes = [], [], [], [], []
    for record in manifest["batches"]:
        if record["split"] != split:
            continue
        values = arrays[record["id"]]
        for origin in origins:
            context = values[origin-design.window+1:origin+1]
            target = values[origin+1:origin+1+design.horizon, target_idx]
            if len(context) != design.window or len(target) != design.horizon:
                raise ValueError("Incomplete history or forecast target.")
            x.append(context)
            y.append(target)
            ids.append(record["id"])
            modes.append(record["mode"])
            origin_times.append(origin * design.step_min)
    if not x:
        raise ValueError(f"No windows for split {split}.")
    return {"x": np.asarray(x), "y": np.asarray(y), "batch_id": np.asarray(ids),
            "origin_min": np.asarray(origin_times), "mode": np.asarray(modes)}
