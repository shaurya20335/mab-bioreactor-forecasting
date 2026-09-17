"""Scientific validity checks for the completed forecasting pipeline."""

from dataclasses import asdict
from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose
import torch

from bioreactor import CSV_COLUMNS, Operation
from forecasting.data import Design, FEATURES, make_windows
from forecasting.experiment import interval_radius
from forecasting.models import Normalizer, baseline, polynomial_basis, predict_artifact, trend
from forecasting.prediction import read_history
from forecasting.serialization import load_mab_results


class DirectForecastTests(unittest.TestCase):
    def setUp(self):
        self.design = Design()
        self.values = np.tile(np.arange(301.)[:, None], (1, len(FEATURES)))
        self.manifest = {"design": asdict(self.design), "batches": [
            {"id": "train_0", "split": "train", "mode": "batch"},
            {"id": "test_0", "split": "test", "mode": "batch"}]}
        self.arrays = {"train_0": self.values, "test_0": self.values+1000}

    def test_whole_batch_split_and_future_boundary(self):
        windows = make_windows(self.manifest, self.arrays, "train")
        self.assertEqual(set(windows["batch_id"]), {"train_0"})
        self.assertLessEqual(windows["origin_min"].max()+1000, 5000)
        for x, y, origin in zip(windows["x"], windows["y"], windows["origin_min"]):
            self.assertEqual(x[-1, 2], origin/20)
            self.assertEqual(y[0], origin/20+1)
            self.assertEqual(y[-1], origin/20+50)
        future = make_windows(self.manifest, self.arrays, "test", future=True)
        assert_allclose(future["origin_min"], [5000])
        self.assertEqual(future["y"][0, -1], 1300)

    def test_training_normalization_unaffected_by_test_values(self):
        train = make_windows(self.manifest, self.arrays, "train")
        norm = Normalizer.fit(train["x"], train["y"])
        self.arrays["test_0"] *= 1e9
        again = make_windows(self.manifest, self.arrays, "train")
        other = Normalizer.fit(again["x"], again["y"])
        assert_allclose(norm.mean, other.mean)
        assert_allclose(norm.std, other.std)

    def test_constant_and_quadratic_baselines(self):
        x = np.zeros((1, 20, len(FEATURES)))
        x[0, :, 2] = np.arange(20)**2
        assert_allclose(baseline(x, 50, "quadratic")[0], np.arange(20, 70)**2)
        assert_allclose(baseline(x, 50, "persistence"), 19**2)

    def test_zero_correction_is_exact_trend_and_reload_works(self):
        x = np.tile(self.values[None, :20], (2, 1, 1))
        norm = Normalizer.fit(x, trend(x, 50)+1)
        artifact = {"kind": "ridge", "features": list(FEATURES), "window": 20,
                    "horizon": 50, "target": "mab_mg_L", "normalizer": norm.to_dict(),
                    "weights": np.zeros((34, 3)).tolist()}
        with TemporaryDirectory() as directory:
            path = Path(directory)/"model.pt"
            torch.save(artifact, path)
            restored = torch.load(path, weights_only=True)
        assert_allclose(predict_artifact(restored, x), trend(x, 50))

    def test_conformal_uses_one_score_per_batch(self):
        actual = np.zeros((24, 50))
        prediction = np.repeat(np.arange(1, 13.), 2)[:, None] * np.ones((24, 50))
        ids = np.repeat(np.arange(12), 2)
        radius, scores = interval_radius(actual, prediction, ids)
        self.assertEqual(len(scores), 12)
        self.assertEqual(radius, 12)  # ceil((12+1)*.9) = 12
        with self.assertRaises(ValueError):
            interval_radius(actual[:2], prediction[:2], ids[:2])

    def test_history_resampling_and_origin_do_not_read_future_values(self):
        times = np.arange(1001.)
        values = np.column_stack((times, np.full(len(times), 5650), np.ones(len(times)),
                                  times/10, np.full(len(times), 6), np.ones(len(times)), np.full(len(times), 99)))
        artifact = {"features": list(FEATURES), "window": 20, "step_min": 20,
                    "support_min": [0]*11, "support_max": [1e6]*11}
        with TemporaryDirectory() as directory:
            path = Path(directory)/"history.csv"
            np.savetxt(path, values, delimiter=",", comments="", header=",".join(CSV_COLUMNS))
            x, origin, _ = read_history(path, artifact, origin=600)
            values[601:, 3] = 99999
            np.savetxt(path, values, delimiter=",", comments="", header=",".join(CSV_COLUMNS))
            changed, _, _ = read_history(path, artifact, origin=600)
            values[601:, 3] = np.nan
            np.savetxt(path, values, delimiter=",", comments="", header=",".join(CSV_COLUMNS))
            ignored, _, _ = read_history(path, artifact, origin=600)
            with self.assertRaises(ValueError):
                read_history(path, artifact, origin=100)
        self.assertEqual(origin, 600)
        assert_allclose(x, changed)
        assert_allclose(x, ignored)
        assert_allclose(x[0, :, 2], np.arange(220, 601, 20)/10)

    def test_bad_design_and_inference_shapes_fail(self):
        with self.assertRaises(ValueError):
            Design(step_min=0)
        with self.assertRaises(ValueError):
            Design(horizon=1000)
        artifact = {"kind": "persistence", "window": 20, "horizon": 50,
                    "target": "mab_mg_L", "features": list(FEATURES)}
        with self.assertRaises(ValueError):
            predict_artifact(artifact, np.zeros((1, 19, 11)))

    def test_old_and_generic_metrics_have_identical_report_values(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)/"metrics.json"
            path.write_text(json.dumps({"test": {"ridge": {"rmse_mg_L": .1}}, "intervals": {"radius_mg_L": .2}}))
            old = load_mab_results(path)
            path.write_text(json.dumps({"target": "mab_mg_L", "test": {"ridge": {"rmse": .1}}, "intervals": {"radius": .2}}))
            new = load_mab_results(path)
            self.assertEqual(old["test"]["ridge"]["rmse_mg_L"], new["test"]["ridge"]["rmse_mg_L"])
            self.assertEqual(old["intervals"]["radius_mg_L"], new["intervals"]["radius_mg_L"])
            path.write_text(json.dumps({"target": "glucose_g_L", "rmse": .1}))
            with self.assertRaises(ValueError):
                load_mab_results(path)

    def test_non_mab_target_uses_the_requested_history_column(self):
        artifact = {"kind": "linear_drift", "window": 20, "horizon": 50,
                    "target": "glucose_g_L", "features": list(FEATURES)}
        x = np.zeros((1, 20, len(FEATURES)))
        x[0, :, 3] = 9-np.arange(20)*.01
        x[0, :, 2] = 100+np.arange(20)
        prediction = predict_artifact(artifact, x)
        assert_allclose(prediction[0], 8.81-np.arange(1, 51)*.01)


if __name__ == "__main__":
    unittest.main()
