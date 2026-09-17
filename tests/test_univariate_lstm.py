"""Forecasting checks: time ordering, scale leakage, recursion, and metric units."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from numpy.testing import assert_allclose

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from baselines.univariate_lstm import Config, MAbLSTM, Scale, error_metrics, prepare_data, recursive_forecast


@unittest.skipUnless(HAS_TORCH, "Install requirements.txt for forecasting tests")
class ForecastTests(unittest.TestCase):
    def test_windows_have_only_past_values_and_disjoint_target_times(self):
        times = np.arange(100)
        x, y, indices, masks, scale, train_end, validation_end = prepare_data(times, times * 2, Config())
        self.assertEqual((train_end, validation_end), (64, 80))
        for row, idx in enumerate(indices):
            assert_allclose(scale.inverse(x[row, :, 0].numpy()), 2 * times[idx-20:idx], atol=1e-5)
            assert_allclose(scale.inverse(y[row].numpy()), [2 * times[idx]], atol=1e-5)
        self.assertTrue(np.all(indices[masks["train"]] < 64))
        self.assertTrue(np.all((indices[masks["validation"]] >= 64) & (indices[masks["validation"]] < 80)))
        self.assertTrue(np.all(indices[masks["test"]] >= 80))
        self.assertTrue(np.all(sum(mask.astype(int) for mask in masks.values()) == 1))

    def test_scaling_ignores_validation_and_test_extremes(self):
        times = np.arange(100)
        values = times.astype(float)
        first = prepare_data(times, values, Config())
        values[64:] = 1000000
        second = prepare_data(times, values, Config())
        self.assertEqual(first[4], second[4])
        assert_allclose(first[0][first[3]["train"]], second[0][second[3]["train"]])
        self.assertEqual(first[4].span, 63)

    def test_recursive_forecast_feeds_back_predictions(self):
        class NextStep(torch.nn.Module):
            def forward(self, x):
                return x[:, -1, :] + 1
        values = recursive_forecast(NextStep(), [0., 10., 20.], Scale(0, 10), 3, 2)
        assert_allclose(values, [30, 40, 50])

    def test_metrics_are_in_original_units(self):
        metrics = error_metrics(np.array([1., 4.]), np.array([3., 2.]))
        self.assertEqual(metrics, {"mse_mg2_L2": 4., "mae_mg_L": 2., "rmse_mg_L": 2.})
        with self.assertRaises(ValueError):
            error_metrics(np.array([1, 2]), np.array([[1, 2]]))

    def test_checkpoint_roundtrip_preserves_predictions(self):
        config = Config()
        model = MAbLSTM(config).eval()
        x = torch.rand(3, 20, 1)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(model.state_dict(), path)
            restored = MAbLSTM(config).eval()
            restored.load_state_dict(torch.load(path, weights_only=True))
        with torch.no_grad():
            self.assertEqual(model(x).shape, (3, 1))
            assert_allclose(model(x), restored(x))

    def test_bad_data_and_settings_are_rejected(self):
        for times, values in (([0, 2, 1], [1, 2, 3]),
                              (np.arange(100), np.ones(100)),
                              (np.arange(100), np.full(100, np.nan)),
                              (np.arange(100) ** 2, np.arange(100))):
            with self.subTest(times=times), self.assertRaises(ValueError):
                prepare_data(times, values, Config())
        with self.assertRaises(ValueError):
            Config(train_fraction=.9, validation_fraction=.2)
        with self.assertRaises(ValueError):
            Config(epochs=0)


if __name__ == "__main__":
    unittest.main()
