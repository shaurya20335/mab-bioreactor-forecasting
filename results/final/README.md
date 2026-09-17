# Completed multi-batch forecasting experiment

Reproduce with `python run_project.py` from the project root. Use
`python run_project.py --report-only` to rebuild the report without training.

The selected model is **ridge regression with a cubic correction to the
extrapolated local trend**. It was selected by validation RMSE before testing.

- `data/manifest.json`: split assignments, all 88 initial conditions and
  constant operations, kinetic parameters, rejected attempts, and data hash.
- `data/batches.npz`: complete simulated trajectories, with separate batch IDs.
- `selection.json`: validation results, ridge search, neural training details,
  and the frozen selection decision.
- `best_model.pt`: selected predictor, scaling, input schema, support ranges,
  and independently calibrated interval radius.
- `checkpoints/`: all tested predictors, including trained LSTMs and MLPs.
- `metrics.json`: validation, unseen-batch test, later-time test, mode/lead
  breakdowns, batch bootstrap intervals, calibration coverage, and noise test.
- `test_predictions.csv`: selected-model predictions on unseen batches.
- `future_time_test_predictions.csv`: unseen batches forecast from 5,000 to
  6,000 minutes, beyond the time range of every training target.
- `project_forecast.csv` / `.json`: saved-model inference for the original
  project history, with interval bounds, assumptions, and support warnings.
- `project_reference.csv`: independent ODE continuation for illustration only;
  inference does not use this future reference.
- `*_loss.csv`: neural training/validation MSE in original concentration units.
- PNG/PDF figures: model comparisons, forecasts, errors, and noise sensitivity.

Clean test RMSE is 0.00521 mg/L; later-time test RMSE is 0.04855 mg/L.
The illustrative noise stress test raises selected-model RMSE to 6.80 mg/L.
Calibration bands apply to the synthetic generating distribution, not noisy
experimental measurements. No clinical or industrial validation is claimed.

The original single-trajectory univariate experiment can be regenerated with
`python -m baselines.univariate_lstm`. Its obsolete generated outputs are not
included. Its metrics are not directly comparable because its task and sampling differ.
