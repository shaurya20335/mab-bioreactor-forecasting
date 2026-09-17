# Development guide

## Environment

Use Python 3.13 and run commands from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`.
Select this environment as the kernel for notebooks.

## Validate changes

```sh
python -m unittest discover -s tests -v
python predict_mab.py --output results/local_check/forecast.csv
```

For reporting changes, run `python run_project.py --report-only` and inspect
`docs/final_report.pdf`. On a headless machine, set `MPLBACKEND=Agg`.
Report-only execution rebuilds final plots and the illustrative forecast from
saved artifacts; it does not retrain models.

## Code and results

- Keep reusable forecasting code in `forecasting/` and numerical balances in
  `bioreactor.py`. Keep the root CLI scripts small.
- Place numerical and forecasting regression tests in `tests/`.
- Preserve physical units, training-only normalization, whole-batch splits,
  validation-only model selection, and separate calibration/test evaluation.
- Keep new experiments in an ignored output directory, for example
  `python run_project.py --output-dir results/my_experiment`.
- Update reference results only as a complete, documented experiment: include
  the split manifest, model selection, checkpoints, metrics, predictions,
  and limitations. Retain weaker comparison and robustness results.
- Keep notebook outputs relevant and free of execution errors. Do not commit
  virtual environments, credentials, caches, or temporary experiments.

The source for the historical single-trajectory LSTM remains in `baselines/`;
its regenerated output is ignored because it belongs to a different task.
