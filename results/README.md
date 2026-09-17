# Results included in this repository

- [Final experiment](final/README.md): the authoritative multi-batch benchmark,
  selected model, candidate checkpoints, held-out predictions, uncertainty
  calibration, and noise sensitivity results.
- `simulation/`: overview plots from the corrected mechanistic simulator.

The main forecast is [project_forecast.png](final/project_forecast.png), with
numerical predictions in [project_forecast.csv](final/project_forecast.csv).
The completed [report](../docs/final_report.pdf) explains the methodology.

All final comparison results are retained, including weaker models and noisy-input
performance, to preserve the evidence supporting model selection and limitations.
The historical single-trajectory experiment uses a different task; its generated
outputs have been removed. Recreate them with `python -m baselines.univariate_lstm`.

Ad-hoc experiments and regenerated historical outputs are ignored by Git.
Run `python run_project.py --output-dir results/my_experiment` to conduct a new
experiment without replacing the published reference results.
