# Advanced usage

Run commands from the repository root. The standard entry points and generated
report target mAb concentration. The Python forecasting API also supports other
state variables; keep their outputs in separate experiment directories.

```python
from forecasting.experiment import run_experiment
from forecasting.prediction import forecast_csv

results = run_experiment(
    "results/glucose_experiment", epochs=150, target_name="glucose_g_L"
)
forecast, metadata = forecast_csv(
    history_path="data/example_batch.csv",
    model_path="results/glucose_experiment/best_model.pt",
    output_path="results/glucose_experiment/forecast.csv",
)
print(results["test"][results["selected_model"]]["rmse"])
print(metadata)
```

This trains a new experiment. Valid targets and units are defined in
`forecasting/serialization.py`. The final report builder specifically describes
mAb; it rejects artifacts for other targets to avoid mislabeled results.

For saved mAb inference, `python predict_mab.py --origin 4000 --minutes 240`
uses only observations up to minute 4,000. Future CSV rows are ignored. Supply
`--operation configs/operation.example.json` with settings that match the actual
history and remain constant during the horizon. Do not infer controls from a
filename. Missing compatible history timestamps are rejected.
