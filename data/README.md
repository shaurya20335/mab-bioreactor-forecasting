# Example data

`example_batch.csv` contains **synthetic**, corrected simulator output: 5,001
observations from 0 to 5,000 minutes at one-minute intervals. Generate it with
`python bioreactor.py` from the repository root. Operating settings are given in
`configs/operation.example.json`; initial conditions are defined in `bioreactor.py`.

| CSV column | Unit |
| --- | --- |
| Time (min) | minutes |
| Volume of Mixture (ml) | mL |
| Viable Cells (10^6 cells/ml) | million cells/mL |
| mAb Concentration (mg/l) | mg/L |
| Glucose Concentration (g/l) | g/L |
| Lactate Concentration (g/l) | g/L |
| Viability (%) | percent |

The original column spellings are retained for CSV compatibility. The internal
simulator integrates dead-cell concentration; exported viability is derived.
Forecasting selects observations at 20-minute intervals without interpolation.

The separate 88-batch benchmark and split manifest are stored in
[`results/final/data/`](../results/final/data/). Neither dataset contains
independent experimental measurements.
