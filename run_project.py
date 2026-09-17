"""One-command training, evaluation, forecasting, and report generation."""

import argparse
from pathlib import Path

from forecasting.experiment import run_experiment
from forecasting.prediction import forecast_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/final")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--report-only", action="store_true", help="Rebuild plots/report from saved results without retraining")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    out = Path(args.output_dir)
    if not args.report_only:
        run_experiment(out, epochs=args.epochs)
    forecast_csv(model_path=out / "best_model.pt", output_path=out / "project_forecast.csv")
    from forecasting.reporting import build_report
    build_report(out)


if __name__ == "__main__":
    main()
