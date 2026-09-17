"""Forecast mAb using the saved selected model; no retraining or ODE solve."""

import argparse
import json
from pathlib import Path

from bioreactor import Operation
from forecasting.prediction import forecast_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", default="data/example_batch.csv")
    parser.add_argument("--model", default="results/final/best_model.pt")
    parser.add_argument("--output", default="results/final/project_forecast.csv")
    parser.add_argument("--operation", help="JSON with constant Operation fields; defaults to the project's 308 K batch")
    parser.add_argument("--origin", type=float, help="Optional observed timestamp to forecast from, ignoring later rows")
    parser.add_argument("--minutes", type=int, help="Forecast duration, multiple of 20 minutes, at most 1000 for the supplied model")
    args = parser.parse_args()
    operation = Operation(**json.loads(Path(args.operation).read_text())) if args.operation else Operation()
    forecast_csv(args.history, args.model, args.output, operation, args.origin, args.minutes)


if __name__ == "__main__":
    main()
