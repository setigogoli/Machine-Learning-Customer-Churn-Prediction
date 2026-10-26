"""Score a CSV with the trained classifier and calibrated scenario model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import prepare_features
from .retention import retention_scenario


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/churn_model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions.csv"))
    args = parser.parse_args()
    # Load only model files created by this project or another trusted source.
    bundle = joblib.load(args.model)
    raw = pd.read_csv(args.input)
    features = prepare_features(raw)
    result, _ = retention_scenario(bundle["calibrated_scenario_model"].predict_proba(features)[:, 1])
    result.insert(0, "predicted_churn", bundle["classifier"].predict(features))
    result.insert(1, "classifier_score", bundle["classifier"].predict_proba(features)[:, 1])
    if "customerID" in raw:
        result.insert(0, "customerID", raw["customerID"].to_numpy())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {len(result)} predictions to {args.output}")


if __name__ == "__main__":
    main()
