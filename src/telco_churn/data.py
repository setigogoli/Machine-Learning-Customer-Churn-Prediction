"""Load and validate the IBM Telco Customer Churn sample."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEFAULT_DATA = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
SERVICE_COLUMNS = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]
NUMERIC_COLUMNS = [
    "tenure", "MonthlyCharges", "TotalCharges", "Avg_Monthly_Spend", "Service_Count",
]
CATEGORICAL_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "Tenure_Group",
]
RAW_FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]


def download_data(destination=DEFAULT_DATA):
    """Fetch the public sample, checking the exact bytes used in this experiment."""
    destination = Path(destination)
    if destination.exists():
        content = destination.read_bytes()
    else:
        with urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != DATA_SHA256:
        raise ValueError("Dataset checksum differs from the documented IBM sample.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return destination


def prepare_features(frame):
    """Create row-wise features without learning statistics from other customers."""
    missing = sorted(set(RAW_FEATURE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    result = frame.loc[:, RAW_FEATURE_COLUMNS].copy()
    for column in ["tenure", "MonthlyCharges", "TotalCharges"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (result["tenure"].dropna() < 0).any():
        raise ValueError("Tenure must be nonnegative.")
    result["SeniorCitizen"] = result["SeniorCitizen"].astype(str)
    result["Tenure_Group"] = pd.cut(
        result["tenure"], [-1, 12, 24, 48, float("inf")],
        labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"],
    ).astype(object)
    result["Avg_Monthly_Spend"] = result["TotalCharges"] / (result["tenure"] + 1)
    result["Service_Count"] = result[SERVICE_COLUMNS].eq("Yes").sum(axis=1)
    # Keep missing categorical values compatible with SimpleImputer.
    result[CATEGORICAL_COLUMNS] = result[CATEGORICAL_COLUMNS].astype(object)
    return result


def load_dataset(path=DEFAULT_DATA):
    """Apply the coursework's complete-charge-row rule and encode the outcome."""
    raw = pd.read_csv(path)
    if "Churn" not in raw or not raw["Churn"].isin(["Yes", "No"]).all():
        raise ValueError("Churn must contain only Yes and No labels.")
    if "customerID" not in raw or raw["customerID"].isna().any():
        raise ValueError("Each training row must have a customerID.")
    raw_rows = len(raw)
    clean = raw.drop_duplicates().copy()
    if clean["customerID"].duplicated().any():
        raise ValueError("Conflicting rows share the same customerID.")
    charges = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    missing_charges = int(charges.isna().sum())
    clean = clean.loc[charges.notna()].copy()
    clean["TotalCharges"] = charges.loc[clean.index]
    features = prepare_features(clean)
    target = clean["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    audit = {
        "raw_rows": raw_rows, "raw_columns": len(raw.columns),
        "duplicate_rows_removed": raw_rows - len(raw.drop_duplicates()),
        "missing_total_charges_removed": missing_charges,
        "clean_rows": len(clean), "non_churn": int(target.eq(0).sum()),
        "churn": int(target.eq(1).sum()),
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    }
    return clean, features, target, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    print(download_data(args.output))


if __name__ == "__main__":
    main()
