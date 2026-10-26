"""Run model comparison, tuning, holdout evaluation and a retention scenario."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from joblib import Memory
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score,
    recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .data import DEFAULT_DATA, load_dataset
from .features import CombinedFeatureSelector, make_preprocessor
from .retention import retention_scenario

SCORING = {"recall": "recall", "precision": "precision", "f1": "f1",
           "roc_auc": "roc_auc", "average_precision": "average_precision"}


def make_pipeline(model, smote=False, cache_dir=None):
    return Pipeline([
        ("preprocess", make_preprocessor()),
        ("select", CombinedFeatureSelector(k=12, random_state=42)),
        ("resample", SMOTE(random_state=42) if smote else "passthrough"),
        ("model", model),
    ], memory=Memory(cache_dir, verbose=0) if cache_dir else None)


def candidate_pipelines(cache_dir=None):
    lr = LogisticRegression(max_iter=3000, random_state=42)
    rf = RandomForestClassifier(
        n_estimators=200, max_leaf_nodes=30, max_features="sqrt", random_state=42, n_jobs=1,
    )
    gb = GradientBoostingClassifier(random_state=42)
    vote = VotingClassifier([
        ("lr", clone(lr)), ("rf", clone(rf)), ("gb", clone(gb)),
    ], voting="soft")
    return {
        "Logistic regression": make_pipeline(clone(lr), cache_dir=cache_dir),
        "Logistic regression (balanced)": make_pipeline(
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42),
            cache_dir=cache_dir,
        ),
        "Logistic regression + SMOTE": make_pipeline(clone(lr), True, cache_dir),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=11), cache_dir=cache_dir),
        "Random forest": make_pipeline(rf, cache_dir=cache_dir),
        "Gradient boosting": make_pipeline(gb, cache_dir=cache_dir),
        "Soft voting": make_pipeline(vote, cache_dir=cache_dir),
    }


def selection_index(results):
    """Maximize mean CV recall subject to mean CV F1 >= 0.50."""
    recall = np.asarray(results["mean_test_recall"])
    f1 = np.asarray(results["mean_test_f1"])
    eligible = np.flatnonzero(f1 >= 0.50)
    if not len(eligible):
        raise ValueError("No candidate meets the mean cross-validation F1 threshold of 0.50.")
    return int(sorted(eligible, key=lambda i: (-recall[i], -f1[i], i))[0])


def compare_models(X_train, y_train, cache_dir=None, n_jobs=1):
    """Use only the development split for all candidate and parameter decisions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = candidate_pipelines(cache_dir)
    rows, fold_rows, searches = [], [], {}
    for name, estimator in candidates.items():
        print(f"Cross-validation: {name}", flush=True)
        scores = cross_validate(
            estimator, X_train, y_train, scoring=SCORING, cv=cv,
            n_jobs=n_jobs, error_score="raise",
        )
        row = {"model": name}
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}"] = float(values.mean())
            row[f"cv_{metric}_std"] = float(values.std(ddof=0))
        rows.append(row)
        for fold in range(5):
            fold_rows.append({"model": name, "fold": fold + 1,
                              **{m: float(scores[f"test_{m}"][fold]) for m in SCORING}})

    grids = {
        "Tuned logistic regression + SMOTE": (
            make_pipeline(LogisticRegression(max_iter=3000, random_state=42), True, cache_dir),
            [{"model__C": [0.01, 0.1, 1.0], "model__solver": ["lbfgs"], "model__l1_ratio": [0.0]},
             {"model__C": [0.01, 0.1, 1.0], "model__solver": ["liblinear"], "model__l1_ratio": [1.0]}],
        ),
        "Tuned random forest + SMOTE": (
            make_pipeline(RandomForestClassifier(random_state=42, n_jobs=1), True, cache_dir),
            {"model__n_estimators": [100, 200], "model__max_depth": [4, 8],
             "model__max_features": ["sqrt"], "model__criterion": ["gini", "entropy"]},
        ),
    }
    for name, (pipeline, grid) in grids.items():
        print(f"Grid search: {name}", flush=True)
        search = GridSearchCV(
            pipeline, grid, scoring=SCORING, refit=selection_index,
            cv=cv, n_jobs=n_jobs, error_score="raise",
        ).fit(X_train, y_train)
        index = search.best_index_
        row = {"model": name}
        for metric in SCORING:
            row[f"cv_{metric}"] = float(search.cv_results_[f"mean_test_{metric}"][index])
            row[f"cv_{metric}_std"] = float(search.cv_results_[f"std_test_{metric}"][index])
        rows.append(row)
        for fold in range(5):
            fold_rows.append({"model": name, "fold": fold + 1,
                              **{m: float(search.cv_results_[f"split{fold}_test_{m}"][index]) for m in SCORING}})
        candidates[name] = search.best_estimator_
        searches[name] = {"best_parameters": search.best_params_, "combinations": len(search.cv_results_["params"])}

    comparison = pd.DataFrame(rows)
    eligible = comparison.loc[comparison.cv_f1 >= 0.50].sort_values(
        ["cv_recall", "cv_f1", "model"], ascending=[False, False, True],
    )
    if eligible.empty:
        raise ValueError("No model meets the CV F1 constraint.")
    selected_name = eligible.iloc[0]["model"]
    selected = clone(candidates[selected_name]).fit(X_train, y_train)
    comparison["selected"] = comparison.model.eq(selected_name)
    comparison = comparison.sort_values("cv_recall", ascending=False).reset_index(drop=True)
    return selected, selected_name, comparison, pd.DataFrame(fold_rows), searches


def metric_summary(y_true, prediction, probability):
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "confusion_matrix": confusion_matrix(y_true, prediction, labels=[0, 1]).tolist(),
    }


def save_eda(training_frame, destination):
    """Plot development-set patterns, leaving the holdout out of exploration."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    counts = training_frame.Churn.value_counts().reindex(["No", "Yes"])
    axes[0].bar(["Stayed", "Churned"], counts, color=["#497899", "#ce7856"])
    axes[0].set(title="Training class counts", ylabel="Customers")
    rate = training_frame.assign(churn=training_frame.Churn.eq("Yes")).groupby("Contract").churn.mean()
    rate.reindex(["Month-to-month", "One year", "Two year"]).mul(100).plot.bar(ax=axes[1], color="#497899", rot=20)
    axes[1].set(title="Churn rate by contract", ylabel="Churn (%)", xlabel="")
    for label, color in [("No", "#497899"), ("Yes", "#ce7856")]:
        axes[2].hist(training_frame.loc[training_frame.Churn.eq(label), "tenure"],
                     bins=18, density=True, histtype="step", linewidth=2, label=label, color=color)
    axes[2].set(title="Tenure by outcome", xlabel="Tenure (months)", ylabel="Density")
    axes[2].legend(title="Churn")
    fig.tight_layout()
    fig.savefig(destination / "training_data_overview.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def evaluate_and_save(selected, selected_name, comparison, folds, searches,
                      X_train, X_test, y_train, y_test, audit, output_dir):
    output = Path(output_dir)
    figures, results = output / "figures", output / "results"
    figures.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    print(f"Selected using training CV: {selected_name}", flush=True)
    prediction = selected.predict(X_test)
    probability = selected.predict_proba(X_test)[:, 1]
    test_metrics = metric_summary(y_test, prediction, probability)
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    dummy_metrics = metric_summary(y_test, dummy.predict(X_test), dummy.predict_proba(X_test)[:, 1])

    # Calibration folds retain natural class prevalence and never see the holdout.
    calibrated = CalibratedClassifierCV(
        clone(selected), method="sigmoid",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=43), ensemble=True, n_jobs=1,
    ).fit(X_train, y_train)
    calibrated_probability = calibrated.predict_proba(X_test)[:, 1]
    calibration_metrics = {
        "uncalibrated_brier": float(brier_score_loss(y_test, probability)),
        "calibrated_brier": float(brier_score_loss(y_test, calibrated_probability)),
        "calibrated_roc_auc": float(roc_auc_score(y_test, calibrated_probability)),
    }
    decisions, totals = retention_scenario(calibrated_probability)
    # Row identifiers, features and customer-level predictions stay out of the published results.
    action_counts = decisions.action.value_counts().reindex(["none", "low", "high"], fill_value=0)
    ranking = selected.named_steps["select"].ranking_frame(
        selected.named_steps["preprocess"].get_feature_names_out(),
    )
    comparison.to_csv(results / "cross_validation.csv", index=False)
    folds.to_csv(results / "cross_validation_folds.csv", index=False)
    ranking.to_csv(results / "feature_ranking.csv", index=False)
    totals.to_csv(results / "retention_scenario.csv", index=False)
    action_counts.rename_axis("action").to_csv(results / "retention_action_counts.csv", header=["customers"])
    pd.DataFrame(classification_report(y_test, prediction, output_dict=True, zero_division=0)).T.to_csv(
        results / "classification_report.csv",
    )
    split_records = [{"source_row": int(i), "partition": label}
                     for label, subset in [("train", X_train), ("test", X_test)] for i in subset.index]
    pd.DataFrame(split_records).sort_values("source_row").to_csv(results / "split_indices.csv", index=False)
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "scikit-learn", "imbalanced-learn", "matplotlib", "joblib",
    ]}
    summary = {
        "dataset": audit, "train_rows": len(X_train), "test_rows": len(X_test),
        "test_churn": int(y_test.sum()), "split_seed": 40, "cv_seed": 42,
        "selected_model": selected_name, "decision_threshold": 0.5,
        "selection_rule": "Highest mean 5-fold training CV recall with mean F1 >= 0.50; tie-break by F1, then name.",
        "test_metrics": test_metrics, "dummy_test_metrics": dummy_metrics,
        "calibration": calibration_metrics, "tuning": searches,
        "retention_assumptions": {"loss": 500, "low_cost": 50, "high_cost": 100,
                                  "low_success": 0.30, "high_success": 0.60},
        "retention_action_counts": {k: int(v) for k, v in action_counts.items()},
        "python_version": platform.python_version(), "package_versions": versions,
    }
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    model_dir = output.parent / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump({"classifier": selected, "calibrated_scenario_model": calibrated,
                 "selected_name": selected_name}, model_dir / "churn_model.joblib")

    fig, ax = plt.subplots(figsize=(9, 5.4))
    view = comparison.sort_values("cv_recall")
    ax.barh(view.model, view.cv_recall, xerr=view.cv_recall_std,
            color=["#ce7856" if s else "#497899" for s in view.selected], capsize=3)
    ax.set(xlabel="Mean recall (error bars: fold standard deviation)", xlim=(0, 1), title="Training cross-validation")
    fig.tight_layout()
    fig.savefig(figures / "cross_validation_recall.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    RocCurveDisplay.from_predictions(y_test, probability, name="Selected classifier", ax=axes[0], plot_chance_level=True)
    PrecisionRecallDisplay.from_predictions(y_test, probability, name="Selected classifier", ax=axes[1], plot_chance_level=True)
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, display_labels=["Stayed", "Churned"],
                                           ax=axes[2], colorbar=False, cmap="Blues")
    for ax, title in zip(axes, ["Test ROC", "Test precision-recall", "Test confusion matrix"]):
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figures / "holdout_evaluation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    top = ranking.head(12).iloc[::-1]
    ax.barh(top.feature, top.average_rank, color="#497899")
    ax.set(xlabel="Mean rank across filter, Lasso and forest (lower is better)", title="Selected training features")
    fig.tight_layout()
    fig.savefig(figures / "feature_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    CalibrationDisplay.from_predictions(y_test, probability, n_bins=8, strategy="quantile", name="Classifier", ax=axes[0])
    CalibrationDisplay.from_predictions(y_test, calibrated_probability, n_bins=8, strategy="quantile", name="Sigmoid calibration", ax=axes[0])
    axes[0].set_title("Holdout calibration check")
    axes[1].hist(calibrated_probability, bins=25, color="#497899")
    axes[1].set(title="Calibrated test scores", xlabel="Estimated churn probability", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(figures / "probability_calibration.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(totals.strategy, totals.expected_loss / 1000, color="#497899")
    axes[0].set(title="Illustrative expected loss", ylabel="USD (thousands)")
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(action_counts.index, action_counts, color=["#497899", "#a7bba6", "#ce7856"])
    axes[1].set(title="Actions under the stated assumptions", ylabel="Customers")
    fig.tight_layout()
    fig.savefig(figures / "retention_scenario.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return summary


def run_experiment(data_path=DEFAULT_DATA, output_dir="reports", n_jobs=1):
    started = time.perf_counter()
    clean, X, y, audit = load_dataset(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y,
    )
    save_eda(clean.loc[X_train.index], Path(output_dir) / "figures")
    selected, name, comparison, folds, searches = compare_models(
        X_train, y_train, cache_dir=".cache/pipeline", n_jobs=n_jobs,
    )
    result = evaluate_and_save(selected, name, comparison, folds, searches,
                               X_train, X_test, y_train, y_test, audit, output_dir)
    print(f"Test recall: {result['test_metrics']['recall']:.4f}; F1: {result['test_metrics']['f1']:.4f}")
    print(f"Finished in {time.perf_counter() - started:.1f} seconds.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    run_experiment(args.data, args.output, args.jobs)


if __name__ == "__main__":
    main()
