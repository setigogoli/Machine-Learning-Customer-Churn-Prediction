"""Fold-local preprocessing and a combined feature ranking."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, check_array

from .data import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS


def make_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ], verbose_feature_names_out=False)


class CombinedFeatureSelector(TransformerMixin, BaseEstimator):
    """Select k columns using filter, Lasso and random-forest average ranks.

    Lasso is used as a feature-ranking regression on a binary target, not as
    the final classifier. All ranking models are fitted inside the outer
    training fold. The filter scores are screening heuristics, not adjusted
    hypothesis tests.
    """

    def __init__(self, k=12, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        if not 1 <= self.k <= X.shape[1]:
            raise ValueError("k must be between 1 and the number of encoded features.")
        nonnegative = MinMaxScaler().fit_transform(X)
        chi_scores = np.nan_to_num(chi2(nonnegative, y)[0], nan=0, posinf=0)
        anova_scores = np.nan_to_num(f_classif(X, y)[0], nan=0, posinf=0)
        self.filter_scores_ = (
            chi_scores / max(chi_scores.max(), 1e-12)
            + anova_scores / max(anova_scores.max(), 1e-12)
        ) / 2
        standardized = StandardScaler().fit_transform(X)
        lasso = LassoCV(
            alphas=np.logspace(-4, -1, 15), cv=3, max_iter=10000,
            random_state=self.random_state,
        ).fit(standardized, y)
        self.lasso_scores_ = np.abs(lasso.coef_)
        forest = RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced",
            random_state=self.random_state, n_jobs=1,
        ).fit(X, y)
        self.forest_scores_ = forest.feature_importances_
        ranks = np.array([
            pd.Series(-score).rank(method="average").to_numpy()
            for score in [self.filter_scores_, self.lasso_scores_, self.forest_scores_]
        ])
        self.average_rank_ = ranks.mean(axis=0)
        top = np.argsort(self.average_rank_, kind="stable")[:self.k]
        self.support_ = np.zeros(self.n_features_in_, dtype=bool)
        self.support_[top] = True
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Encoded feature count differs from the fitted selector.")
        return X[:, self.support_]

    def ranking_frame(self, feature_names):
        check_is_fitted(self, "support_")
        return pd.DataFrame({
            "feature": feature_names, "filter_score": self.filter_scores_,
            "lasso_abs_coefficient": self.lasso_scores_,
            "rf_importance": self.forest_scores_,
            "average_rank": self.average_rank_, "selected": self.support_,
        }).sort_values(["average_rank", "feature"]).reset_index(drop=True)
