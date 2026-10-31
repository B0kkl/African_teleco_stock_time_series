"""Walk-forward training and scoring for direction (classification) and return (regression)."""
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def make_classifiers():
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "random_forest": RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=20,
                                                random_state=42, n_jobs=-1),
    }


def make_regressors():
    return {
        "zero_baseline": DummyRegressor(strategy="constant", constant=0.0),
        "train_mean": DummyRegressor(strategy="mean"),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "random_forest": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=20,
                                               random_state=42, n_jobs=-1),
    }


def walk_forward(feat, features, kind="classification", n_splits=5, gap=1):
    """Out-of-sample predictions from expanding-window walk-forward folds.

    Each fold trains on the past only; `gap` rows are skipped between train and test so
    a training label (which looks one day ahead) never overlaps the test period.
    Returns a DataFrame of predictions (probability of an up day, or predicted return).
    """
    X = feat[features]
    y = feat["target"] if kind == "classification" else feat["next_ret"]
    factory = make_classifiers if kind == "classification" else make_regressors
    out = {m: pd.Series(np.nan, index=feat.index) for m in factory()}
    for tr, te in TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X):
        for name, model in factory().items():
            model.fit(X.iloc[tr], y.iloc[tr])
            out[name].iloc[te] = (model.predict_proba(X.iloc[te])[:, 1]
                                  if kind == "classification" else model.predict(X.iloc[te]))
    return pd.DataFrame(out).dropna()


def score_classifiers(feat, preds):
    test = feat.loc[preds.index]
    rows = []
    for m in preds:
        rows.append({"model": m,
                     "accuracy": accuracy_score(test["target"], preds[m] > 0.5),
                     "roc_auc": 0.5 if m == "majority_baseline" else roc_auc_score(test["target"], preds[m]),
                     "test_up_share": test["target"].mean(), "test_days": len(test)})
    return pd.DataFrame(rows).set_index("model")


def score_regressors(feat, preds):
    """MAE/RMSE plus out-of-sample R^2 measured against the predict-zero baseline.
    oos_r2_vs_zero > 0 means the model beats 'tomorrow's return is zero'."""
    y = feat.loc[preds.index, "next_ret"]
    sse_zero = float(((y - 0.0) ** 2).sum())
    rows = []
    for m in preds:
        p = preds[m]
        rows.append({"model": m,
                     "MAE": mean_absolute_error(y, p),
                     "RMSE": float(np.sqrt(mean_squared_error(y, p))),
                     "oos_r2_vs_zero": 1 - float(((y - p) ** 2).sum()) / sse_zero,
                     "direction_acc": float(((p > 0) == (y > 0)).mean()),
                     "test_days": len(y)})
    return pd.DataFrame(rows).set_index("model")
