"""Model training with walk-forward validation.

Three models, in increasing order of ambition:

``heuristic``
    The dumb rule from the product plan: predict yes when the back's trailing
    five-game average first-drive rushing yards is at least the threshold. It
    is turned into a two-bin probability model by reading the empirical hit
    rate of each bin off the *training* data, so it can be scored on log loss
    and Brier alongside everything else rather than only on accuracy.

``logistic``
    Median imputation, standardization, L2 logistic regression.

``lgbm``
    Gradient boosted trees, shallow and heavily regularized.

Both learned models are regularized far harder than defaults would suggest.
That is not timidity, it is what the walk-forward folds asked for: sweeping the
logistic penalty from C=1.0 down to C=0.0001 improved validation log loss
monotonically until roughly C=0.001, and an unconstrained LightGBM scored
*worse than predicting the base rate* on log loss despite an AUC above 0.55.
With a few thousand rows and correlations in the 0.1 range, any model confident
enough to be interesting is confident enough to be wrong.

Validation is walk-forward by season: predict season S having trained on every
season before it. There is no random splitting anywhere. The final season is
held out entirely and never used for model selection.
"""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (CV_START_SEASON, FEATURES_PATH, MODEL_DIR, RANDOM_STATE,
                    REPORT_DIR, TEST_SEASON, YARDS_THRESHOLD)
from features import HEURISTIC_COL, feature_columns

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# The baseline to beat
# --------------------------------------------------------------------------
class HeuristicBaseline(BaseEstimator, ClassifierMixin):
    """"Yes if the back's trailing first-drive average clears the threshold."

    Kept honest as a probability model: each side of the rule gets the hit
    rate that side achieved in training, so it is scored on the same footing
    as the learned models.
    """

    def __init__(self, column: str = HEURISTIC_COL,
                 threshold: float = YARDS_THRESHOLD):
        self.column = column
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        says_yes = self._rule(X)
        prior = y.mean()
        # Fall back to the overall prior when a bin is empty or tiny.
        self.p_yes_ = y[says_yes].mean() if says_yes.sum() >= 30 else prior
        self.p_no_ = y[~says_yes].mean() if (~says_yes).sum() >= 30 else prior
        self.classes_ = np.array([0, 1])
        return self

    def _rule(self, X: pd.DataFrame) -> np.ndarray:
        # A back with no history does not trigger the rule.
        return (X[self.column].fillna(-1).to_numpy() >= self.threshold)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.where(self._rule(X), self.p_yes_, self.p_no_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class PriorBaseline(BaseEstimator, ClassifierMixin):
    """Predict the training base rate for everyone. The floor for log loss."""

    def fit(self, X, y):
        self.p_ = float(np.asarray(y).mean())
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X) -> np.ndarray:
        p = np.full(len(X), self.p_)
        return np.column_stack([1 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------------------
# Learned models
# --------------------------------------------------------------------------
def make_logistic() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=0.001, max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])


def make_lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=200,
        learning_rate=0.01,
        num_leaves=3,
        max_depth=2,
        min_child_samples=150,
        subsample=0.7,
        subsample_freq=1,
        colsample_bytree=0.5,
        reg_alpha=5.0,
        reg_lambda=20.0,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


MODEL_BUILDERS = {
    "prior": PriorBaseline,
    "heuristic": HeuristicBaseline,
    "logistic": make_logistic,
    "lgbm": make_lgbm,
}


# --------------------------------------------------------------------------
# Data + splits
# --------------------------------------------------------------------------
def load_modeling_frame() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["is_ambiguous"] == 0].copy()
    # A back with no prior game at all has no player features to speak of.
    df = df[df[HEURISTIC_COL].notna() | (df["week"] > 1)]
    return df.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


@dataclass
class FoldResult:
    model: str
    season: int
    n_train: int
    n_valid: int
    auc: float
    logloss: float
    brier: float
    base_rate: float
    preds: pd.DataFrame = field(repr=False, default=None)


def score(y_true, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "auc": roc_auc_score(y_true, p),
        "logloss": log_loss(y_true, p, labels=[0, 1]),
        "brier": brier_score_loss(y_true, p),
    }


def walk_forward(df: pd.DataFrame, features: list[str],
                 model_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on everything before season S, validate on S, for each S."""
    seasons = sorted(s for s in df["season"].unique()
                     if CV_START_SEASON <= s < TEST_SEASON)
    rows, all_preds = [], []

    for season in seasons:
        train = df[df["season"] < season]
        valid = df[df["season"] == season]
        X_tr, y_tr = train[features], train["label"].to_numpy()
        X_va, y_va = valid[features], valid["label"].to_numpy()

        for name in model_names:
            model = MODEL_BUILDERS[name]()
            model.fit(X_tr, y_tr)
            p = model.predict_proba(X_va)[:, 1]
            metrics = score(y_va, p)
            rows.append(FoldResult(model=name, season=season, n_train=len(train),
                                   n_valid=len(valid), base_rate=float(y_va.mean()),
                                   **metrics).__dict__)
            all_preds.append(pd.DataFrame({
                "model": name, "season": season,
                "game_id": valid["game_id"].to_numpy(),
                "team": valid["team"].to_numpy(),
                "week": valid["week"].to_numpy(),
                "starter_name": valid["starter_name"].to_numpy(),
                "y": y_va, "p": p,
            }))
        log.info("season %s: trained on %s rows, validated on %s",
                 season, len(train), len(valid))

    folds = pd.DataFrame(rows).drop(columns=["preds"])
    return folds, pd.concat(all_preds, ignore_index=True)


def fit_final(df: pd.DataFrame, features: list[str], name: str):
    """Fit on every season before the held-out test season.

    This is the model ``evaluate.py`` scores, so it must not have seen the test
    season. It is deliberately *not* the model used for live inference.
    """
    train = df[df["season"] < TEST_SEASON]
    model = MODEL_BUILDERS[name]()
    model.fit(train[features], train["label"].to_numpy())
    return model, train


def fit_production(df: pd.DataFrame, features: list[str], name: str):
    """Fit on every row available, including the most recent completed season.

    Live predictions should use every game that has actually been played.
    Holding out the latest season is the right call when measuring the model
    and the wrong call when running it, so the two fits are kept separate and
    saved to separate files.
    """
    model = MODEL_BUILDERS[name]()
    model.fit(df[features], df["label"].to_numpy())
    return model, df


def main() -> None:
    df = load_modeling_frame()
    features = feature_columns(df)
    model_names = list(MODEL_BUILDERS)

    log.info("modeling rows: %s | features: %s", len(df), len(features))

    folds, preds = walk_forward(df, features, model_names)
    summary = (folds.groupby("model")[["auc", "logloss", "brier"]]
                    .mean().sort_values("logloss"))

    log.info("\n--- walk-forward CV, mean across seasons ---\n%s",
             summary.round(4).to_string())
    log.info("\n--- per season ---\n%s",
             folds.pivot(index="season", columns="model", values="auc")
                  .round(4).to_string())

    folds.to_csv(REPORT_DIR / "cv_folds.csv", index=False)
    preds.to_parquet(REPORT_DIR / "cv_predictions.parquet", index=False)
    summary.round(5).to_csv(REPORT_DIR / "cv_summary.csv")

    # Fit every model on all pre-test seasons and persist, so evaluate.py can
    # score the held-out season without retraining.
    artifacts = {}
    for name in model_names:
        model, train = fit_final(df, features, name)
        artifacts[name] = model
        log.info("fitted final %s on %s rows (seasons < %s)",
                 name, len(train), TEST_SEASON)

    with open(MODEL_DIR / "models.pkl", "wb") as fh:
        pickle.dump({"models": artifacts, "features": features,
                     "test_season": TEST_SEASON}, fh)

    # Separate fit for live inference, trained on everything available.
    production = {}
    for name in model_names:
        model, train = fit_production(df, features, name)
        production[name] = model
    log.info("fitted production models on all %s rows (through %s week %s)",
             len(df), int(df["season"].max()),
             int(df.loc[df["season"] == df["season"].max(), "week"].max()))
    with open(MODEL_DIR / "models_production.pkl", "wb") as fh:
        pickle.dump({"models": production, "features": features,
                     "trained_through_season": int(df["season"].max()),
                     "trained_through_week": int(
                         df.loc[df["season"] == df["season"].max(),
                                "week"].max()),
                     "n_rows": int(len(df))}, fh)

    (MODEL_DIR / "metadata.json").write_text(json.dumps({
        "n_rows": int(len(df)),
        "n_features": len(features),
        "features": features,
        "cv_seasons": sorted(int(s) for s in folds["season"].unique()),
        "test_season": int(TEST_SEASON),
        "cv_summary": summary.round(5).to_dict(),
    }, indent=2))
    log.info("saved models and metadata to %s", MODEL_DIR)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Re-enter through the module namespace so the custom estimators pickle as
    # ``train.HeuristicBaseline`` rather than ``__main__.HeuristicBaseline``,
    # which would make the saved models unloadable from anywhere else.
    import train

    train.main()
