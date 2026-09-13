"""How accurate can this model possibly get?

The question this answers: when the model says 55%, how much better than that
could *any* pregame model do? Rather than guess, we measure it by handing a
model information it could never have on Sunday morning and seeing how much it
improves.

The oracle is the number of carries the back actually took on the opening
drive. That is unknowable pregame (it depends on how long the drive lasted,
which depends on the drive), but it bounds what perfect opportunity forecasting
would be worth.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import CV_START_SEASON, RANDOM_STATE, REPORT_DIR, TEST_SEASON
from features import feature_columns
from train import load_modeling_frame

log = logging.getLogger(__name__)


def _pipe(C: float = 0.001) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=5000,
                                   random_state=RANDOM_STATE)),
    ])


def _score(y, p) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": roc_auc_score(y, p),
            "logloss": log_loss(y, p, labels=[0, 1]),
            "brier": brier_score_loss(y, p)}


def _walk(df: pd.DataFrame, cols: list[str], target: str = "label") -> dict:
    seasons = [s for s in sorted(df["season"].unique())
               if CV_START_SEASON <= s < TEST_SEASON]
    out = []
    for season in seasons:
        tr, va = df[df["season"] < season], df[df["season"] == season]
        model = _pipe().fit(tr[cols], tr[target].to_numpy())
        out.append(_score(va[target].to_numpy(),
                          model.predict_proba(va[cols])[:, 1]))
    return pd.DataFrame(out).mean().to_dict()


def murphy_decomposition(y: np.ndarray, p: np.ndarray,
                         n_bins: int = 10) -> dict:
    """Brier = reliability - resolution + uncertainty.

    * uncertainty is fixed by the base rate and no model can reduce it,
    * resolution is how far the model moves away from the base rate correctly,
    * reliability is miscalibration, and lower is better.
    """
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    base = y.mean()
    rel = res = 0.0
    for b in np.unique(idx):
        m = idx == b
        w = m.sum() / len(y)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - base) ** 2
    return {"reliability": rel, "resolution": res,
            "uncertainty": base * (1 - base),
            "brier_check": rel - res + base * (1 - base)}


def run() -> str:
    df = load_modeling_frame()
    cols = feature_columns(df)
    df = df.copy()
    df["oracle_carries"] = df["fd_carries"]

    pregame = _walk(df, cols)
    oracle_only = _walk(df, ["oracle_carries"])
    oracle_plus = _walk(df, cols + ["oracle_carries"])

    lines = [
        "# How accurate can this model get?", "",
        "Every attempt to squeeze more out of this problem lands in the same "
        "place, so the useful question is not 'can we do better' but 'how much "
        "is knowable at all'. This report measures that.", "",
        "## The experiment", "",
        "Give a model the one thing it could never know before kickoff: how "
        "many carries the back actually took on the opening drive. Compare.",
        "",
        "| information available | AUC | log loss | Brier |",
        "|---|---|---|---|",
        f"| pregame features only (the real model) | {pregame['auc']:.4f} | "
        f"{pregame['logloss']:.4f} | {pregame['brier']:.4f} |",
        f"| **oracle: carry count alone** | **{oracle_only['auc']:.4f}** | "
        f"**{oracle_only['logloss']:.4f}** | **{oracle_only['brier']:.4f}** |",
        f"| oracle carries + all pregame features | {oracle_plus['auc']:.4f} | "
        f"{oracle_plus['logloss']:.4f} | {oracle_plus['brier']:.4f} |",
        "",
        "Two things fall out of this table.",
        "",
        "**Carry count is very nearly the whole answer.** Knowing only how "
        f"many times the back carried gives AUC {oracle_only['auc']:.3f}. "
        f"Every pregame feature we built gives {pregame['auc']:.3f}.",
        "",
        "**Pregame features add almost nothing on top of it.** Going from "
        f"carry count alone to carry count plus all "
        f"{len(cols)} pregame features moves log loss only "
        f"{oracle_only['logloss'] - oracle_plus['logloss']:.4f}. Once you know "
        "the opportunity, the back's talent, the matchup, and the game script "
        "are nearly irrelevant.",
        "",
        "## Why the outcome is so close to a coin flip", "",
    ]

    tbl = (df.groupby(df["fd_carries"].clip(upper=6))["label"]
             .agg(["mean", "size"]).round(3))
    tbl.index.name = "carries on first drive"
    lines += ["```", tbl.to_string(), "```", "",
              "The threshold sits exactly where the carry distribution is "
              "densest. One carry is a 25% proposition, two carries is 67%. "
              f"The average opening drive gives the starter "
              f"{df['fd_carries'].mean():.2f} carries, so most team-games land "
              "on the steepest part of that curve, where a single extra "
              "handoff flips the answer.", ""]

    # How well can each part be predicted pregame?
    lines += ["## What pregame data does and does not know", ""]
    rows = []
    for thr in (1, 2, 3):
        tgt = f"_ge{thr}"
        df[tgt] = (df["fd_carries"] >= thr).astype(int)
        m = _walk(df, cols, target=tgt)
        rows.append({"question": f"P(at least {thr} carries)",
                     "base rate": df[tgt].mean(), "AUC": m["auc"],
                     "log loss": m["logloss"]})
    rows.append({"question": "P(5+ rushing yards) [the label]",
                 "base rate": df["label"].mean(), "AUC": pregame["auc"],
                 "log loss": pregame["logloss"]})
    lines += ["```",
              pd.DataFrame(rows).round(4).to_string(index=False),
              "```", "",
              "This is the crux. Pregame data predicts **whether** the back "
              "touches the ball reasonably well, because that is a question "
              "about his role and teams telegraph roles. It predicts **how "
              "many times** much worse, because carry count depends on how "
              "long the drive lasts, and drive length is decided by the drive "
              "itself: a third-down conversion, a holding penalty, an "
              "interception. None of that is knowable on Saturday night.", ""]

    # Murphy decomposition of the real model.
    seasons = [s for s in sorted(df["season"].unique())
               if CV_START_SEASON <= s < TEST_SEASON]
    ys, ps = [], []
    for season in seasons:
        tr, va = df[df["season"] < season], df[df["season"] == season]
        model = _pipe().fit(tr[cols], tr["label"].to_numpy())
        ys.append(va["label"].to_numpy())
        ps.append(model.predict_proba(va[cols])[:, 1])
    y, p = np.concatenate(ys), np.concatenate(ps)
    mur = murphy_decomposition(y, p)

    lines += [
        "## Where the Brier score actually goes", "",
        "Brier = reliability - resolution + uncertainty.", "",
        "```",
        f"uncertainty  {mur['uncertainty']:.4f}   irreducible; fixed by the "
        "0.5 base rate",
        f"resolution   {mur['resolution']:.4f}   how far the model correctly "
        "moves off the base rate",
        f"reliability  {mur['reliability']:.4f}   miscalibration; smaller is "
        "better",
        f"-> Brier     {mur['brier_check']:.4f}",
        "```",
        "",
        f"Resolution of {mur['resolution']:.4f} against uncertainty of "
        f"{mur['uncertainty']:.4f} means the model explains roughly "
        f"{100 * mur['resolution'] / mur['uncertainty']:.1f}% of the available "
        "variance. That is small, and it is small for a real reason rather "
        "than a fixable one.",
        "",
        "Reliability is near zero, which is the part that matters for your "
        "use case: the probabilities the model emits are close to the "
        "frequencies they describe. It is not confidently wrong, it is "
        "honestly uncertain.",
        "",
        "## Practical read", "",
        "The model's spread of predictions is narrow on purpose. Most "
        f"team-games come back between {np.quantile(p, 0.05):.2f} and "
        f"{np.quantile(p, 0.95):.2f}, and that narrowness is the correct "
        "answer, not a limitation to engineer away. A model that confidently "
        "said 0.75 here would be lying.",
        "",
        "If you want materially sharper probabilities, the only lever that "
        "would move them is better opportunity forecasting: projected snap "
        "share and game-script-conditional carry projections, ideally a "
        "beat-reporter or projection feed. More RB efficiency features will "
        "not do it, and this table is why.",
        "",
    ]

    report = "\n".join(lines)
    (REPORT_DIR / "ceiling.md").write_text(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())
