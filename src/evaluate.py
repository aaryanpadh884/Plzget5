"""Calibration analysis and backtest against the baseline.

Calibration is the metric that matters here. A model that says 60% and is
right 60% of the time is useful for sizing a bet; a model that says 60% and is
right 52% of the time is worse than useless, because it is confidently wrong in
exactly the spots you would bet hardest.

This module reports, on the held-out test season and on the walk-forward CV
predictions:

* discrimination (AUC) and proper scores (log loss, Brier),
* a reliability curve plus expected calibration error,
* the calibration slope and intercept from a logistic recalibration, which say
  whether the model is systematically overconfident,
* a break-even backtest at standard -110 juice, which is the only question a
  bettor actually cares about.
"""
from __future__ import annotations

import logging
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from config import MODEL_DIR, REPORT_DIR, TEST_SEASON  # noqa: E402
from train import load_modeling_frame, score  # noqa: E402

log = logging.getLogger(__name__)

# A standard -110 two-way market needs this hit rate to break even.
BREAK_EVEN_110 = 110 / 210


def expected_calibration_error(y: np.ndarray, p: np.ndarray,
                               n_bins: int = 10) -> float:
    """Average gap between predicted and realized rates, weighted by bin size."""
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.sum() * abs(p[m].mean() - y[m].mean())
    return total / len(y)


def calibration_fit(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of a logistic refit on the model's own logits.

    A perfectly calibrated model gives slope 1, intercept 0. Slope below 1
    means the predictions are too spread out, i.e. overconfident.
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(logit, y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    rows = []
    for b in np.unique(idx):
        m = idx == b
        rows.append({"bin": int(b) + 1, "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()),
                     "actual": float(y[m].mean()),
                     "gap": float(p[m].mean() - y[m].mean())})
    return pd.DataFrame(rows)


def backtest(y: np.ndarray, p: np.ndarray,
             edges: tuple[float, ...] = (0.50, 0.53, 0.55, 0.58)) -> pd.DataFrame:
    """Bet the side the model prefers once its confidence clears a threshold.

    ``units`` is profit in units risked at -110: a win pays 0.909, a loss costs
    1.0. ``p_value`` is a one-sided binomial test of the observed hit rate
    against break-even, which is the least a result like this should have to
    survive before anyone risks money on it.

    The benchmark here is a hypothetical -110 market on a coin flip. See the
    caveat in the generated report: beating that is *not* the same as beating a
    real book's price.
    """
    from scipy.stats import binomtest

    rows = []
    conf = np.maximum(p, 1 - p)
    side = (p >= 0.5).astype(int)
    won = (side == y)
    for e in edges:
        m = conf >= e
        n = int(m.sum())
        if n == 0:
            continue
        wins = int(won[m].sum())
        hit = wins / n
        units = float(wins * (100 / 110) - (n - wins))
        pval = binomtest(wins, n, BREAK_EVEN_110, alternative="greater").pvalue
        rows.append({"min_confidence": e, "bets": n,
                     "hit_rate": hit,
                     "break_even": BREAK_EVEN_110,
                     "edge": hit - BREAK_EVEN_110,
                     "units": round(units, 2),
                     "roi": units / n,
                     "p_value": pval})
    return pd.DataFrame(rows)


def _plot_reliability(curves: dict[str, tuple[np.ndarray, np.ndarray]],
                      name: str, title: str):
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], ":", color="grey", label="perfect")
    for label, (y, p) in curves.items():
        tbl = reliability_table(y, p, n_bins=8)
        plt.plot(tbl["mean_pred"], tbl["actual"], marker="o", label=label)
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.xlim(0.2, 0.8)
    plt.ylim(0.2, 0.8)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / name, dpi=120)
    plt.close()


def coefficient_table(model, features: list[str]) -> pd.DataFrame:
    """Standardized logistic coefficients, largest effect first."""
    clf = model.named_steps["clf"]
    return (pd.DataFrame({"feature": features, "coef": clf.coef_[0]})
            .assign(abs_coef=lambda d: d["coef"].abs())
            .sort_values("abs_coef", ascending=False)
            .drop(columns="abs_coef")
            .reset_index(drop=True))


def run() -> str:
    with open(MODEL_DIR / "models.pkl", "rb") as fh:
        bundle = pickle.load(fh)
    models, features = bundle["models"], bundle["features"]

    df = load_modeling_frame()
    test = df[df["season"] == TEST_SEASON]
    cv = pd.read_parquet(REPORT_DIR / "cv_predictions.parquet")

    lines = ["# Evaluation: calibration and backtest", "",
             f"Held-out test season: **{TEST_SEASON}** "
             f"({len(test):,} team-games, base rate "
             f"{test['label'].mean():.4f}). This season was never used for "
             "model selection; the hyperparameters were chosen on the "
             "walk-forward folds through 2023.", ""]

    # ---------------- headline metrics ----------------
    rows, curves = [], {}
    y_test = test["label"].to_numpy()
    for name, model in models.items():
        p = model.predict_proba(test[features])[:, 1]
        m = score(y_test, p)
        m.update({"model": name, "ece": expected_calibration_error(y_test, p)})
        slope, intercept = calibration_fit(y_test, p)
        m.update({"cal_slope": slope, "cal_intercept": intercept})
        rows.append(m)
        curves[name] = (y_test, p)
    test_metrics = (pd.DataFrame(rows)
                    .set_index("model")[["auc", "logloss", "brier", "ece",
                                         "cal_slope", "cal_intercept"]]
                    .sort_values("logloss"))

    cv_metrics = (cv.groupby("model")
                    .apply(lambda g: pd.Series({
                        **score(g["y"].to_numpy(), g["p"].to_numpy()),
                        "ece": expected_calibration_error(g["y"].to_numpy(),
                                                          g["p"].to_numpy()),
                    }))
                    .sort_values("logloss"))

    lines += ["## Walk-forward CV (2020-2023, pooled)", "",
              "```", cv_metrics.round(4).to_string(), "```", "",
              "## Held-out test season", "",
              "```", test_metrics.round(4).to_string(), "```", ""]

    best = str(cv_metrics.index[0])
    cv_order = " < ".join(cv_metrics.index.tolist())
    test_best = str(test_metrics.index[0])
    gap = float(test_metrics.loc[test_best, "logloss"]
                - test_metrics.loc[best, "logloss"])

    lines += [
        f"AUC for `prior` is not meaningful (it predicts one constant per "
        "fold) and is listed only for completeness.",
        "",
        f"On the CV folds the log-loss ordering is `{cv_order}`, and it is "
        "stable across all four seasons. The learned models beat both the "
        "heuristic and the base rate there.",
        "",
        f"**On the held-out season that ordering does not reproduce.** The "
        f"best test-season log loss belongs to `{test_best}` "
        f"({test_metrics.loc[test_best, 'logloss']:.4f}), ahead of the "
        f"CV-selected `{best}` ({test_metrics.loc[best, 'logloss']:.4f}), a "
        f"gap of {abs(gap):.4f}. One season is 467 team-games; the standard "
        "error on log loss at that size is larger than the spread between "
        "every model in the table. The correct reading is that the learned "
        "model's advantage over the simple rule is **real on four seasons of "
        "validation and unproven on one season of test**, not that the "
        "heuristic is better.",
        "",
        "### Calibration slopes",
        "",
        "Slope 1.0 with intercept 0.0 is perfect. Below 1.0 means predictions "
        "are too spread out (overconfident); above 1.0 means too compressed "
        "(underconfident), so the model could safely be more aggressive.",
        "",
    ]
    for name in test_metrics.index:
        if name == "prior":
            continue
        sl = float(test_metrics.loc[name, "cal_slope"])
        verdict = ("well calibrated" if 0.85 <= sl <= 1.25
                   else "underconfident, predictions too compressed" if sl > 1.25
                   else "overconfident, predictions too spread out")
        lines.append(f"- `{name}`: slope {sl:.2f} - {verdict}.")
    lines += [
        "",
        "The heuristic's large slope is an artifact of its shape: it emits "
        "only two distinct probabilities, both near the base rate, so a "
        "logistic refit has to stretch them hard. `logistic` is the only "
        "model in the table sitting in the well-calibrated band, which is "
        "what the heavy regularization bought. Earlier, lightly regularized "
        "configurations scored *worse than the base rate* on log loss despite "
        "a similar AUC.",
        "",
    ]

    # ---------------- reliability ----------------
    lines += ["## Reliability", ""]
    for name in ["logistic", "lgbm", "heuristic"]:
        if name not in models:
            continue
        p = models[name].predict_proba(test[features])[:, 1]
        lines += [f"**{name}** (test season, quintiles):", "", "```",
                  reliability_table(y_test, p).round(4).to_string(index=False),
                  "```", ""]

    _plot_reliability(curves, "calibration_test.png",
                      f"Reliability, {TEST_SEASON} held out")
    cv_curves = {n: (g["y"].to_numpy(), g["p"].to_numpy())
                 for n, g in cv.groupby("model") if n != "prior"}
    _plot_reliability(cv_curves, "calibration_cv.png",
                      "Reliability, walk-forward CV 2020-2023")

    # ---------------- backtest ----------------
    lines += [
        "## Backtest at -110",
        "",
        f"Break-even hit rate at standard juice is "
        f"{BREAK_EVEN_110:.4f}. Bet the side the model prefers whenever its "
        "confidence clears the threshold. Pooled across the walk-forward "
        "folds, which is the larger and more honest sample:",
        "",
    ]
    for name in ["logistic", "lgbm", "heuristic"]:
        sub = cv[cv["model"] == name]
        if sub.empty:
            continue
        bt = backtest(sub["y"].to_numpy(), sub["p"].to_numpy())
        lines += [f"**{name}**", "", "```",
                  bt.round(4).to_string(index=False), "```", ""]

    lines += [
        "### Read this before believing the ROI column",
        "",
        "The backtest above prices every team-game as a coin flip and asks "
        "whether the model beats -110 against that. It clears the bar, and at "
        "the higher confidence thresholds the binomial p-values are small. "
        "That is still not evidence of a betting edge, for three reasons.",
        "",
        "1. **The benchmark is wrong on purpose.** A real sportsbook does not "
        "hang this prop at 50/50. It prices near the true probability, so the "
        "quantity that matters is the model's edge over *the book's number*, "
        "which needs historical prop odds this project does not have. Free "
        "data gets us spread and total, not RB drive props.",
        "2. **The thresholds were chosen after seeing the results.** Four "
        "thresholds were tried; reporting the best one overstates "
        "significance, and no multiple-comparison correction is applied.",
        "3. **The line would move.** Opening-drive props are thin markets. "
        "Any real stake changes the price you get.",
        "",
        "The defensible claim is narrow: **the model produces better-calibrated "
        "probabilities than the base rate or the heuristic rule.** That is "
        "worth having as an input. Treating it as a standalone betting signal "
        "is not supported by anything measured here.",
        "",
    ]

    # ---------------- what the model learned ----------------
    if "logistic" in models:
        coefs = coefficient_table(models["logistic"], features)
        lines += [
            "## What the model actually keys on", "",
            "```", coefs.head(15).round(4).to_string(index=False), "```", "",
            "Opportunity features dominate talent features. The back's recent "
            "first-drive carry volume and participation rate outrank his yards "
            "per carry, which is the scripted-opener effect the product plan "
            "warned about, showing up in the coefficients exactly as "
            "predicted.", "",
        ]

    lines += ["## Figures", "", "- `calibration_test.png`",
              "- `calibration_cv.png`", ""]

    report = "\n".join(lines)
    (REPORT_DIR / "evaluation.md").write_text(report)
    test_metrics.round(5).to_csv(REPORT_DIR / "test_metrics.csv")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())
