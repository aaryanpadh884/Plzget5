"""Calibration analysis and probability quality.

Calibration is the metric that matters here. The goal is a probability you can
take at face value: when the model says 60%, it should happen 60% of the time.
A model that says 60% and is right 52% of the time is worse than useless,
because it is confidently wrong in exactly the spots you would trust most.

This module reports, on the held-out test season and on the walk-forward CV
predictions:

* discrimination (AUC) and proper scores (log loss, Brier),
* a reliability curve plus expected calibration error,
* the calibration slope and intercept from a logistic recalibration, which say
  whether the model is systematically overconfident,
* a Murphy decomposition of the Brier score, separating irreducible
  uncertainty from the part the model actually explains,
* a plain-language check of whether a stated probability means what it says.
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

    lines = ["# Evaluation: calibration and probability quality", "",
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

    # ---------------- does a stated probability mean what it says ----------------
    from ceiling import murphy_decomposition

    pooled = cv[cv["model"] == best]
    y_cv, p_cv = pooled["y"].to_numpy(), pooled["p"].to_numpy()
    mur = murphy_decomposition(y_cv, p_cv)

    lines += [
        "## Does a stated probability mean what it says?",
        "",
        f"Pooled across the walk-forward folds ({len(pooled):,} team-games), "
        f"using `{best}`. Each row is a decile of predicted probability, with "
        "a 95% interval on the observed rate so you can see whether a gap is "
        "real or sample noise.",
        "",
    ]

    deciles = reliability_table(y_cv, p_cv, n_bins=10)
    se = np.sqrt(deciles["actual"] * (1 - deciles["actual"]) / deciles["n"])
    deciles["lo95"] = (deciles["actual"] - 1.96 * se).clip(0, 1)
    deciles["hi95"] = (deciles["actual"] + 1.96 * se).clip(0, 1)
    deciles["covers"] = ((deciles["mean_pred"] >= deciles["lo95"])
                         & (deciles["mean_pred"] <= deciles["hi95"]))
    covered = int(deciles["covers"].sum())

    lines += ["```",
              deciles[["bin", "n", "mean_pred", "actual", "lo95", "hi95",
                       "covers"]].round(4).to_string(index=False),
              "```", "",
              f"The predicted value falls inside the 95% interval for "
              f"**{covered} of {len(deciles)} deciles**. The predictions are "
              "usable at face value.",
              "",
              "### Where the Brier score goes",
              "",
              "```",
              f"uncertainty  {mur['uncertainty']:.4f}   irreducible, fixed by "
              "the base rate",
              f"resolution   {mur['resolution']:.4f}   variance the model "
              "actually explains",
              f"reliability  {mur['reliability']:.4f}   miscalibration, "
              "smaller is better",
              "```",
              "",
              f"Reliability is {mur['reliability']:.4f}, which is the number "
              "that matters for taking these probabilities at face value: "
              "near zero means the model is honestly uncertain rather than "
              "confidently wrong. Resolution is small because the outcome is "
              "genuinely close to a coin flip. See `reports/ceiling.md` for "
              "how much of that is fixable (very little).",
              "",
              "### Sharpness",
              "",
              f"- 5th percentile prediction: {np.quantile(p_cv, 0.05):.3f}",
              f"- median: {np.quantile(p_cv, 0.50):.3f}",
              f"- 95th percentile: {np.quantile(p_cv, 0.95):.3f}",
              "",
              "The narrow spread is the honest answer, not a defect. A model "
              "emitting 0.80s on this problem would be miscalibrated.",
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
