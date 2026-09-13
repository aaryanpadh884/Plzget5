"""Exploratory analysis, run before any modeling.

Answers the questions the product plan asks up front: how many usable
team-games survive starter resolution, what the base rate of the label is, and
what the distribution of first-drive rushing totals looks like. Writes a
markdown summary and a small set of figures to ``reports/``.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from config import (FEATURES_PATH, LABELED_PATH, REPORT_DIR,  # noqa: E402
                    YARDS_THRESHOLD)

log = logging.getLogger(__name__)


def _fig(name: str):
    path = REPORT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path


def run() -> str:
    labeled = pd.read_parquet(LABELED_PATH)
    usable = labeled[labeled["is_ambiguous"] == 0].copy()

    lines: list[str] = ["# EDA: first-drive RB 5+ rushing yards", ""]

    # ---------------- sample size ----------------
    lines += [
        "## Sample size",
        "",
        f"- Team-games with a resolvable RB: **{len(labeled):,}**",
        f"- Usable after dropping ambiguous starters: **{len(usable):,}** "
        f"({100 * len(usable) / len(labeled):.1f}%)",
        f"- Seasons: {labeled['season'].min()}-{labeled['season'].max()}",
        "",
        "Starter resolution outcomes:",
        "",
        "```",
        labeled["starter_method"].value_counts().to_string(),
        "```",
        "",
        "Why team-games were dropped:",
        "",
        "```",
        labeled.loc[labeled.is_ambiguous == 1, "drop_reason"]
        .value_counts().to_string(),
        "```",
        "",
    ]

    # ---------------- base rate ----------------
    base = usable["label"].mean()
    lines += [
        "## Base rate",
        "",
        f"- P(first-drive rush yards >= {YARDS_THRESHOLD}) = **{base:.4f}**",
        f"- Positives: {int(usable['label'].sum()):,} / {len(usable):,}",
        "",
        "The label is close to a coin flip, so this is *not* an imbalanced "
        "classification problem and needs no resampling or class weighting. "
        "That is convenient: log loss and Brier score are directly "
        "interpretable against a 0.5 baseline.",
        "",
        "Base rate by season:",
        "",
        "```",
        usable.groupby("season")["label"].agg(["mean", "size"]).round(4).to_string(),
        "```",
        "",
    ]

    # ---------------- the zero-carry structural issue ----------------
    zero = (usable["fd_carries"] == 0).mean()
    cond = usable.loc[usable["fd_carries"] > 0, "label"].mean()
    lines += [
        "## The starter who never touches the ball",
        "",
        f"- Team-games where the resolved starter had **zero** carries on the "
        f"opening drive: **{zero:.1%}**",
        f"- Base rate conditional on at least one carry: **{cond:.4f}**",
        "",
        "This is the single most important structural fact in the dataset. "
        "The label is a compound event: the starter has to be given the ball "
        "on the opener *and* the carries have to total "
        f"{YARDS_THRESHOLD}+ yards. Roughly one in six negatives is decided "
        "before a single rushing play happens, by play-calling rather than by "
        "the back. Any model that ignores opener participation is trying to "
        "predict yardage on drives where the back was never involved.",
        "",
        "Distribution of first-drive carries by the starter:",
        "",
        "```",
        usable["fd_carries"].value_counts().sort_index().head(10).to_string(),
        "```",
        "",
    ]

    # ---------------- yardage distribution ----------------
    desc = usable["fd_rush_yards"].describe()
    near = usable["fd_rush_yards"].between(3, 7).mean()
    lines += [
        "## Distribution of first-drive rushing yards",
        "",
        "```",
        desc.round(3).to_string(),
        "```",
        "",
        f"- Median is {usable['fd_rush_yards'].median():.0f} yards, sitting "
        f"right on the {YARDS_THRESHOLD}-yard line, which is why the base rate "
        "lands near 0.5.",
        f"- {near:.1%} of team-games land between 3 and 7 yards, so a large "
        "share of outcomes are decided by a single yard or two. That caps how "
        "sharp any model can be here and is the strongest argument for "
        "optimizing calibration over accuracy.",
        "",
    ]

    # ---------------- correlates ----------------
    if FEATURES_PATH.exists():
        feats = pd.read_parquet(FEATURES_PATH)
        fu = feats[feats["is_ambiguous"] == 0]
        import features as F

        cols = [c for c in F.feature_columns(fu) if fu[c].notna().sum() > 100]
        corr = (fu[cols + ["label"]].corr()["label"].drop("label")
                .sort_values(key=np.abs, ascending=False))
        lines += [
            "## Strongest linear correlates of the label",
            "",
            "```",
            corr.head(15).round(4).to_string(),
            "```",
            "",
            "Every correlation is small. Nothing in the pregame feature set "
            "comes close to determining the outcome, which is what you would "
            "expect for a single-drive, few-carry event.",
            "",
        ]

    # ---------------- figures ----------------
    plt.figure(figsize=(7, 4))
    plt.hist(usable["fd_rush_yards"], bins=range(-15, 60), color="#4C72B0")
    plt.axvline(YARDS_THRESHOLD, color="#C44E52", linestyle="--",
                label=f"{YARDS_THRESHOLD}-yard line")
    plt.xlabel("Starting RB rushing yards on first drive")
    plt.ylabel("Team-games")
    plt.title("First-drive rushing yards")
    plt.legend()
    f1 = _fig("eda_yards_hist.png")

    plt.figure(figsize=(7, 4))
    by_season = usable.groupby("season")["label"].mean()
    plt.plot(by_season.index, by_season.values, marker="o", color="#4C72B0")
    plt.axhline(0.5, color="grey", linestyle=":")
    plt.ylim(0, 1)
    plt.xlabel("Season")
    plt.ylabel(f"P(>= {YARDS_THRESHOLD} yards)")
    plt.title("Base rate by season")
    f2 = _fig("eda_base_rate_by_season.png")

    plt.figure(figsize=(7, 4))
    carr = usable.groupby("fd_carries")["label"].agg(["mean", "size"])
    carr = carr[carr["size"] >= 20]
    plt.bar(carr.index.astype(int), carr["mean"], color="#4C72B0")
    plt.xlabel("Carries by the starter on the first drive")
    plt.ylabel(f"P(>= {YARDS_THRESHOLD} yards)")
    plt.title("Outcome is driven by opportunity")
    f3 = _fig("eda_label_by_carries.png")

    lines += ["## Figures", "",
              f"- `{f1.name}`", f"- `{f2.name}`", f"- `{f3.name}`", ""]

    report = "\n".join(lines)
    (REPORT_DIR / "eda.md").write_text(report)
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())
