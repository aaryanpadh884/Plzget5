"""Leakage guards.

The headline test is :func:`test_features_are_invariant_to_the_future`, which
rebuilds features from a play-by-play feed truncated to everything before a
cutoff week and asserts the cutoff week's features come out identical. If any
feature reached forward in time, truncating the future would change it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features as F  # noqa: E402
import ingest  # noqa: E402
from config import FEATURES_PATH, LABELED_PATH, SEASONS  # noqa: E402


@pytest.fixture(scope="module")
def feats() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip("run src/features.py first")
    return pd.read_parquet(FEATURES_PATH)


def test_feature_columns_exclude_outcome_columns(feats):
    cols = set(F.feature_columns(feats))
    overlap = cols & set(F.LEAKY_COLUMNS)
    assert not overlap, f"outcome columns leaked into the model inputs: {overlap}"


def test_lagged_never_sees_the_current_row():
    df = pd.DataFrame({
        "player_id": ["a"] * 5,
        "season": [2020] * 5,
        "week": [1, 2, 3, 4, 5],
        "carries": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    out = F._lagged(df, ["player_id"], ["carries"], window=3, suffix="t3")
    got = out.sort_values("week")["carries_t3"].tolist()

    # Week 1 has no history; week 4 sees weeks 1-3; week 5 sees weeks 2-4.
    assert np.isnan(got[0])
    assert got[1] == 10.0
    assert got[3] == 10.0 + 20.0 + 30.0
    assert got[4] == 20.0 + 30.0 + 40.0


def test_expanding_never_sees_the_current_row():
    df = pd.DataFrame({
        "player_id": ["a"] * 4,
        "season": [2020] * 4,
        "week": [1, 2, 3, 4],
        "carries": [5.0, 5.0, 5.0, 5.0],
    })
    out = F._expanding(df, ["player_id", "season"], ["carries"], suffix="std")
    got = out.sort_values("week")["carries_std"].tolist()
    assert np.isnan(got[0])
    assert got[1:] == [5.0, 10.0, 15.0]


def test_lagged_does_not_bleed_across_players():
    df = pd.DataFrame({
        "player_id": ["a", "a", "b", "b"],
        "season": [2020] * 4,
        "week": [1, 2, 1, 2],
        "carries": [10.0, 10.0, 99.0, 99.0],
    })
    out = F._lagged(df, ["player_id"], ["carries"], window=5, suffix="t5")
    a2 = out[(out.player_id == "a") & (out.week == 2)]["carries_t5"].iloc[0]
    assert a2 == 10.0, "player a's window picked up player b's games"


def test_season_expanding_resets_each_season():
    df = pd.DataFrame({
        "player_id": ["a"] * 4,
        "season": [2020, 2020, 2021, 2021],
        "week": [1, 2, 1, 2],
        "carries": [7.0, 7.0, 7.0, 7.0],
    })
    out = F._expanding(df, ["player_id", "season"], ["carries"], suffix="std")
    first_2021 = out[(out.season == 2021) & (out.week == 1)]["carries_std"].iloc[0]
    assert np.isnan(first_2021), "season-to-date carried over from the prior season"


@pytest.mark.slow
def test_features_are_invariant_to_the_future():
    """Truncate the future, rebuild, and demand the same answers.

    Any feature that peeked at a later game would shift when those later games
    are removed from the input.
    """
    if not LABELED_PATH.exists():
        pytest.skip("run src/label.py first")

    cutoff_season, cutoff_week = 2023, 10
    labeled = pd.read_parquet(LABELED_PATH)
    pbp = ingest.load_pbp(SEASONS)
    schedules = ingest.load_schedules(SEASONS)
    snaps = ingest.load_snap_counts(SEASONS)
    rosters = ingest.load_weekly_rosters(SEASONS)
    depth = ingest.load_depth_charts(SEASONS)

    full = F.build_features(labeled, pbp, schedules, snaps, rosters, depth)

    is_past = (pbp["season"] < cutoff_season) | (
        (pbp["season"] == cutoff_season) & (pbp["week"] <= cutoff_week))
    lab_past = labeled[(labeled["season"] < cutoff_season) | (
        (labeled["season"] == cutoff_season) & (labeled["week"] <= cutoff_week))]
    snaps_past = snaps[(snaps["season"] < cutoff_season) | (
        (snaps["season"] == cutoff_season) & (snaps["week"] <= cutoff_week))]
    truncated = F.build_features(lab_past, pbp[is_past], schedules, snaps_past,
                                 rosters, depth)

    key = ["game_id", "team"]
    sel = ((full["season"] == cutoff_season) & (full["week"] == cutoff_week))
    cols = [c for c in F.feature_columns(full) if c in truncated.columns]

    a = full.loc[sel, key + cols].sort_values(key).reset_index(drop=True)
    b = (truncated[(truncated["season"] == cutoff_season)
                   & (truncated["week"] == cutoff_week)][key + cols]
         .sort_values(key).reset_index(drop=True))

    assert len(a) == len(b) and len(a) > 0
    assert a[key].equals(b[key])

    for col in cols:
        x, y = a[col].astype(float), b[col].astype(float)
        same_nulls = x.isna().equals(y.isna())
        close = np.allclose(x.dropna().to_numpy(), y.dropna().to_numpy(),
                            rtol=1e-9, atol=1e-9)
        assert same_nulls and close, f"{col} changed when the future was removed"
