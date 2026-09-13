"""Training and inference must compute identical features.

Train/serve skew is the classic way a model that validates well produces
garbage in production: the offline path and the online path drift apart and
nobody notices because they are never compared. These tests compare them
directly on a replayed past week.
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
import predict as P  # noqa: E402
from config import FEATURES_PATH, SEASONS  # noqa: E402

SEASON, WEEK = 2023, 12


@pytest.fixture(scope="module")
def offline() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip("run src/features.py first")
    return pd.read_parquet(FEATURES_PATH)


@pytest.mark.slow
def test_inference_features_match_training_features(offline):
    """Replay a past week through the inference path and demand agreement."""
    pbp = ingest.load_pbp(SEASONS)
    schedules = ingest.load_schedules(SEASONS)

    truth = offline[(offline["season"] == SEASON) & (offline["week"] == WEEK)
                    & (offline["is_ambiguous"] == 0)]
    assert len(truth) > 0

    targets = P.week_targets(schedules, SEASON, WEEK)
    # Use the resolved starters from the labeled data so this test isolates
    # feature computation; starter projection is tested separately.
    targets = targets.merge(
        truth[["game_id", "team", "starter_id"]], on=["game_id", "team"],
        how="inner")

    online = F.build_inference_features(targets, pbp, schedules)

    cols = [c for c in F.feature_columns(offline) if c in online.columns]
    key = ["game_id", "team"]
    a = truth[key + cols].sort_values(key).reset_index(drop=True)
    b = online[key + cols].sort_values(key).reset_index(drop=True)

    assert len(a) == len(b) and len(a) > 0
    assert a[key].equals(b[key])

    for col in cols:
        x, y = a[col].astype(float), b[col].astype(float)
        assert x.isna().equals(y.isna()), f"{col}: null pattern differs"
        assert np.allclose(x.dropna().to_numpy(), y.dropna().to_numpy(),
                           rtol=1e-9, atol=1e-9), f"{col}: values differ"


def test_projected_starters_never_use_in_game_usage():
    """The projection may only read the depth chart and the injury report."""
    depth = ingest.load_depth_charts(SEASONS)
    inj = ingest.load_injuries(SEASONS)
    teams = ["KC", "SF", "BAL", "PHI"]

    proj = P.project_starters(depth, inj, SEASON, WEEK, teams)
    assert set(proj["team"]) == set(teams)
    assert proj["starter_id"].notna().any()

    # Every projected starter must appear on that week's published depth chart.
    listed = P.depth_chart_rbs(depth, SEASON, WEEK)
    for _, row in proj[proj["starter_id"].notna()].iterrows():
        pool = set(listed.loc[listed["team"] == row["team"], "gsis_id"])
        assert row["starter_id"] in pool


def test_players_ruled_out_are_never_projected_to_start():
    depth = ingest.load_depth_charts(SEASONS)
    inj = ingest.load_injuries(SEASONS)
    blocked = P.blocked_players(inj, SEASON, WEEK)
    teams = sorted(P.depth_chart_rbs(depth, SEASON, WEEK)["team"].unique())

    proj = P.project_starters(depth, inj, SEASON, WEEK, teams)
    picked = set(proj["starter_id"].dropna())
    assert not (picked & blocked), "projected a back who was Out or Doubtful"


def test_promotion_is_flagged_for_review():
    """Any deviation from 'the listed RB1 starts' must be auditable."""
    depth = ingest.load_depth_charts(SEASONS)
    inj = ingest.load_injuries(SEASONS)
    teams = sorted(P.depth_chart_rbs(depth, SEASON, WEEK)["team"].unique())

    proj = P.project_starters(depth, inj, SEASON, WEEK, teams)
    promoted = proj[proj["depth_rank"] > 1]
    assert (promoted["needs_review"] == 1).all()
    assert (promoted["note"].str.len() > 0).all()
