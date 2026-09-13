"""Central configuration: paths, season range, and modeling constants.

Everything that another module might want to tweak lives here so that no
magic numbers are buried inside the pipeline steps.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABELED_PATH = PROCESSED_DIR / "labeled.parquet"
FEATURES_PATH = PROCESSED_DIR / "features.parquet"

# --------------------------------------------------------------------------
# Season range
# --------------------------------------------------------------------------
# nflfastR play-by-play goes back to 1999, but 2013 is the first season of
# Pro-Football-Reference snap counts, which the starter cross-check in
# label.py depends on. Earlier seasons cannot be labeled to the same standard.
FIRST_SEASON = 2013

# The latest season to pull. Keep this current: it is the single constant that
# decides whether the model is training on the most recent completed season or
# a stale one. Pulling an in-progress season is fine and expected; it simply
# contributes however many games have been played.
LAST_SEASON = 2026
SEASONS = list(range(FIRST_SEASON, LAST_SEASON + 1))

# --------------------------------------------------------------------------
# Label definition
# --------------------------------------------------------------------------
# A team-game is a positive if the starting RB's rushing yards on that team's
# first offensive drive sum to at least this many yards.
YARDS_THRESHOLD = 5

# Regular season only. Playoff games have different scripting dynamics and a
# heavily selected team population.
GAME_TYPES = ("REG",)

# --------------------------------------------------------------------------
# Starter resolution
# --------------------------------------------------------------------------
# If snap-count data is unavailable for a game, fall back to the rush-attempt
# leader only when they out-carried the next RB by at least this margin.
MIN_ATTEMPT_MARGIN_NO_SNAPS = 3

# Minimum offensive snap share for a snap-count leader to count as a starter
# rather than a rotation back.
MIN_SNAP_PCT = 0.35

# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
TRAILING_WINDOW = 5  # trailing N-game rolling windows

# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
# Walk-forward CV: train on everything before a season, validate on it.
# Seasons used as validation folds (each trained on all prior seasons).
CV_START_SEASON = 2020

# Final held-out test season, never touched during model selection. This must
# be the last *completed* season, which is not the same as LAST_SEASON once an
# in-progress season is being pulled. Models scored against this are trained
# only on earlier seasons; the model used for live inference is trained on
# everything available and saved separately (see train.py).
TEST_SEASON = 2025

RANDOM_STATE = 1701
