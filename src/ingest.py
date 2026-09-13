"""Pull and cache the free data sources this project depends on.

All sources are nflverse releases served from GitHub, or the nfldata schedule
mirror. nfl_data_py's own ``import_schedules``/``import_weekly_rosters`` reach
for ``habitatring.com`` over plain HTTP, which is unavailable behind many
corporate egress proxies, so those two are loaded from the GitHub mirrors
directly instead.

Every loader caches to ``data/raw`` as parquet. Pass ``refresh=True`` to force
a re-pull.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from config import GAME_TYPES, RAW_DIR, SEASONS

log = logging.getLogger(__name__)

NFLVERSE_RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
NFLDATA_GAMES = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

# Play-by-play columns we actually use. Pulling all 390+ columns for nine
# seasons is ~2GB of memory for no benefit.
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "game_date",
    "posteam", "defteam", "home_team", "away_team",
    "fixed_drive", "drive", "play_id", "play_type", "down", "ydstogo",
    "yardline_100", "qtr", "goal_to_go",
    "rush_attempt", "pass_attempt", "penalty", "aborted_play",
    "two_point_attempt", "qb_scramble",
    "rusher_player_id", "rusher_player_name", "rushing_yards",
    "epa", "success", "wp",
    "spread_line", "total_line", "roof", "surface", "temp", "wind",
    "desc",
]


def _cache_path(name: str, seasons: Sequence[int]) -> Path:
    span = f"{min(seasons)}_{max(seasons)}"
    return RAW_DIR / f"{name}_{span}.parquet"


def _cached(name: str, seasons: Sequence[int], builder, refresh: bool = False):
    """Return a cached frame, building and caching it on a miss."""
    path = _cache_path(name, seasons)
    if path.exists() and not refresh:
        log.info("cache hit: %s", path.name)
        return pd.read_parquet(path)
    log.info("pulling %s for %s-%s", name, min(seasons), max(seasons))
    df = builder(seasons)
    df.to_parquet(path, index=False)
    log.info("cached %s rows to %s", len(df), path.name)
    return df


# --------------------------------------------------------------------------
# Play-by-play
# --------------------------------------------------------------------------
def _build_pbp(seasons: Sequence[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    df = nfl.import_pbp_data(list(seasons), columns=PBP_COLUMNS, downcast=True,
                             cache=False)
    if GAME_TYPES:
        df = df[df["season_type"].isin(GAME_TYPES)]
    return df.reset_index(drop=True)


def load_pbp(seasons: Iterable[int] = SEASONS, refresh: bool = False) -> pd.DataFrame:
    """Play-by-play, regular season, restricted to the columns we use."""
    return _cached("pbp", list(seasons), _build_pbp, refresh)


# --------------------------------------------------------------------------
# Weekly rosters (position lookup + gsis/pfr crosswalk)
# --------------------------------------------------------------------------
ROSTER_COLUMNS = [
    "season", "week", "team", "position", "depth_chart_position", "status",
    "gsis_id", "pfr_id", "full_name", "years_exp",
]


def _build_weekly_rosters(seasons: Sequence[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        url = f"{NFLVERSE_RELEASE}/weekly_rosters/roster_weekly_{season}.parquet"
        frame = pd.read_parquet(url)
        keep = [c for c in ROSTER_COLUMNS if c in frame.columns]
        frames.append(frame[keep])
    out = pd.concat(frames, ignore_index=True)
    out["week"] = pd.to_numeric(out["week"], errors="coerce")
    return out


def load_weekly_rosters(seasons: Iterable[int] = SEASONS,
                        refresh: bool = False) -> pd.DataFrame:
    """Week-by-week rosters, which is how we know who is an RB."""
    return _cached("weekly_rosters", list(seasons), _build_weekly_rosters, refresh)


# --------------------------------------------------------------------------
# Snap counts (Pro-Football-Reference sourced; our starter cross-check)
# --------------------------------------------------------------------------
def _build_snap_counts(seasons: Sequence[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    df = nfl.import_snap_counts(list(seasons))
    if GAME_TYPES:
        df = df[df["game_type"].isin(GAME_TYPES)]
    return df.reset_index(drop=True)


def load_snap_counts(seasons: Iterable[int] = SEASONS,
                     refresh: bool = False) -> pd.DataFrame:
    """Per-game offensive snap shares, scraped by nflverse from PFR box scores."""
    return _cached("snap_counts", list(seasons), _build_snap_counts, refresh)


# --------------------------------------------------------------------------
# Depth charts (starter signal available *before* kickoff)
# --------------------------------------------------------------------------
def _build_depth_charts(seasons: Sequence[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    df = nfl.import_depth_charts(list(seasons))
    if GAME_TYPES and "game_type" in df.columns:
        df = df[df["game_type"].isin(GAME_TYPES)]
    return df.reset_index(drop=True)


def load_depth_charts(seasons: Iterable[int] = SEASONS,
                      refresh: bool = False) -> pd.DataFrame:
    return _cached("depth_charts", list(seasons), _build_depth_charts, refresh)


# --------------------------------------------------------------------------
# Injury reports (pregame availability for inference)
# --------------------------------------------------------------------------
def _build_injuries(seasons: Sequence[int]) -> pd.DataFrame:
    import nfl_data_py as nfl

    df = nfl.import_injuries(list(seasons))
    if GAME_TYPES and "game_type" in df.columns:
        df = df[df["game_type"].isin(GAME_TYPES)]
    return df.reset_index(drop=True)


def load_injuries(seasons: Iterable[int] = SEASONS,
                  refresh: bool = False) -> pd.DataFrame:
    return _cached("injuries", list(seasons), _build_injuries, refresh)


# --------------------------------------------------------------------------
# Schedules (rest days, kickoff time, betting lines)
# --------------------------------------------------------------------------
SCHEDULE_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "home_team", "away_team", "home_rest", "away_rest", "spread_line",
    "total_line", "roof", "surface", "temp", "wind", "div_game",
]


def _build_schedules(seasons: Sequence[int]) -> pd.DataFrame:
    df = pd.read_csv(NFLDATA_GAMES, low_memory=False)
    df = df[df["season"].isin(list(seasons))]
    if GAME_TYPES:
        df = df[df["game_type"].isin(GAME_TYPES)]
    keep = [c for c in SCHEDULE_COLUMNS if c in df.columns]
    return df[keep].reset_index(drop=True)


def load_schedules(seasons: Iterable[int] = SEASONS,
                   refresh: bool = False) -> pd.DataFrame:
    """Schedules from the nfldata GitHub mirror (not habitatring over HTTP)."""
    return _cached("schedules", list(seasons), _build_schedules, refresh)


def load_all(seasons: Iterable[int] = SEASONS, refresh: bool = False) -> dict:
    """Warm every cache and hand back the frames."""
    seasons = list(seasons)
    return {
        "pbp": load_pbp(seasons, refresh),
        "rosters": load_weekly_rosters(seasons, refresh),
        "snaps": load_snap_counts(seasons, refresh),
        "depth_charts": load_depth_charts(seasons, refresh),
        "injuries": load_injuries(seasons, refresh),
        "schedules": load_schedules(seasons, refresh),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for name, frame in load_all().items():
        print(f"{name:<13} {frame.shape[0]:>8,} rows  {frame.shape[1]:>3} cols")
