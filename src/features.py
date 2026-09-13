"""Pregame feature engineering.

Every feature here answers the question "what did we know before kickoff?".
The mechanism that enforces that is :func:`_lagged`, which shifts a team's or
player's own game log forward by one game before any rolling window touches it.
A team's Week 8 row therefore sees Weeks 1-7 and nothing else.

Rate features (yards per carry, snap shares, run rates) are built by rolling
the numerator and denominator separately and dividing at the end. Rolling the
per-game ratio instead would weight a 3-carry game as heavily as a 25-carry
game.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import FEATURES_PATH, LABELED_PATH, SEASONS, TRAILING_WINDOW
from label import rush_plays, standardize_team

log = logging.getLogger(__name__)

# Order a game log by real time so "previous game" means what it should.
ORDER = ["season", "week"]


def _lagged(df: pd.DataFrame, keys: list[str], cols: list[str], window: int,
            suffix: str) -> pd.DataFrame:
    """Trailing-``window`` sums of ``cols`` using only prior games.

    The ``shift(1)`` happens inside the group, before the rolling window, so
    the current game can never contribute to its own feature.
    """
    df = df.sort_values(keys + ORDER).copy()
    grouped = df.groupby(keys, sort=False)
    for col in cols:
        df[f"{col}_{suffix}"] = grouped[col].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).sum())
    return df


def _expanding(df: pd.DataFrame, keys: list[str], cols: list[str],
               suffix: str) -> pd.DataFrame:
    """Season-to-date sums of ``cols`` using only prior games."""
    df = df.sort_values(keys + ORDER).copy()
    grouped = df.groupby(keys, sort=False)
    for col in cols:
        df[f"{col}_{suffix}"] = grouped[col].transform(
            lambda s: s.shift(1).expanding(min_periods=1).sum())
    return df


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return np.where(den > 0, num / den.replace(0, np.nan), np.nan)


# --------------------------------------------------------------------------
# Game logs
# --------------------------------------------------------------------------
def player_game_logs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-game rushing usage, including first-drive usage."""
    runs = rush_plays(pbp).copy()
    runs["team"] = standardize_team(runs["posteam"])

    first = (pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
             .groupby(["game_id", "posteam"], as_index=False)["fixed_drive"]
             .min().rename(columns={"fixed_drive": "first_drive"}))
    runs = runs.merge(first, on=["game_id", "posteam"], how="left")

    runs["on_first_drive"] = (runs["fixed_drive"] == runs["first_drive"]).astype(int)
    runs["fd_carry"] = runs["on_first_drive"]
    runs["fd_yards"] = runs["rushing_yards"] * runs["on_first_drive"]
    runs["rz_carry"] = (runs["yardline_100"] <= 20).astype(int)
    runs["stuffed"] = (runs["rushing_yards"] <= 0).astype(int)
    runs["explosive"] = (runs["rushing_yards"] >= 10).astype(int)

    team_tot = (runs.groupby(["game_id", "team"], as_index=False)
                    .agg(team_carries=("rush_attempt", "sum"),
                         team_fd_carries=("fd_carry", "sum")))

    logs = (runs.groupby(["game_id", "season", "week", "team",
                          "rusher_player_id"], as_index=False)
                .agg(carries=("rush_attempt", "sum"),
                     rush_yards=("rushing_yards", "sum"),
                     rush_epa=("epa", "sum"),
                     fd_carries=("fd_carry", "sum"),
                     fd_yards=("fd_yards", "sum"),
                     rz_carries=("rz_carry", "sum"),
                     stuffs=("stuffed", "sum"),
                     explosives=("explosive", "sum"))
                .rename(columns={"rusher_player_id": "player_id"}))
    logs = logs.merge(team_tot, on=["game_id", "team"], how="left")
    logs["games"] = 1
    # Did this player clear the threshold on the opening drive in this game?
    logs["fd_hit"] = (logs["fd_yards"] >= 5).astype(int)
    # Did this player touch the ball on the opener at all? Rolling this is the
    # "first-drive rush share" the product plan asks for.
    logs["fd_played"] = (logs["fd_carries"] > 0).astype(int)
    return logs


def player_snap_logs(snaps: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    """Per player-game offensive snap share, keyed to gsis_id.

    Snap share is a cleaner read on a back's role than carries are: it counts
    the passing downs he stayed on the field for, which is exactly what
    separates a bell-cow from a early-down-only back who cedes the opener.
    """
    from label import rb_player_weeks, rb_snap_shares

    rb = rb_player_weeks(rosters)
    share = rb_snap_shares(snaps, rb)
    meta = (snaps[["game_id", "season", "week"]]
            .drop_duplicates(subset=["game_id"]))
    out = share.merge(meta, on="game_id", how="left")
    out = out.rename(columns={"gsis_id": "player_id"})
    out["offense_pct"] = pd.to_numeric(out["offense_pct"], errors="coerce")
    out["games"] = 1
    return out.dropna(subset=["season", "week"])


def team_offense_logs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game offensive rushing and first-drive play-calling."""
    plays = pbp[pbp["posteam"].notna() & pbp["play_type"].isin(["run", "pass"])].copy()
    plays["team"] = standardize_team(plays["posteam"])

    first = (pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
             .groupby(["game_id", "posteam"], as_index=False)["fixed_drive"]
             .min().rename(columns={"fixed_drive": "first_drive"}))
    plays = plays.merge(first, on=["game_id", "posteam"], how="left")
    on_first = plays["fixed_drive"] == plays["first_drive"]

    plays["is_run"] = (plays["play_type"] == "run").astype(int)
    plays["rush_yds"] = plays["rushing_yards"].fillna(0) * plays["is_run"]
    plays["rush_epa_sum"] = plays["epa"].fillna(0) * plays["is_run"]
    # Neutral game script: close game, before garbage time. Blowouts distort
    # run rate in a way that says nothing about a coach's opener tendency.
    neutral = (plays["wp"].between(0.2, 0.8)) & (plays["qtr"] <= 3)
    plays["neutral_plays"] = neutral.astype(int)
    plays["neutral_runs"] = (neutral & (plays["play_type"] == "run")).astype(int)
    plays["fd_plays"] = on_first.astype(int)
    plays["fd_runs"] = (on_first & (plays["play_type"] == "run")).astype(int)
    plays["fd_rush_yds"] = plays["rush_yds"] * on_first.astype(int)

    logs = (plays.groupby(["game_id", "season", "week", "team"], as_index=False)
                 .agg(off_plays=("is_run", "size"),
                      off_rushes=("is_run", "sum"),
                      off_rush_yards=("rush_yds", "sum"),
                      off_rush_epa=("rush_epa_sum", "sum"),
                      off_fd_plays=("fd_plays", "sum"),
                      off_fd_runs=("fd_runs", "sum"),
                      off_fd_rush_yards=("fd_rush_yds", "sum"),
                      off_neutral_plays=("neutral_plays", "sum"),
                      off_neutral_runs=("neutral_runs", "sum")))
    logs["games"] = 1
    return logs


def team_defense_logs(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per team-game run defense, keyed by the defending team."""
    runs = rush_plays(pbp).copy()
    runs["team"] = standardize_team(runs["defteam"])
    runs = runs[runs["team"].notna()]
    runs["stuffed"] = (runs["rushing_yards"] <= 0).astype(int)

    logs = (runs.groupby(["game_id", "season", "week", "team"], as_index=False)
                .agg(def_rushes_faced=("rush_attempt", "sum"),
                     def_rush_yards_allowed=("rushing_yards", "sum"),
                     def_rush_epa_allowed=("epa", "sum"),
                     def_stuffs=("stuffed", "sum")))
    logs["games"] = 1
    return logs


# --------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------
W = TRAILING_WINDOW


def build_player_features(logs: pd.DataFrame) -> pd.DataFrame:
    cols = ["carries", "rush_yards", "rush_epa", "fd_carries", "fd_yards",
            "rz_carries", "stuffs", "explosives", "games", "fd_hit", "fd_played",
            "team_carries", "team_fd_carries"]
    out = _lagged(logs, ["player_id"], cols, W, f"t{W}")
    out = _expanding(out, ["player_id", "season"], cols, "std")

    t = f"t{W}"
    out[f"rb_ypc_{t}"] = _safe_div(out[f"rush_yards_{t}"], out[f"carries_{t}"])
    out["rb_ypc_season"] = _safe_div(out["rush_yards_std"], out["carries_std"])
    out[f"rb_carries_per_game_{t}"] = _safe_div(out[f"carries_{t}"], out[f"games_{t}"])
    out[f"rb_fd_carries_per_game_{t}"] = _safe_div(out[f"fd_carries_{t}"],
                                                   out[f"games_{t}"])
    out[f"rb_fd_yards_per_game_{t}"] = _safe_div(out[f"fd_yards_{t}"],
                                                 out[f"games_{t}"])
    # Share of this back's recent games in which he carried on the opener.
    # Roughly one usable team-game in six has the starter take zero opening
    # carries, so this separates bell-cows from backs who cede the script.
    out[f"rb_fd_participation_{t}"] = _safe_div(out[f"fd_played_{t}"],
                                                out[f"games_{t}"])
    out["rb_fd_participation_season"] = _safe_div(out["fd_played_std"],
                                                  out["games_std"])
    out[f"rb_fd_hit_rate_{t}"] = _safe_div(out[f"fd_hit_{t}"], out[f"games_{t}"])
    # Backfield concentration. A back taking 80% of his team's carries is a
    # different proposition from one splitting them, even at equal volume.
    out[f"rb_carry_share_{t}"] = _safe_div(out[f"carries_{t}"],
                                           out[f"team_carries_{t}"])
    out[f"rb_fd_carry_share_{t}"] = _safe_div(out[f"fd_carries_{t}"],
                                              out[f"team_fd_carries_{t}"])
    out[f"rb_rz_share_{t}"] = _safe_div(out[f"rz_carries_{t}"], out[f"carries_{t}"])
    out[f"rb_stuff_rate_{t}"] = _safe_div(out[f"stuffs_{t}"], out[f"carries_{t}"])
    out[f"rb_explosive_rate_{t}"] = _safe_div(out[f"explosives_{t}"],
                                              out[f"carries_{t}"])
    out[f"rb_epa_per_carry_{t}"] = _safe_div(out[f"rush_epa_{t}"], out[f"carries_{t}"])
    out["rb_games_played_season"] = out["games_std"]

    keep = ["game_id", "team", "player_id"] + [c for c in out.columns
                                               if c.startswith("rb_")]
    return out[keep]


def build_snap_features(logs: pd.DataFrame) -> pd.DataFrame:
    out = _lagged(logs, ["player_id"], ["offense_pct", "games"], W, f"t{W}")
    t = f"t{W}"
    out[f"rb_snap_share_{t}"] = _safe_div(out[f"offense_pct_{t}"],
                                          out[f"games_{t}"])
    return out[["game_id", "team", "player_id", f"rb_snap_share_{t}"]]


def build_team_features(logs: pd.DataFrame) -> pd.DataFrame:
    cols = ["off_plays", "off_rushes", "off_rush_yards", "off_rush_epa",
            "off_fd_plays", "off_fd_runs", "off_fd_rush_yards", "games",
            "off_neutral_plays", "off_neutral_runs"]
    out = _lagged(logs, ["team"], cols, W, f"t{W}")
    out = _expanding(out, ["team", "season"], cols, "std")

    t = f"t{W}"
    out[f"tm_run_rate_{t}"] = _safe_div(out[f"off_rushes_{t}"], out[f"off_plays_{t}"])
    out[f"tm_fd_run_rate_{t}"] = _safe_div(out[f"off_fd_runs_{t}"],
                                           out[f"off_fd_plays_{t}"])
    out["tm_fd_run_rate_season"] = _safe_div(out["off_fd_runs_std"],
                                             out["off_fd_plays_std"])
    # The scripted-opener signal: how run-heavy the OC's opener has been
    # lately versus his season baseline.
    out[f"tm_fd_run_rate_trend_{t}"] = (out[f"tm_fd_run_rate_{t}"]
                                        - out["tm_fd_run_rate_season"])
    out[f"tm_neutral_run_rate_{t}"] = _safe_div(out[f"off_neutral_runs_{t}"],
                                                out[f"off_neutral_plays_{t}"])
    out[f"tm_rush_epa_per_play_{t}"] = _safe_div(out[f"off_rush_epa_{t}"],
                                                 out[f"off_rushes_{t}"])
    out[f"tm_ypc_{t}"] = _safe_div(out[f"off_rush_yards_{t}"], out[f"off_rushes_{t}"])
    out[f"tm_fd_rush_yards_per_game_{t}"] = _safe_div(out[f"off_fd_rush_yards_{t}"],
                                                      out[f"games_{t}"])
    out[f"tm_fd_plays_per_game_{t}"] = _safe_div(out[f"off_fd_plays_{t}"],
                                                 out[f"games_{t}"])

    keep = ["game_id", "team"] + [c for c in out.columns if c.startswith("tm_")]
    return out[keep]


def build_defense_features(logs: pd.DataFrame) -> pd.DataFrame:
    cols = ["def_rushes_faced", "def_rush_yards_allowed", "def_rush_epa_allowed",
            "def_stuffs", "games"]
    out = _lagged(logs, ["team"], cols, W, f"t{W}")

    t = f"t{W}"
    out[f"opp_ypc_allowed_{t}"] = _safe_div(out[f"def_rush_yards_allowed_{t}"],
                                            out[f"def_rushes_faced_{t}"])
    out[f"opp_stuff_rate_{t}"] = _safe_div(out[f"def_stuffs_{t}"],
                                           out[f"def_rushes_faced_{t}"])
    out[f"opp_rush_epa_allowed_{t}"] = _safe_div(out[f"def_rush_epa_allowed_{t}"],
                                                 out[f"def_rushes_faced_{t}"])

    keep = ["game_id", "team"] + [c for c in out.columns if c.startswith("opp_")]
    return out[keep].rename(columns={"team": "opponent"})


def depth_rank_features(depth_charts: pd.DataFrame) -> pd.DataFrame:
    """Published depth-chart standing for each RB in each week.

    Unlike the rolling features this is a current-week value, not a lagged one,
    and that is legitimate: depth charts are published before kickoff. It is
    the same signal inference leans on to pick a starter, so giving the model
    access to it in training keeps the two paths honest with each other.
    """
    from label import standardize_team

    dc = depth_charts[(depth_charts["position"] == "RB")
                      & (depth_charts["formation"] == "Offense")].copy()
    dc["team"] = standardize_team(dc["club_code"])
    dc["depth"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    dc["week"] = pd.to_numeric(dc["week"], errors="coerce")
    dc = dc.dropna(subset=["gsis_id", "depth", "week"])

    # How many backs the team lists as RB1: 2+ is a declared committee.
    n_top = (dc[dc["depth"] == 1]
             .groupby(["season", "week", "team"], as_index=False)["gsis_id"]
             .nunique().rename(columns={"gsis_id": "rb_n_listed_first"}))

    out = (dc.groupby(["season", "week", "team", "gsis_id"], as_index=False)["depth"]
             .min().rename(columns={"depth": "rb_depth_rank",
                                    "gsis_id": "player_id"}))
    return out.merge(n_top, on=["season", "week", "team"], how="left")


def add_context_features(df: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Game context that is public before kickoff: lines, rest, venue, weather."""
    sched = schedules.copy()
    sched["home_team"] = standardize_team(sched["home_team"])
    sched["away_team"] = standardize_team(sched["away_team"])
    rest = pd.concat([
        sched[["game_id", "home_team", "home_rest"]]
        .rename(columns={"home_team": "team", "home_rest": "rest_days"}),
        sched[["game_id", "away_team", "away_rest"]]
        .rename(columns={"away_team": "team", "away_rest": "rest_days"}),
    ], ignore_index=True)
    df = df.merge(rest, on=["game_id", "team"], how="left")
    if "div_game" in sched.columns:
        df = df.merge(sched[["game_id", "div_game"]], on="game_id", how="left")

    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    # Temperature and wind are meaningless indoors; neutralize rather than
    # leave a null the model reads as an outdoor extreme.
    df["temp_f"] = np.where(df["is_dome"] == 1, 70.0, df["temp"])
    df["wind_mph"] = np.where(df["is_dome"] == 1, 0.0, df["wind"])
    df["is_grass"] = (df["surface"] == "grass").astype(int)
    df["implied_team_total"] = df["total_line"] / 2 + df["team_spread"] / 2
    return df


HEURISTIC_COL = f"rb_fd_yards_per_game_t{W}"


def build_features(labeled: pd.DataFrame, pbp: pd.DataFrame,
                   schedules: pd.DataFrame, snaps: pd.DataFrame,
                   rosters: pd.DataFrame,
                   depth_charts: pd.DataFrame) -> pd.DataFrame:
    plog = player_game_logs(pbp)
    tlog = team_offense_logs(pbp)
    dlog = team_defense_logs(pbp)
    slog = player_snap_logs(snaps, rosters)

    pf = build_player_features(plog)
    tf = build_team_features(tlog)
    df_ = build_defense_features(dlog)
    sf = build_snap_features(slog)
    dr = depth_rank_features(depth_charts)

    out = labeled.merge(pf, left_on=["game_id", "team", "starter_id"],
                        right_on=["game_id", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(sf, left_on=["game_id", "team", "starter_id"],
                    right_on=["game_id", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(dr, left_on=["season", "week", "team", "starter_id"],
                    right_on=["season", "week", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(tf, on=["game_id", "team"], how="left")
    out = out.merge(df_, on=["game_id", "opponent"], how="left")
    out = add_context_features(out, schedules)
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


FEATURE_COLUMNS = None  # resolved at runtime by feature_columns()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs: the engineered pregame columns, nothing derived from the
    game being predicted."""
    # The rb_/tm_/opp_ prefixes already cover the depth-chart columns, which
    # are pregame-legal despite not being lagged (see depth_rank_features).
    prefixes = ("rb_", "tm_", "opp_")
    engineered = [c for c in df.columns if c.startswith(prefixes)]
    context = ["team_spread", "total_line", "implied_team_total", "is_home",
               "rest_days", "is_dome", "temp_f", "wind_mph", "is_grass",
               "div_game", "week"]
    context = [c for c in context if c in df.columns]
    return engineered + context


# Columns that describe the outcome of the game being predicted. Anything in
# here reaching a model is leakage by definition; tests assert on this list.
LEAKY_COLUMNS = [
    "fd_rush_yards", "fd_carries", "label", "starter_attempts",
    "runner_up_attempts", "attempt_margin", "starter_snap_pct",
    "snap_leader_pct", "drive_plays", "drive_start_yardline", "n_rb_rushers",
    # Decided by the coin toss, so it is not knowable before kickoff even
    # though it is not a game *outcome*. It is kept on the labeled frame
    # because it is useful for analysis, but it must never be a model input.
    # Measured cost of excluding it: none. Walk-forward log loss was 0.6848
    # with it and 0.6847 without.
    "received_opening_kickoff",
]




# --------------------------------------------------------------------------
# Inference-time features
# --------------------------------------------------------------------------
def build_inference_features(targets: pd.DataFrame, pbp: pd.DataFrame,
                             schedules: pd.DataFrame, snaps: pd.DataFrame,
                             rosters: pd.DataFrame,
                             depth_charts: pd.DataFrame) -> pd.DataFrame:
    """Features for games that have not been played yet.

    Training and serving must not drift apart, so this reuses the exact
    aggregation and lag functions the training path uses. The trick is to
    append a placeholder game-log row for each upcoming team-game with zeroed
    stats, then run the normal pipeline: because :func:`_lagged` shifts within
    the group *before* rolling, a placeholder's own zeros can never enter its
    own features, and it picks up precisely the trailing window that a real row
    in that slot would have.

    ``pbp`` is truncated to strictly before the target week first, so replaying
    a past week produces the same numbers that would have been available on the
    morning of that game.
    """
    season = int(targets["season"].iloc[0])
    week = int(targets["week"].iloc[0])
    history = pbp[(pbp["season"] < season)
                  | ((pbp["season"] == season) & (pbp["week"] < week))]

    snap_hist = snaps[(snaps["season"] < season)
                      | ((snaps["season"] == season) & (snaps["week"] < week))]

    plog = player_game_logs(history)
    tlog = team_offense_logs(history)
    dlog = team_defense_logs(history)
    slog = player_snap_logs(snap_hist, rosters)

    def _placeholders(template: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
        rows = keys.copy()
        for col in template.columns:
            if col not in rows.columns:
                rows[col] = 0 if template[col].dtype.kind in "biufc" else None
        return pd.concat([template, rows[template.columns]], ignore_index=True)

    p_keys = (targets[["game_id", "season", "week", "team", "starter_id"]]
              .rename(columns={"starter_id": "player_id"}))
    t_keys = targets[["game_id", "season", "week", "team"]]
    d_keys = (targets[["game_id", "season", "week", "opponent"]]
              .rename(columns={"opponent": "team"}))

    pf = build_player_features(_placeholders(plog, p_keys))
    tf = build_team_features(_placeholders(tlog, t_keys))
    df_ = build_defense_features(_placeholders(dlog, d_keys))
    sf = build_snap_features(_placeholders(slog, p_keys))
    dr = depth_rank_features(depth_charts)

    out = targets.merge(pf, left_on=["game_id", "team", "starter_id"],
                        right_on=["game_id", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(sf, left_on=["game_id", "team", "starter_id"],
                    right_on=["game_id", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(dr, left_on=["season", "week", "team", "starter_id"],
                    right_on=["season", "week", "team", "player_id"], how="left")
    out = out.drop(columns=["player_id"])
    out = out.merge(tf, on=["game_id", "team"], how="left")
    out = out.merge(df_, on=["game_id", "opponent"], how="left")
    return add_context_features(out, schedules)



def main(refresh: bool = False) -> pd.DataFrame:
    import ingest

    labeled = pd.read_parquet(LABELED_PATH)
    pbp = ingest.load_pbp(SEASONS, refresh)
    schedules = ingest.load_schedules(SEASONS, refresh)
    snaps = ingest.load_snap_counts(SEASONS, refresh)
    rosters = ingest.load_weekly_rosters(SEASONS, refresh)
    depth_charts = ingest.load_depth_charts(SEASONS, refresh)

    feats = build_features(labeled, pbp, schedules, snaps, rosters, depth_charts)
    feats.to_parquet(FEATURES_PATH, index=False)

    cols = feature_columns(feats)
    usable = feats[feats["is_ambiguous"] == 0]
    log.info("feature rows: %s (%s usable)", len(feats), len(usable))
    log.info("feature columns: %s", len(cols))
    log.info("null rate on usable rows:\n%s",
             usable[cols].isna().mean().sort_values(ascending=False)
             .head(12).to_string())
    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
