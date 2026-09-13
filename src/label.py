"""Drive-level label construction and starting-RB resolution.

The label: for each team's *first offensive drive* of a game, did the starting
running back total at least ``YARDS_THRESHOLD`` rushing yards on that drive?

Two details in that sentence do the heavy lifting:

1. "First offensive drive" is per team, not per game. The team that kicks off
   opens the game on defense, so its first possession is the game's second
   drive. We take each team's minimum ``fixed_drive`` within the game.

2. "The starting RB" is the single biggest failure point in this project (see
   README). For training labels we follow the product plan and use the RB with
   the most rush attempts in that game as a proxy, cross-checked against
   Pro-Football-Reference offensive snap shares. Games where the two disagree,
   or where nothing resolves cleanly, are dropped rather than force-labeled.

Note on penalties: nflfastR marks plays negated by penalty as
``play_type == 'no_play'`` and never sets ``rush_attempt`` on them, so
filtering to ``play_type == 'run' & rush_attempt == 1`` already excludes
negated carries. Run plays that carry a penalty flag but still counted (an
offsetting or declined penalty, say) keep their rushing yards and are included,
which is correct since those yards stood.
"""
from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from config import (LABELED_PATH, MIN_ATTEMPT_MARGIN_NO_SNAPS, MIN_SNAP_PCT,
                    SEASONS, YARDS_THRESHOLD)

log = logging.getLogger(__name__)

# nflfastR back-applies current franchise codes to old seasons; the roster and
# snap-count feeds do not. Map everything onto the nflfastR convention.
TEAM_FIXES = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "WSH": "WAS"}


def standardize_team(s: pd.Series) -> pd.Series:
    return s.astype("object").replace(TEAM_FIXES)


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(s: pd.Series) -> pd.Series:
    """Fold a full name to a join-safe key: ascii, lowercase, no punctuation."""
    out = (s.astype("object")
            .fillna("")
            .map(lambda x: unicodedata.normalize("NFKD", str(x))
                 .encode("ascii", "ignore").decode("ascii"))
            .str.lower()
            .map(lambda x: re.sub(r"[^a-z ]", "", x))
            .str.split()
            .map(lambda parts: " ".join(p for p in parts if p not in _SUFFIXES)))
    return out


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def rush_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Designed and scrambled runs that actually counted."""
    mask = (
        (pbp["play_type"] == "run")
        & (pbp["rush_attempt"] == 1)
        & (pbp["two_point_attempt"] != 1)
        & pbp["rusher_player_id"].notna()
        & pbp["posteam"].notna()
    )
    out = pbp.loc[mask].copy()
    out["rushing_yards"] = out["rushing_yards"].fillna(0.0)
    return out


def rb_player_weeks(rosters: pd.DataFrame) -> pd.DataFrame:
    """Every (season, week, team, player) row where the player is an RB."""
    is_rb = (rosters["position"] == "RB") | (rosters["depth_chart_position"] == "RB")
    rb = rosters.loc[is_rb, ["season", "week", "team", "gsis_id", "full_name",
                             "pfr_id"]].copy()
    rb = rb.dropna(subset=["gsis_id"])
    rb["team"] = standardize_team(rb["team"])
    rb["name_key"] = normalize_name(rb["full_name"])
    return rb.drop_duplicates(subset=["season", "week", "team", "gsis_id"])


def game_rush_attempts(runs: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    """Per game-team-player rush attempts, restricted to rostered RBs.

    Restricting to RBs is what keeps QB scrambles and WR jet sweeps from ever
    being mistaken for the starting back.
    """
    usage = (runs.groupby(["game_id", "season", "week", "posteam",
                           "rusher_player_id"], as_index=False)
                 .agg(attempts=("rush_attempt", "sum"),
                      game_rush_yards=("rushing_yards", "sum")))
    usage = usage.rename(columns={"posteam": "team",
                                  "rusher_player_id": "gsis_id"})
    usage["team"] = standardize_team(usage["team"])
    rb_keys = rb[["season", "week", "team", "gsis_id", "full_name", "name_key"]]
    return usage.merge(rb_keys, on=["season", "week", "team", "gsis_id"],
                       how="inner")


def rb_snap_shares(snaps: pd.DataFrame, rb: pd.DataFrame) -> pd.DataFrame:
    """Offensive snap share per game-team-RB, keyed back to gsis_id.

    Snap counts carry only a PFR player id, and the roster feed is missing
    ``pfr_id`` for about a third of RBs, so we join on the PFR id where we have
    it and fall back to a normalized name within the same season and team.
    """
    sn = snaps.loc[snaps["position"] == "RB",
                   ["game_id", "season", "week", "team", "player",
                    "pfr_player_id", "offense_snaps", "offense_pct"]].copy()
    sn["team"] = standardize_team(sn["team"])
    sn["name_key"] = normalize_name(sn["player"])

    by_pfr = (rb.dropna(subset=["pfr_id"])
                .drop_duplicates(subset=["season", "team", "pfr_id"])
                [["season", "team", "pfr_id", "gsis_id"]])
    merged = sn.merge(by_pfr, left_on=["season", "team", "pfr_player_id"],
                      right_on=["season", "team", "pfr_id"], how="left")

    by_name = (rb.drop_duplicates(subset=["season", "team", "name_key"])
                 [["season", "team", "name_key", "gsis_id"]]
                 .rename(columns={"gsis_id": "gsis_id_name"}))
    merged = merged.merge(by_name, on=["season", "team", "name_key"], how="left")
    merged["gsis_id"] = merged["gsis_id"].fillna(merged["gsis_id_name"])

    merged = merged.dropna(subset=["gsis_id"])
    merged["offense_pct"] = pd.to_numeric(merged["offense_pct"], errors="coerce")
    return (merged[["game_id", "team", "gsis_id", "offense_snaps", "offense_pct"]]
            .drop_duplicates(subset=["game_id", "team", "gsis_id"]))


# --------------------------------------------------------------------------
# Starter resolution
# --------------------------------------------------------------------------
def resolve_starters(usage: pd.DataFrame, snap_share: pd.DataFrame) -> pd.DataFrame:
    """Pick one starting RB per team-game, or mark the team-game ambiguous.

    Resolution rules, in order:

    * ``confirmed``  - the rush-attempt leader is also the offensive snap
      leader among RBs and cleared ``MIN_SNAP_PCT``. Both signals agree.
    * ``usage_only`` - no usable snap data, but the attempt leader out-carried
      the next RB by ``MIN_ATTEMPT_MARGIN_NO_SNAPS`` or more.
    * dropped        - a tie in carries, a usage/snap disagreement (the
      committee-backfield case), or a thin margin with no snap data.
    """
    df = usage.merge(snap_share, on=["game_id", "team", "gsis_id"], how="left")

    # Rank by carries, breaking ties on in-game rushing yards only to make the
    # ordering deterministic; a true tie in carries is still flagged below.
    df = df.sort_values(["game_id", "team", "attempts", "game_rush_yards"],
                        ascending=[True, True, False, False])
    grp = df.groupby(["game_id", "team"], sort=False)
    df["rb_rank"] = grp.cumcount()
    df["n_rb_rushers"] = grp["gsis_id"].transform("size")

    top = df[df["rb_rank"] == 0].copy()
    top = top.rename(columns={"gsis_id": "starter_id",
                              "full_name": "starter_name",
                              "attempts": "starter_attempts",
                              "offense_pct": "starter_snap_pct"})

    second = (df.loc[df["rb_rank"] == 1, ["game_id", "team", "attempts"]]
                .rename(columns={"attempts": "runner_up_attempts"}))
    top = top.merge(second, on=["game_id", "team"], how="left")
    top["runner_up_attempts"] = top["runner_up_attempts"].fillna(0)
    top["attempt_margin"] = top["starter_attempts"] - top["runner_up_attempts"]

    # Snap leader among RBs for the same team-game.
    snaps_ranked = snap_share.dropna(subset=["offense_pct"]).sort_values(
        ["game_id", "team", "offense_pct"], ascending=[True, True, False])
    snap_leader = (snaps_ranked.groupby(["game_id", "team"], as_index=False)
                               .head(1)
                               .rename(columns={"gsis_id": "snap_leader_id",
                                                "offense_pct": "snap_leader_pct"})
                               [["game_id", "team", "snap_leader_id",
                                 "snap_leader_pct"]])
    top = top.merge(snap_leader, on=["game_id", "team"], how="left")

    has_snaps = top["snap_leader_id"].notna()
    agrees = has_snaps & (top["snap_leader_id"] == top["starter_id"])
    clears_bar = top["snap_leader_pct"].fillna(0) >= MIN_SNAP_PCT
    tied = top["attempt_margin"] <= 0
    wide_margin = top["attempt_margin"] >= MIN_ATTEMPT_MARGIN_NO_SNAPS

    confirmed = agrees & clears_bar & ~tied
    usage_only = ~has_snaps & wide_margin & ~tied

    top["starter_method"] = np.select(
        [confirmed, usage_only], ["confirmed", "usage_only"], default="ambiguous")
    top["drop_reason"] = np.select(
        [confirmed | usage_only,
         tied,
         has_snaps & ~agrees,
         has_snaps & agrees & ~clears_bar],
        ["", "tied_carries", "usage_snap_disagreement", "snap_share_below_floor"],
        default="thin_margin_no_snaps")

    cols = ["game_id", "season", "week", "team", "starter_id", "starter_name",
            "starter_attempts", "runner_up_attempts", "attempt_margin",
            "n_rb_rushers", "starter_snap_pct", "snap_leader_pct",
            "starter_method", "drop_reason"]
    return top[cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# First drive + label
# --------------------------------------------------------------------------
def first_drives(pbp: pd.DataFrame) -> pd.DataFrame:
    """Each team's first offensive drive of each game, with drive context."""
    plays = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()].copy()
    first = (plays.groupby(["game_id", "posteam"], as_index=False)["fixed_drive"]
                  .min()
                  .rename(columns={"fixed_drive": "first_drive"}))
    drive_plays = plays.merge(
        first, left_on=["game_id", "posteam", "fixed_drive"],
        right_on=["game_id", "posteam", "first_drive"], how="inner")

    context = (drive_plays.groupby(["game_id", "posteam"], as_index=False)
               .agg(first_drive=("first_drive", "first"),
                    drive_plays=("play_id", "count"),
                    drive_start_yardline=("yardline_100", "first")))
    context = context.rename(columns={"posteam": "team"})
    context["team"] = standardize_team(context["team"])
    # Whether this team also opened the game, i.e. it received the kickoff.
    # Receiving teams get a scripted opener from a standing start; the other
    # team's opener follows whatever the receiving team just did.
    context["received_opening_kickoff"] = (context["first_drive"] == 1).astype(int)

    # Keep posteam intact so rush_plays() can still be applied downstream.
    drive_plays = drive_plays.copy()
    drive_plays["team"] = standardize_team(drive_plays["posteam"])
    return drive_plays, context


def build_labels(pbp: pd.DataFrame, rosters: pd.DataFrame,
                 snaps: pd.DataFrame) -> pd.DataFrame:
    """Assemble the labeled team-game dataset."""
    rb = rb_player_weeks(rosters)
    runs = rush_plays(pbp)
    usage = game_rush_attempts(runs, rb)
    snap_share = rb_snap_shares(snaps, rb)
    starters = resolve_starters(usage, snap_share)

    drive_plays, context = first_drives(pbp)

    # Rushing yards by the resolved starter on that first drive.
    fd_runs = rush_plays(drive_plays)
    fd_runs = fd_runs.rename(columns={"rusher_player_id": "gsis_id"})
    fd_usage = (fd_runs.groupby(["game_id", "team", "gsis_id"], as_index=False)
                       .agg(fd_rush_yards=("rushing_yards", "sum"),
                            fd_carries=("rush_attempt", "sum")))

    out = starters.merge(context, on=["game_id", "team"], how="left")
    out = out.merge(fd_usage, left_on=["game_id", "team", "starter_id"],
                    right_on=["game_id", "team", "gsis_id"], how="left")
    out = out.drop(columns=["gsis_id"])

    # A starter who never touched the ball on the opening drive gained zero
    # yards on it. That is a real negative, not a missing value.
    out["fd_rush_yards"] = out["fd_rush_yards"].fillna(0.0)
    out["fd_carries"] = out["fd_carries"].fillna(0).astype(int)
    out["label"] = (out["fd_rush_yards"] >= YARDS_THRESHOLD).astype(int)

    # Opponent and game context, taken from the play-by-play game rows.
    game_meta = (pbp.groupby("game_id", as_index=False)
                    .agg(home_team=("home_team", "first"),
                         away_team=("away_team", "first"),
                         game_date=("game_date", "first"),
                         spread_line=("spread_line", "first"),
                         total_line=("total_line", "first"),
                         roof=("roof", "first"),
                         surface=("surface", "first"),
                         temp=("temp", "first"),
                         wind=("wind", "first")))
    game_meta["home_team"] = standardize_team(game_meta["home_team"])
    game_meta["away_team"] = standardize_team(game_meta["away_team"])
    out = out.merge(game_meta, on="game_id", how="left")
    out["is_home"] = (out["team"] == out["home_team"]).astype(int)
    out["opponent"] = np.where(out["is_home"] == 1, out["away_team"],
                               out["home_team"])
    # spread_line is quoted from the home team's perspective; flip it so a
    # positive number always means this team is favored.
    out["team_spread"] = np.where(out["is_home"] == 1, out["spread_line"],
                                  -out["spread_line"])

    out["is_ambiguous"] = (out["starter_method"] == "ambiguous").astype(int)
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def main(refresh: bool = False) -> pd.DataFrame:
    import ingest

    pbp = ingest.load_pbp(SEASONS, refresh)
    rosters = ingest.load_weekly_rosters(SEASONS, refresh)
    snaps = ingest.load_snap_counts(SEASONS, refresh)

    labeled = build_labels(pbp, rosters, snaps)
    labeled.to_parquet(LABELED_PATH, index=False)

    total = len(labeled)
    usable = labeled[labeled["is_ambiguous"] == 0]
    log.info("team-games resolved: %s", total)
    log.info("usable (clear starter): %s (%.1f%%)", len(usable),
             100 * len(usable) / max(total, 1))
    log.info("base rate label==1: %.4f", usable["label"].mean())
    log.info("\n%s", labeled["starter_method"].value_counts().to_string())
    log.info("\ndrop reasons:\n%s",
             labeled.loc[labeled.is_ambiguous == 1, "drop_reason"]
             .value_counts().to_string())
    return labeled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
