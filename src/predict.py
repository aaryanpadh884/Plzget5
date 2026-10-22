"""Weekly inference for upcoming games.

The rule that governs this whole module: **the starting RB is resolved from
pregame information only.** Training labels are allowed to use who actually
carried the ball, because the game is over. Inference is not. Using in-game
usage here would produce a model that looks excellent in backtests and cannot
be bet, because on Sunday morning you do not yet know who got the carries.

So starters come from the published depth chart, filtered by the injury
report, with a documented promotion chain when the listed starter is out. When
that chain cannot produce a confident answer, the team-game is emitted with
``needs_review`` set rather than quietly guessed at.

Usage::

    python src/predict.py --season 2024 --week 10
    python src/predict.py --season 2024 --week 10 --model lgbm --csv out.csv
"""
from __future__ import annotations

import argparse
import logging
import pickle

import numpy as np
import pandas as pd

from config import MODEL_DIR, REPORT_DIR, SEASONS
from features import build_inference_features, feature_columns
from label import standardize_team

log = logging.getLogger(__name__)

def _add_note(notes: pd.Series, mask: np.ndarray, text: str) -> pd.Series:
    """Append a note to the flagged rows, keeping any note already there."""
    notes = notes.fillna("").astype(str)
    existing = notes.where(notes == "", notes + "; ")
    return notes.mask(mask, existing + text)


# Injury designations that rule a back out of starting consideration.
# "Questionable" is deliberately not here: questionable backs start all the
# time. It is surfaced as a warning instead.
BLOCKING_STATUSES = {"Out", "Doubtful"}

# Weekly-roster status codes that mean a back is not available to start.
# Measured over 2013-2026, the share of RBs carrying the ball in a week they
# held each of these codes: INA 0.0%, RET 0.0%, PUP 0.0%, EXE 0.0%, E14 0.0%,
# UFA 0.0%, CUT 0.5%, DEV 0.2%, SUS 1.9%, RSN 3.6%, NWT 6.3%, RES 7.4% --
# against 60.9% for ACT. Codes left out of this set (TRD 44.1%, TRC 36.1%,
# TRT 37.8%, RSR 37.9%) describe mid-season moves where the player does play
# for the team the roster row belongs to, so they stay eligible.
#
# The set is explicit rather than an ACT-only allowlist so an unrecognized
# future code leaves a back eligible and merely warns, instead of silently
# emptying every depth chart.
UNAVAILABLE_ROSTER_STATUSES = {
    "RES", "CUT", "INA", "RET", "PUP", "SUS", "EXE", "E14", "UFA", "RSN",
    "NWT", "DEV", "RFA",
}

# A back's status can lag reality: the week-1 2014 snapshot lists Adrian
# Peterson as SUS even though he played that game and was suspended after it.
# That is 1 of 54 SUS cases, so the signal is kept, but a back skipped for
# status is always reported rather than quietly dropped.
KNOWN_ROSTER_STATUSES = UNAVAILABLE_ROSTER_STATUSES | {
    "ACT", "TRD", "TRC", "TRT", "RSR",
}


# --------------------------------------------------------------------------
# Pregame starter projection
# --------------------------------------------------------------------------
def depth_chart_rbs(depth_charts: pd.DataFrame, season: int,
                    week: int) -> pd.DataFrame:
    """Offensive RBs for one week, ordered by depth-chart position."""
    dc = depth_charts[(depth_charts["season"] == season)
                      & (depth_charts["week"] == week)
                      & (depth_charts["position"] == "RB")
                      & (depth_charts["formation"] == "Offense")].copy()
    dc["team"] = standardize_team(dc["club_code"])
    dc["depth"] = pd.to_numeric(dc["depth_team"], errors="coerce")
    dc = dc.dropna(subset=["gsis_id", "depth"])
    return (dc[["team", "gsis_id", "full_name", "depth"]]
            .drop_duplicates(subset=["team", "gsis_id"])
            .sort_values(["team", "depth", "full_name"]))


def blocked_players(injuries: pd.DataFrame, season: int, week: int) -> set[str]:
    inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
    return set(inj.loc[inj["report_status"].isin(BLOCKING_STATUSES), "gsis_id"]
               .dropna())


def questionable_players(injuries: pd.DataFrame, season: int,
                         week: int) -> set[str]:
    inj = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
    return set(inj.loc[inj["report_status"] == "Questionable", "gsis_id"].dropna())


def roster_status(rosters: pd.DataFrame, season: int,
                  week: int) -> pd.DataFrame:
    """This week's roster status per RB, used to rule out unavailable backs."""
    r = rosters[(rosters["season"] == season) & (rosters["position"] == "RB")].copy()
    r["week"] = pd.to_numeric(r["week"], errors="coerce")
    r = r[r["week"] == week].dropna(subset=["gsis_id"])
    unknown = set(r["status"].dropna().unique()) - KNOWN_ROSTER_STATUSES
    if unknown:
        log.warning("unrecognized roster status codes, treated as available: %s",
                    sorted(unknown))
    return (r[["gsis_id", "status"]]
            .drop_duplicates(subset=["gsis_id"])
            .rename(columns={"status": "roster_status"}))


def _resolve_override(pool: pd.DataFrame, wanted: str):
    """Find a named back on a team's depth chart, by substring, case-insensitive."""
    hit = pool[pool["full_name"].str.contains(wanted, case=False, na=False)]
    if len(hit) == 1:
        return hit.iloc[0], None
    if hit.empty:
        return None, f"{wanted!r} is not on this week's depth chart"
    names = ", ".join(hit["full_name"])
    return None, f"{wanted!r} is ambiguous (matches {names})"


def project_starters(depth_charts: pd.DataFrame, injuries: pd.DataFrame,
                     season: int, week: int, teams: list[str],
                     rosters: pd.DataFrame = None,
                     overrides: dict = None) -> pd.DataFrame:
    """Project each team's starting RB from the depth chart and injury report.

    Promotion chain: take the highest available depth-chart RB. If the listed
    RB1 is Out or Doubtful, the RB2 is promoted, and so on. Every deviation
    from "RB1 starts" is recorded in ``note`` so the output is auditable.
    """
    rbs = depth_chart_rbs(depth_charts, season, week)
    out_ids = blocked_players(injuries, season, week)
    quest_ids = questionable_players(injuries, season, week)

    # A depth chart can keep listing a back who is suspended, on IR, cut or
    # traded away. Roughly 7% of listed RB1s across 2013-2026 carry a roster
    # status that means they are not available, so the chart alone is not
    # enough to pick a starter from.
    status = (roster_status(rosters, season, week) if rosters is not None
              else pd.DataFrame(columns=["gsis_id", "roster_status"]))
    rbs = rbs.merge(status, on="gsis_id", how="left")
    unavailable = rbs["roster_status"].isin(UNAVAILABLE_ROSTER_STATUSES)
    status_by_id = dict(zip(rbs["gsis_id"], rbs["roster_status"]))
    blocked_by_status = set(rbs.loc[unavailable, "gsis_id"])

    overrides = {k.upper(): v for k, v in (overrides or {}).items()}

    rows = []
    for team in teams:
        pool = rbs[rbs["team"] == team]

        # An explicit override wins over every automatic rule. The roster feed
        # can lag reality -- a reinstated back may still read EXE for days --
        # and when the user knows better than the data, they should be able to
        # say so rather than work around the projection.
        if team in overrides and not pool.empty:
            pick, problem = _resolve_override(pool, overrides[team])
            if pick is not None:
                st = status_by_id.get(pick["gsis_id"])
                note = f"MANUAL OVERRIDE: starter set to {pick['full_name']}"
                if st in UNAVAILABLE_ROSTER_STATUSES:
                    note += f" (roster feed still reads {st})"
                rows.append({"team": team, "starter_id": pick["gsis_id"],
                             "starter_name": pick["full_name"],
                             "depth_rank": float(pick["depth"]),
                             "needs_review": 1, "note": note})
                continue
            log.warning("override for %s ignored: %s", team, problem)
        if pool.empty:
            rows.append({"team": team, "starter_id": None, "starter_name": None,
                         "depth_rank": np.nan, "needs_review": 1,
                         "note": "no depth chart published for this team-week"})
            continue

        available = pool[~pool["gsis_id"].isin(out_ids)
                         & ~pool["gsis_id"].isin(blocked_by_status)]
        if available.empty:
            listed = pool.iloc[0]
            why = (f"roster status {status_by_id.get(listed['gsis_id'])}"
                   if listed["gsis_id"] in blocked_by_status
                   else "Out or Doubtful")
            rows.append({"team": team, "starter_id": None,
                         "starter_name": listed["full_name"],
                         "depth_rank": float(listed["depth"]), "needs_review": 1,
                         "note": f"no available listed RB (top choice: {why})"})
            continue

        pick = available.iloc[0]
        top_depth = float(pool["depth"].min())
        note, review = "", 0

        if float(pick["depth"]) > top_depth:
            skipped = pool[pool["depth"] == top_depth].iloc[0]
            reason = (f"roster status "
                      f"{status_by_id.get(skipped['gsis_id'])}"
                      if skipped["gsis_id"] in blocked_by_status
                      else "ruled out")
            note = (f"listed RB1 {skipped['full_name']} skipped ({reason}); "
                    f"promoted from depth {int(pick['depth'])}")
            review = 1
        elif (pool["depth"] == top_depth).sum() > 1:
            note = "multiple RBs listed as RB1 (committee backfield)"
            review = 1

        if pick["gsis_id"] in quest_ids:
            note = (note + "; " if note else "") + "listed Questionable"
            review = 1

        rows.append({"team": team, "starter_id": pick["gsis_id"],
                     "starter_name": pick["full_name"],
                     "depth_rank": float(pick["depth"]),
                     "needs_review": review, "note": note})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Target construction
# --------------------------------------------------------------------------
def week_targets(schedules: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    """One row per team-game for the requested week."""
    games = schedules[(schedules["season"] == season)
                      & (schedules["week"] == week)].copy()
    if games.empty:
        raise ValueError(f"no scheduled games found for {season} week {week}")
    games["home_team"] = standardize_team(games["home_team"])
    games["away_team"] = standardize_team(games["away_team"])

    frames = []
    for side, opp, is_home in (("home_team", "away_team", 1),
                               ("away_team", "home_team", 0)):
        part = games.rename(columns={side: "team", opp: "opponent"}).copy()
        part["is_home"] = is_home
        frames.append(part)
    t = pd.concat(frames, ignore_index=True)

    t["team_spread"] = np.where(t["is_home"] == 1, t["spread_line"],
                                -t["spread_line"])
    keep = ["game_id", "season", "week", "team", "opponent", "is_home",
            "spread_line", "team_spread", "total_line", "roof", "surface",
            "temp", "wind"]
    return t[[c for c in keep if c in t.columns]]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_week(season: int, week: int, model_name: str = "logistic",
                 refresh: bool = False, overrides: dict = None) -> pd.DataFrame:
    import ingest

    # Live inference uses the production fit, trained on every game played,
    # not the evaluation fit that holds out the most recent season.
    bundle_path = MODEL_DIR / "models_production.pkl"
    if not bundle_path.exists():
        bundle_path = MODEL_DIR / "models.pkl"
        log.warning("production models not found, falling back to %s",
                    bundle_path.name)
    with open(bundle_path, "rb") as fh:
        bundle = pickle.load(fh)
    if model_name not in bundle["models"]:
        raise ValueError(f"unknown model {model_name!r}; "
                         f"have {sorted(bundle['models'])}")
    model = bundle["models"][model_name]

    pbp = ingest.load_pbp(SEASONS, refresh)
    schedules = ingest.load_schedules(SEASONS, refresh)
    depth_charts = ingest.load_depth_charts(SEASONS, refresh)
    injuries = ingest.load_injuries(SEASONS, refresh)
    snaps = ingest.load_snap_counts(SEASONS, refresh)
    rosters = ingest.load_weekly_rosters(SEASONS, refresh)

    targets = week_targets(schedules, season, week)
    starters = project_starters(depth_charts, injuries, season, week,
                                sorted(targets["team"].unique()), rosters,
                                overrides)
    targets = targets.merge(starters, on="team", how="left")

    resolved = targets[targets["starter_id"].notna()].copy()
    unresolved = targets[targets["starter_id"].isna()].copy()

    if resolved.empty:
        raise RuntimeError("no starters could be projected for this week")

    feats = build_inference_features(resolved, pbp, schedules, snaps, rosters,
                                     depth_charts)
    cols = bundle["features"]
    missing = [c for c in cols if c not in feats.columns]
    for c in missing:
        feats[c] = np.nan
    if missing:
        log.warning("features unavailable at inference, imputed: %s", missing)

    feats["p_5plus"] = model.predict_proba(feats[cols])[:, 1]
    feats["model"] = model_name

    # Continuity diagnostics. These are not model inputs (see
    # features.DIAGNOSTIC_COLUMNS for the measurements that decided that); they
    # are here so a reader can see which projections rest on a situation that
    # may no longer exist. The graded shares replace an earlier binary flag
    # that cleared as soon as a back logged one game, which made week 2 output
    # read fresher than it was.
    # How often this back has actually done the thing being predicted. These
    # are counts of prior games only, so week 3 shows weeks 1-2.
    feats["hits_this_season"] = feats["rb_fd_hits_season"].fillna(0).astype(int)
    feats["games_this_season"] = feats["rb_fd_games_season"].fillna(0).astype(int)
    feats["hits_last5"] = feats["rb_fd_hits_t5"].fillna(0).astype(int)
    feats["games_last5"] = feats["rb_fd_games_t5"].fillna(0).astype(int)

    # The same counts from the defence's side: how often this opponent has
    # already given the event up. Shown because it is the obvious thing to want
    # to know, and kept out of the model because it does not predict (see
    # features.DIAGNOSTIC_COLUMNS).
    # Play-calling on the opener, both directions: how often this team runs on
    # its own opener, and how often offences run at this defence on theirs.
    # The team-side rate is a model input; the defence-side one is not (see
    # features.DIAGNOSTIC_COLUMNS).
    feats["fd_run_pct"] = feats["tm_fd_run_rate_season"]
    feats["opp_fd_run_pct_faced"] = feats["opp_fd_run_rate_faced_season"]

    feats["opp_gave_up_season"] = feats["opp_fd_conceded_season"].fillna(0).astype(int)
    feats["opp_faced_season"] = feats["opp_fd_faced_season"].fillna(0).astype(int)
    feats["opp_gave_up_last5"] = feats["opp_fd_conceded_t5"].fillna(0).astype(int)
    feats["opp_faced_last5"] = feats["opp_fd_faced_t5"].fillna(0).astype(int)
    feats["trailing_from_this_season"] = feats["rb_trailing_this_season_share"]
    feats["same_team_share"] = feats["rb_trailing_same_team_share"]
    feats["new_head_coach"] = feats["tm_new_head_coach"]
    feats["oline_continuity"] = feats["tm_oline_continuity"]

    # Most of a back's form coming from last season is worth saying out loud.
    mostly_last_season = (feats["trailing_from_this_season"].fillna(0) < 0.5)
    feats["note"] = _add_note(
        feats["note"], mostly_last_season.to_numpy(),
        "form mostly from last season")

    # Form earned somewhere else describes a role he may no longer have.
    changed = (feats["same_team_share"].fillna(1.0) < 1.0)
    feats["note"] = _add_note(
        feats["note"], changed.to_numpy(),
        "changed teams; prior form is from another roster")

    # A new head coach is the clearest reason a team's scripted-opener history
    # may not carry forward, since the opener is the most scheme-driven part.
    feats["note"] = _add_note(
        feats["note"], (feats["new_head_coach"].fillna(0) == 1).to_numpy(),
        "team has a new head coach")

    # A projected starter with no resolvable game this season is the extreme
    # staleness case: every form feature describes last year. It slips past the
    # rookie check (he has career carries) and past the depth-chart checks (he
    # is the listed RB1), so flag it explicitly once the season is underway.
    no_season_games = ((feats["games_this_season"] == 0) & (week > 1)).to_numpy()
    feats["note"] = _add_note(feats["note"], no_season_games,
                              "NO resolvable game this season; every form "
                              "feature is from last year")
    feats.loc[no_season_games, "needs_review"] = 1

    # Players with no prior NFL game at all have no player features whatsoever;
    # every one of them is median-imputed, so the prediction is really just the
    # team and matchup talking. Rookies must be called out explicitly.
    no_history = feats["rb_carries_per_game_t5"].isna().to_numpy()
    feats["note"] = _add_note(feats["note"], no_history,
                              "NO career carries on record (rookie); player "
                              "features are all imputed")
    feats.loc[no_history, "needs_review"] = 1

    # A game that has already kicked off is not a projection.
    played = set(pbp.loc[pbp["season"] == season, "game_id"].unique())
    feats["already_played"] = feats["game_id"].isin(played).astype(int)
    feats["note"] = _add_note(feats["note"],
                              feats["already_played"].to_numpy() == 1,
                              "GAME ALREADY PLAYED")

    out_cols = ["season", "week", "game_id", "team", "opponent", "is_home",
                "starter_name", "depth_rank", "p_5plus", "model",
                "hits_this_season", "games_this_season",
                "hits_last5", "games_last5",
                "fd_run_pct", "opp_fd_run_pct_faced",
                "opp_gave_up_season", "opp_faced_season",
                "opp_gave_up_last5", "opp_faced_last5",
                "trailing_from_this_season",
                "same_team_share", "new_head_coach", "oline_continuity",
                "already_played", "needs_review", "note"]
    out = feats[out_cols].sort_values(["already_played", "p_5plus"],
                                      ascending=[True, False])

    if not unresolved.empty:
        pad = unresolved.assign(p_5plus=np.nan, model=model_name,
                                hits_this_season=np.nan, hits_last5=np.nan,
                                games_last5=np.nan, fd_run_pct=np.nan,
                                opp_fd_run_pct_faced=np.nan,
                                opp_gave_up_season=np.nan,
                                opp_faced_season=np.nan,
                                opp_gave_up_last5=np.nan,
                                opp_faced_last5=np.nan,
                                games_this_season=np.nan, already_played=0,
                                trailing_from_this_season=np.nan,
                                same_team_share=np.nan, new_head_coach=np.nan,
                                oline_continuity=np.nan)
        out = pd.concat([out, pad[out_cols]], ignore_index=True)
    return out.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--week", type=int, required=True)
    ap.add_argument("--model", default="logistic",
                    choices=["logistic", "lgbm", "heuristic", "prior"])
    ap.add_argument("--csv", help="also write predictions to this path")
    ap.add_argument("--refresh", action="store_true",
                    help="re-pull the data caches before predicting")
    ap.add_argument("--starter", action="append", metavar="TEAM=NAME",
                    default=[],
                    help="force a team's starting RB, e.g. --starter GB=Jacobs. "
                         "Use when you know the roster feed is stale. "
                         "Repeatable.")
    args = ap.parse_args()

    overrides = {}
    for item in args.starter:
        if "=" not in item:
            ap.error(f"--starter expects TEAM=NAME, got {item!r}")
        team, name = item.split("=", 1)
        overrides[team.strip()] = name.strip()

    preds = predict_week(args.season, args.week, args.model, args.refresh,
                         overrides)

    shown = preds.copy()
    shown["p_5plus"] = shown["p_5plus"].round(4)
    with pd.option_context("display.width", 200, "display.max_rows", 100,
                           "display.max_colwidth", 44):
        print(shown.to_string(index=False))

    flagged = int(preds["needs_review"].fillna(1).sum())
    print(f"\n{len(preds)} team-games | {flagged} flagged for manual review")
    print("Flagged rows have an uncertain starter. Confirm against the "
          "beat-reporter consensus before using them.")

    done = int(preds["already_played"].sum())
    if done:
        print(f"{done} team-games have already kicked off and are listed last; "
              "those are not projections.")

    share = preds["trailing_from_this_season"]
    if share.notna().any():
        print(f"\nTrailing-window freshness: median "
              f"{share.median():.0%} of each back's trailing five games are "
              f"from {args.season}. "
              f"{int((share.fillna(0) < 0.5).sum())} of {len(preds)} rest "
              "mostly on last season.")
    moved = int((preds["same_team_share"].fillna(1.0) < 1.0).sum())
    coaches = int((preds["new_head_coach"].fillna(0) == 1).sum())
    print(f"Continuity: {moved} back(s) carry form from another roster; "
          f"{coaches} team(s) have a new head coach.")
    print("These are context, not model inputs. Measured on 2020-2024, the "
          "model is not less accurate in these situations, so treat them as "
          "reasons to sanity-check a row rather than to discount it.")

    path = args.csv or (REPORT_DIR /
                        f"predictions_{args.season}_wk{args.week}.csv")
    preds.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
