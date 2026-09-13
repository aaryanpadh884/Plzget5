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


def project_starters(depth_charts: pd.DataFrame, injuries: pd.DataFrame,
                     season: int, week: int, teams: list[str]) -> pd.DataFrame:
    """Project each team's starting RB from the depth chart and injury report.

    Promotion chain: take the highest available depth-chart RB. If the listed
    RB1 is Out or Doubtful, the RB2 is promoted, and so on. Every deviation
    from "RB1 starts" is recorded in ``note`` so the output is auditable.
    """
    rbs = depth_chart_rbs(depth_charts, season, week)
    out_ids = blocked_players(injuries, season, week)
    quest_ids = questionable_players(injuries, season, week)

    rows = []
    for team in teams:
        pool = rbs[rbs["team"] == team]
        if pool.empty:
            rows.append({"team": team, "starter_id": None, "starter_name": None,
                         "depth_rank": np.nan, "needs_review": 1,
                         "note": "no depth chart published for this team-week"})
            continue

        available = pool[~pool["gsis_id"].isin(out_ids)]
        if available.empty:
            listed = pool.iloc[0]
            rows.append({"team": team, "starter_id": None,
                         "starter_name": listed["full_name"],
                         "depth_rank": float(listed["depth"]), "needs_review": 1,
                         "note": "every listed RB is Out or Doubtful"})
            continue

        pick = available.iloc[0]
        top_depth = float(pool["depth"].min())
        note, review = "", 0

        if float(pick["depth"]) > top_depth:
            note = (f"RB1 ruled out; promoted from depth {int(pick['depth'])}")
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
                 refresh: bool = False) -> pd.DataFrame:
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
                                sorted(targets["team"].unique()))
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

    # Early in a season the trailing windows are filled entirely by the prior
    # season's games. Backs change teams and roles over an offseason, so those
    # features describe a situation that may no longer exist. Flag it rather
    # than present week 1 numbers with the same confidence as week 10.
    feats["games_this_season"] = feats["rb_games_played_season"].fillna(0)
    feats["note"] = _add_note(feats["note"],
                              (feats["games_this_season"] == 0).to_numpy(),
                              "form is from last season only")

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
                "games_this_season", "already_played", "needs_review", "note"]
    out = feats[out_cols].sort_values(["already_played", "p_5plus"],
                                      ascending=[True, False])

    if not unresolved.empty:
        pad = unresolved.assign(p_5plus=np.nan, model=model_name,
                                games_this_season=np.nan, already_played=0)
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
    args = ap.parse_args()

    preds = predict_week(args.season, args.week, args.model, args.refresh)

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

    stale = int((preds["games_this_season"].fillna(0) == 0).sum())
    if stale:
        print(f"\n{stale} of {len(preds)} backs have no games logged in "
              f"{args.season}, so their form features come entirely from the "
              "prior season. Offseason team and role changes are not reflected "
              "in those numbers. Early-season predictions are softer than the "
              "calibration report implies; treat them accordingly.")

    path = args.csv or (REPORT_DIR /
                        f"predictions_{args.season}_wk{args.week}.csv")
    preds.to_csv(path, index=False)
    print(f"wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
