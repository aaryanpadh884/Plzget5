# NFL RB First-Drive 5+ Rush Yards

Binary classifier: for each team's **first offensive drive** of a game, predict
whether that team's **starting RB** totals **5+ rushing yards** on the drive
(sum across all carries, negative yards count against the total, receiving
yards excluded).

Built from free data only: nflfastR play-by-play via `nfl_data_py`, nflverse
rosters and depth charts, and Pro-Football-Reference snap counts.

---

## Headline results

Walk-forward validation, training on every prior season and predicting the
next, 2020 through 2024. 2025 is held out entirely and was never used for model
selection. Seasons 2013-2026, 5,316 usable team-games, 41 pregame features.
`TEST_SEASON` is the last *completed* season, which is not the same as
`LAST_SEASON` once an in-progress season is being pulled.

| model | AUC | log loss | Brier |
|---|---|---|---|
| **logistic** | **0.5807** | **0.6831** | **0.2450** |
| lgbm | 0.5751 | 0.6846 | 0.2458 |
| heuristic (the rule to beat) | 0.5596 | 0.6864 | 0.2466 |
| base rate | 0.5000 | 0.6931 | 0.2500 |

**The probabilities are trustworthy at face value.** Pooled across the
walk-forward folds, the predicted value lands inside a 95% interval on the
observed rate in **9 of 10 deciles**. The Brier decomposition puts
miscalibration (reliability) at 0.0017 against an irreducible uncertainty of
0.2499: the model is honestly uncertain rather than confidently wrong.

**It is also not very sharp, and that is correct.** Most predictions fall
between 0.39 and 0.60. A model emitting 0.80s on this problem would be lying.

## How accurate can this get? (read `reports/ceiling.md`)

Rather than guess at the ceiling, it is measured. Give a model the one thing it
cannot know before kickoff, the number of carries the back actually took on the
opening drive, and compare:

| information available | AUC | log loss | Brier |
|---|---|---|---|
| pregame features only (the real model) | 0.5730 | 0.6842 | 0.2456 |
| **oracle: carry count alone** | **0.8920** | **0.5467** | **0.1802** |
| oracle carries + all 41 pregame features | 0.8807 | 0.5444 | 0.1796 |

**Carry count is essentially the whole answer**, and every pregame feature in
the project adds almost nothing on top of it (log loss 0.5467 → 0.5444). This
is an *opportunity forecasting* problem wearing the costume of an RB evaluation
problem.

Why that caps accuracy:

| question | base rate | AUC | log loss |
|---|---|---|---|
| P(at least 1 carry) | 0.827 | 0.6556 | 0.4510 |
| P(at least 2 carries) | 0.529 | 0.5880 | 0.6782 |
| P(5+ rushing yards) — the label | 0.505 | 0.5730 | 0.6842 |

Pregame data predicts **whether** the back touches the ball reasonably well,
because that is a question about his role and teams telegraph roles. It
predicts **how many times** much worse, because carry count depends on drive
length, and drive length is decided by the drive itself: a third-down stop, a
holding penalty, an interception. The starter averages 1.91 carries on the
opener, and the threshold sits right where that curve is steepest (1 carry =
25%, 2 carries = 67%), so most team-games are one handoff away from flipping.

**What would actually move the needle**: projected snap share and
game-script-conditional carry projections, ideally a beat-reporter or
projection feed. More RB efficiency features will not, and the table above is
why.

---

## Quick start

```bash
pip install -r requirements.txt

python run_pipeline.py                          # ingest -> ... -> evaluate
python src/predict.py --season 2024 --week 10   # weekly inference
pytest                                          # leakage + train/serve guards
```

First run pulls about 56 MB and takes a few minutes. Everything caches to
`data/raw/`, after which the full pipeline runs in roughly 20 seconds.

---

## Project structure

```
nfl-first-drive-rb/
  data/raw/           # cached nflverse pulls (gitignored)
  data/processed/     # labeled dataset + feature matrix
  src/config.py       # paths, season range, thresholds, CV settings
  src/ingest.py       # pull + cache nfl_data_py / nflverse
  src/label.py        # drive-level labels + starter resolution
  src/features.py     # pregame feature engineering, lagged
  src/train.py        # model training + walk-forward CV
  src/evaluate.py     # calibration + probability quality
  src/ceiling.py      # how much is knowable at all (oracle analysis)
  src/predict.py      # weekly inference
  src/eda.py          # base rate and distribution checks
  notebooks/eda.ipynb
  tests/              # leakage and train/serve consistency
  models/             # fitted models + metadata (gitignored)
  reports/            # generated markdown + figures
```

---

## How the label is built

1. **First drive, per team.** The team that kicks off opens the game on
   defense, so its first possession is not the game's first drive. We take each
   team's minimum `fixed_drive` within the game. (`fixed_drive` is used over
   `drive` because `drive` is null on about 1% of plays.)

2. **Starting RB.** For training labels, the RB with the most rush attempts in
   that game, cross-checked against PFR offensive snap share:

   | outcome | rule | count |
   |---|---|---|
   | `confirmed` | attempt leader is also the snap leader and cleared a 35% snap floor | 4,829 |
   | `usage_only` | no snap data, but out-carried the next RB by 3+ | 14 |
   | dropped | tied carries, usage/snap disagreement, or thin margin | 1,186 |

   **5,316 of 6,577 team-games (80.8%) survive.** Ambiguous backfields are
   dropped rather than force-labeled, as the plan requires. Seasons start at
   2013 because that is the first year of PFR snap counts, which the starter
   cross-check depends on.

3. **Yards.** Sum `rushing_yards` over that back's carries on that drive.
   Plays negated by penalty are already excluded: nflfastR marks them
   `play_type == 'no_play'` and never sets `rush_attempt`, which we verified
   across all nine seasons. Run plays carrying a *declined or offsetting*
   penalty keep their yards and are correctly included.

4. **Label** = 1 if the sum ≥ 5.

A starter who took no carries on the opener gained zero yards on it, which is a
real negative, not a missing value.

---

## Two findings that shaped the modeling

**The base rate is 0.5045 and stable across every season from 2013 to 2024.**
This is not an imbalanced classification problem. No resampling, no class
weights, and log loss is directly interpretable against a 0.5 baseline.

**The starter takes zero opening-drive carries in 17.3% of team-games.** The
label is a compound event: the back has to be given the ball *and* the carries
have to total 5+. Roughly one negative in six is decided by play-calling before
a rushing play happens. Conditional on at least one carry, the base rate jumps
to 0.6100. This is why the top features are all opportunity features, not
talent features.

**A quarter of outcomes land within two yards of the line** (25.4% between 3
and 7 yards). That caps how sharp any model can be and is the strongest
argument for optimizing calibration over accuracy.

---

## Features

41 features, all strictly pregame, in three groups.

- **Player** — trailing-5 and season-to-date YPC, first-drive carries per game,
  first-drive participation rate, first-drive hit rate, share of the team's
  carries and of its opening-drive carries (bell-cow vs committee), offensive
  snap share, published depth-chart rank, red-zone carry share, stuff rate,
  explosive rate, EPA per carry.
- **Team** — first-drive run rate and its trend against the season baseline
  (the OC scripting signal), neutral-script run rate, rushing EPA per play,
  YPC, first-drive plays per game, Vegas spread and total, implied team total,
  home/away, rest days, venue and weather.
- **Matchup** — opponent YPC allowed, stuff rate, and run-defense EPA allowed.

Rate features roll the numerator and denominator separately and divide at the
end, so a 3-carry game does not count as heavily as a 25-carry game.

### Leakage control

Every rolling feature is built by shifting within the group *before* the
rolling window, so a game can never contribute to its own features. Splits are
time-based only; there is no random splitting anywhere in the codebase.

`tests/test_leakage.py` proves this rather than asserting it. The headline test
rebuilds the entire feature matrix from a play-by-play feed truncated to
everything before a cutoff week, then demands the cutoff week's features come
out bit-identical. Any feature reaching forward in time would shift.

`received_opening_kickoff` was **removed** from the model inputs: it is decided
by the coin toss, so it is not knowable before kickoff even though it is not a
game outcome. Measured cost of removing it was nil (log loss 0.6848 → 0.6847).

---

## Modeling notes

Both learned models are regularized far harder than defaults suggest. That was
not caution, it is what the folds asked for:

- An unconstrained LightGBM scored **worse than predicting the base rate** on
  log loss (0.6950 vs 0.6933) despite an AUC above 0.55. Classic
  overconfidence: fine ranking, useless probabilities.
- Sweeping the logistic penalty from C=1.0 down to C=0.0001 improved validation
  log loss monotonically until roughly C=0.001.
- Isotonic calibration made things *worse* (0.6940 → 0.7289). There is not
  enough data to fit a flexible calibration map. Platt scaling helped the tree
  model but not enough to catch the regularized linear one.

With correlations in the 0.1 range, any model confident enough to be
interesting is confident enough to be wrong.

---

## Weekly inference

```bash
python src/predict.py --season 2024 --week 10 --model logistic
```

**Starters at inference time come from the published depth chart and the injury
report, never from in-game usage.** Training labels may look at who actually
carried the ball because the game is over; inference may not, because on Sunday
morning you do not know yet. Using in-game usage here is the single easiest way
to build a model that backtests beautifully and cannot be bet.

The promotion chain: take the highest available depth-chart RB, skipping anyone
listed Out or Doubtful. Questionable backs are *not* skipped, because
questionable backs start all the time, but they are flagged. Every deviation
from "the listed RB1 starts" is written into a `note` column and sets
`needs_review`, so the output is auditable rather than silently guessed.

Sample output (2024 week 10, 28 team-games, 3 flagged):

```
team opponent  starter_name       depth_rank  p_5plus  needs_review  note
MIN  JAX       Aaron Jones               1.0   0.5896             0
BAL  CIN       Derrick Henry             1.0   0.5705             0
WAS  PIT       Austin Ekeler             2.0   0.4343             1  RB1 ruled out; promoted from depth 2
KC   DEN       Carson Steele             1.0   0.4095             1  multiple RBs listed as RB1 (committee backfield)
```

`tests/test_train_serve_consistency.py` replays a past week through the
inference path and asserts the features come out identical to the training
path, which is the failure mode most likely to go unnoticed.

---

## Risks, stated plainly

**Starter ID is the biggest failure point, and it is not solved.** 20% of
team-games are dropped in training because the backfield is genuinely
ambiguous. At inference the depth chart is a weaker signal than in-game usage,
and depth charts lie: teams list a nominal RB1 who splits work evenly. Rows
flagged `needs_review` should be confirmed against beat-reporter consensus
before use. The cleanest available improvement is a paid or scraped projected
-starters feed.

**First drives are scripted, so team signal should dominate RB talent.** The
data confirms it: `rb_fd_participation_t5` and `rb_fd_carries_per_game_t5`
outrank every efficiency feature in the model coefficients, and raw YPC does
not crack the top ten. The feature set deliberately carries team-level
scripting tendencies (`tm_fd_run_rate_t5`, `tm_fd_run_rate_trend_t5`) so the
model is not forced to explain opener outcomes with RB talent alone.

**17 games a season is a small sample for detecting an OC's script tendency,**
and coordinator changes reset it. The trailing-5 window weights recent games
over the season baseline, and `tm_fd_run_rate_trend_t5` measures recent form
*against* that baseline, which is the closest this feature set gets to noticing
a scheme change. It does not model coordinator changes explicitly, which is the
most obvious next improvement.

**The model is honest, not sharp, and those are different things.** Resolution
is 0.0058 against an irreducible uncertainty of 0.2499, so it explains about
2.3% of the available variance. The probabilities are trustworthy at face
value, but they will cluster near 0.5 for most team-games because that is what
the outcome actually is. Do not mistake a well-calibrated 0.52 for a weak
signal that better modeling would strengthen; `reports/ceiling.md` shows the
remaining headroom is small and lives entirely in opportunity forecasting.

**Model form is not the lever, and several plausible ones were tried and
rejected.** A two-stage decomposition, P(5+) = P(gets a carry) × P(5+ | carry),
is mathematically exact here but did not beat the direct binary model (0.6850
vs 0.6847). Neither did modelling the full carry distribution and integrating
an empirical conditional (0.6851), nor logistic/LightGBM ensembles (0.6843),
nor isotonic calibration (0.7289, much worse). Three extra seasons of data
bought 0.0001 and six new role features bought 0.0006. They are in the model
because they are free, not because they rescued it.

---

## Two model fits, on purpose

`train.py` saves two bundles, because holding out the latest season is right
when measuring a model and wrong when running it.

- `models.pkl` trains on seasons before `TEST_SEASON` and is what
  `evaluate.py` scores. It has never seen the test season.
- `models_production.pkl` trains on every row available, including the most
  recent completed season and any games already played in the current one.
  This is what `predict.py` loads.

## Depth-chart schema change (2025+)

nflverse replaced the depth-chart feed starting in 2025. The legacy format was
keyed by season and week with `position`, `depth_team` and `formation`. The
modern format is a stream of timestamped snapshots keyed only by `dt`, carrying
`pos_abb`, `pos_rank` and `pos_grp` instead, with no season or week column at
all. `ingest.py` normalizes both onto the legacy names: each snapshot is
assigned to the next week whose games had not finished when it was published,
and within a team-week only the latest snapshot is kept, which is the chart
that stood closest to kickoff.

Worth knowing because the failure was silent: the old ingest filtered on
`game_type == "REG"`, a column the modern feed does not have, so every 2025 and
2026 row was dropped without an error.

## Environment notes

`nfl_data_py`'s `import_schedules` and `import_weekly_rosters` fetch
`http://www.habitatring.com/games.csv` over plain HTTP, which many corporate
egress proxies block. `src/ingest.py` reads the nflverse GitHub mirrors
directly instead, so the pipeline works behind a restrictive proxy.

The product plan calls for scraping Pro-Football-Reference box scores to
cross-check starters. PFR is blocked by policy in the environment this was
built in, so the cross-check uses nflverse's snap-count release instead, which
is PFR box-score data delivered over an allowed host. Same source, same check,
no fragile scraping.
