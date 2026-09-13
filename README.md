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
next, 2020 through 2023. The 2024 season is held out entirely and was never
used for model selection.

| model | AUC | log loss | Brier |
|---|---|---|---|
| **logistic** | **0.5725** | **0.6847** | **0.2458** |
| lgbm | 0.5702 | 0.6856 | 0.2463 |
| heuristic (the rule to beat) | 0.5506 | 0.6884 | 0.2476 |
| base rate | 0.5000 | 0.6933 | 0.2501 |

The ordering is stable across all four validation folds. **It does not
reproduce on the single held-out season**, where the heuristic edges the
learned models on log loss (0.6795 vs 0.6823). One season is 467 team-games and
cannot separate models this close. See `reports/evaluation.md`.

The honest summary: **the model is a modestly better-calibrated probability
estimate than the base rate or the simple rule. It is not a demonstrated
betting edge.** The backtest in the report benchmarks against a hypothetical
50/50 market, not a real book's price, and says so.

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
  src/evaluate.py     # calibration, backtest vs baseline
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
   | `confirmed` | attempt leader is also the snap leader and cleared a 35% snap floor | 3,788 |
   | `usage_only` | no snap data, but out-carried the next RB by 3+ | 14 |
   | dropped | tied carries, usage/snap disagreement, or thin margin | 932 |

   **3,802 of 4,734 team-games (80.3%) survive.** Ambiguous backfields are
   dropped rather than force-labeled, as the plan requires.

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

**The base rate is 0.5047 and stable across every season from 2016 to 2024.**
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

35 features, all strictly pregame, in three groups.

- **Player** — trailing-5 and season-to-date YPC, first-drive carries per game,
  first-drive participation rate, first-drive hit rate, red-zone carry share,
  stuff rate, explosive rate, EPA per carry.
- **Team** — first-drive run rate and its trend against the season baseline
  (the OC scripting signal), rushing EPA per play, YPC, first-drive plays per
  game, Vegas spread and total, implied team total, home/away, rest days,
  venue and weather.
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

**The backtest is not evidence of a betting edge.** It prices every team-game
as a coin flip. A real sportsbook prices near the true probability, so the
quantity that matters is the model's edge over the book's number, which needs
historical prop odds this project does not have. The thresholds in that table
were also chosen after seeing results, with no multiple-comparison correction.

---

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
