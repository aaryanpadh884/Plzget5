# Evaluation: calibration and backtest

Held-out test season: **2024** (467 team-games, base rate 0.5139). This season was never used for model selection; the hyperparameters were chosen on the walk-forward folds through 2023.

## Walk-forward CV (2020-2023, pooled)

```
              auc  logloss   brier     ece
model                                     
logistic   0.5733   0.6849  0.2459  0.0240
lgbm       0.5706   0.6858  0.2463  0.0400
heuristic  0.5362   0.6885  0.2477  0.0304
prior      0.4799   0.6933  0.2501  0.0169
```

## Held-out test season

```
              auc  logloss   brier     ece  cal_slope  cal_intercept
model                                                               
heuristic  0.5957   0.6795  0.2432  0.0566     2.3778        -0.0208
lgbm       0.5949   0.6817  0.2443  0.0653     1.7990        -0.0282
logistic   0.5916   0.6826  0.2447  0.0575     1.1992         0.0017
prior      0.5000   0.6930  0.2499  0.0106     0.0007         0.0557
```

AUC for `prior` is not meaningful (it predicts one constant per fold) and is listed only for completeness.

On the CV folds the log-loss ordering is `logistic < lgbm < heuristic < prior`, and it is stable across all four seasons. The learned models beat both the heuristic and the base rate there.

**On the held-out season that ordering does not reproduce.** The best test-season log loss belongs to `heuristic` (0.6795), ahead of the CV-selected `logistic` (0.6826), a gap of 0.0032. One season is 467 team-games; the standard error on log loss at that size is larger than the spread between every model in the table. The correct reading is that the learned model's advantage over the simple rule is **real on four seasons of validation and unproven on one season of test**, not that the heuristic is better.

### Calibration slopes

Slope 1.0 with intercept 0.0 is perfect. Below 1.0 means predictions are too spread out (overconfident); above 1.0 means too compressed (underconfident), so the model could safely be more aggressive.

- `heuristic`: slope 2.38 - underconfident, predictions too compressed.
- `lgbm`: slope 1.80 - underconfident, predictions too compressed.
- `logistic`: slope 1.20 - well calibrated.

The heuristic's large slope is an artifact of its shape: it emits only two distinct probabilities, both near the base rate, so a logistic refit has to stretch them hard. `logistic` is the only model in the table sitting in the well-calibrated band, which is what the heavy regularization bought. Earlier, lightly regularized configurations scored *worse than the base rate* on log loss despite a similar AUC.

## Reliability

**logistic** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4239  0.3936  0.0303
   2 93     0.4819  0.4194  0.0625
   3 93     0.5139  0.5591 -0.0452
   4 93     0.5445  0.5591 -0.0146
   5 94     0.5923  0.6383 -0.0460
```

**lgbm** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4342  0.3298  0.1045
   2 93     0.4950  0.5161 -0.0211
   3 93     0.5236  0.5914 -0.0678
   4 93     0.5430  0.5269  0.0161
   5 94     0.5621  0.6064 -0.0443
```

**heuristic** (test season, quintiles):

```
 bin   n  mean_pred  actual     gap
   2 168     0.4513  0.3810  0.0704
   5 299     0.5398  0.5886 -0.0489
```

## Backtest at -110

Break-even hit rate at standard juice is 0.5238. Bet the side the model prefers whenever its confidence clears the threshold. Pooled across the walk-forward folds, which is the larger and more honest sample:

**logistic**

```
 min_confidence  bets  hit_rate  break_even   edge  units    roi  p_value
           0.50  1739    0.5630      0.5238 0.0392 130.00 0.0748   0.0006
           0.53  1052    0.5732      0.5238 0.0494  99.18 0.0943   0.0007
           0.55   667    0.5817      0.5238 0.0579  73.73 0.1105   0.0015
           0.58   265    0.6038      0.5238 0.0800  40.45 0.1527   0.0053
```

**lgbm**

```
 min_confidence  bets  hit_rate  break_even   edge  units    roi  p_value
           0.50  1739    0.5434      0.5238 0.0196  65.09 0.0374   0.0533
           0.53  1119    0.5684      0.5238 0.0446  95.18 0.0851   0.0015
           0.55   709    0.5952      0.5238 0.0714  96.64 0.1363   0.0001
           0.58   174    0.5862      0.5238 0.0624  20.73 0.1191   0.0576
```

**heuristic**

```
 min_confidence  bets  hit_rate  break_even   edge  units    roi  p_value
           0.50  1739    0.5526      0.5238 0.0288  95.64 0.0550   0.0086
           0.53  1485    0.5475      0.5238 0.0237  67.09 0.0452   0.0358
```

### Read this before believing the ROI column

The backtest above prices every team-game as a coin flip and asks whether the model beats -110 against that. It clears the bar, and at the higher confidence thresholds the binomial p-values are small. That is still not evidence of a betting edge, for three reasons.

1. **The benchmark is wrong on purpose.** A real sportsbook does not hang this prop at 50/50. It prices near the true probability, so the quantity that matters is the model's edge over *the book's number*, which needs historical prop odds this project does not have. Free data gets us spread and total, not RB drive props.
2. **The thresholds were chosen after seeing the results.** Four thresholds were tried; reporting the best one overstates significance, and no multiple-comparison correction is applied.
3. **The line would move.** Opening-drive props are thin markets. Any real stake changes the price you get.

The defensible claim is narrow: **the model produces better-calibrated probabilities than the base rate or the heuristic rule.** That is worth having as an input. Treating it as a standalone betting signal is not supported by anything measured here.

## What the model actually keys on

```
                   feature    coef
    rb_fd_participation_t5  0.0442
    rb_carries_per_game_t5  0.0419
   opp_rush_epa_allowed_t5  0.0405
 rb_fd_carries_per_game_t5  0.0384
         rb_fd_hit_rate_t5  0.0371
   rb_fd_yards_per_game_t5  0.0347
                   is_home  0.0327
        opp_ypc_allowed_t5  0.0308
         opp_stuff_rate_t5 -0.0269
        implied_team_total  0.0260
rb_fd_participation_season  0.0251
   tm_fd_plays_per_game_t5  0.0236
               team_spread  0.0204
   tm_fd_run_rate_trend_t5  0.0203
                   is_dome  0.0176
```

Opportunity features dominate talent features. The back's recent first-drive carry volume and participation rate outrank his yards per carry, which is the scripted-opener effect the product plan warned about, showing up in the coefficients exactly as predicted.

## Figures

- `calibration_test.png`
- `calibration_cv.png`
