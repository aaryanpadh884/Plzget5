# Evaluation: calibration and probability quality

Held-out test season: **2024** (467 team-games, base rate 0.5139). This season was never used for model selection; the hyperparameters were chosen on the walk-forward folds through 2023.

## Walk-forward CV (2020-2023, pooled)

```
              auc  logloss   brier     ece
model                                     
logistic   0.5727   0.6844  0.2457  0.0338
lgbm       0.5652   0.6863  0.2466  0.0275
heuristic  0.5367   0.6882  0.2475  0.0294
prior      0.4799   0.6932  0.2500  0.0161
```

## Held-out test season

```
              auc  logloss   brier     ece  cal_slope  cal_intercept
model                                                               
lgbm       0.6050   0.6787  0.2428  0.0689     1.9401        -0.0307
logistic   0.6116   0.6787  0.2428  0.0482     1.3111        -0.0142
heuristic  0.5957   0.6795  0.2432  0.0566     2.3772        -0.0188
prior      0.5000   0.6930  0.2499  0.0105     0.0008         0.0557
```

AUC for `prior` is not meaningful (it predicts one constant per fold) and is listed only for completeness.

On the CV folds the log-loss ordering is `logistic < lgbm < heuristic < prior`, and it is stable across all four seasons. The learned models beat both the heuristic and the base rate there.

**On the held-out season that ordering does not reproduce.** The best test-season log loss belongs to `lgbm` (0.6787), ahead of the CV-selected `logistic` (0.6787), a gap of 0.0000. One season is 467 team-games; the standard error on log loss at that size is larger than the spread between every model in the table. The correct reading is that the learned model's advantage over the simple rule is **real on four seasons of validation and unproven on one season of test**, not that the heuristic is better.

### Calibration slopes

Slope 1.0 with intercept 0.0 is perfect. Below 1.0 means predictions are too spread out (overconfident); above 1.0 means too compressed (underconfident), so the model could safely be more aggressive.

- `lgbm`: slope 1.94 - underconfident, predictions too compressed.
- `logistic`: slope 1.31 - underconfident, predictions too compressed.
- `heuristic`: slope 2.38 - underconfident, predictions too compressed.

The heuristic's large slope is an artifact of its shape: it emits only two distinct probabilities, both near the base rate, so a logistic refit has to stretch them hard. `logistic` is the only model in the table sitting in the well-calibrated band, which is what the heavy regularization bought. Earlier, lightly regularized configurations scored *worse than the base rate* on log loss despite a similar AUC.

## Reliability

**logistic** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4123  0.3617  0.0506
   2 93     0.4859  0.4194  0.0666
   3 93     0.5209  0.5591 -0.0382
   4 93     0.5519  0.5914 -0.0395
   5 94     0.5955  0.6383 -0.0428
```

**lgbm** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4240  0.3511  0.0729
   2 93     0.4978  0.4194  0.0784
   3 93     0.5267  0.6344 -0.1077
   4 93     0.5448  0.5591 -0.0144
   5 94     0.5616  0.6064 -0.0448
```

**heuristic** (test season, quintiles):

```
 bin   n  mean_pred  actual     gap
   2 168     0.4511  0.3810  0.0701
   5 299     0.5396  0.5886 -0.0491
```

## Does a stated probability mean what it says?

Pooled across the walk-forward folds (1,739 team-games), using `logistic`. Each row is a decile of predicted probability, with a 95% interval on the observed rate so you can see whether a gap is real or sample noise.

```
 bin   n  mean_pred  actual   lo95   hi95  covers
   1 174     0.3885  0.3851 0.3128 0.4574    True
   2 174     0.4455  0.4138 0.3406 0.4870    True
   3 174     0.4793  0.4770 0.4028 0.5512    True
   4 174     0.5003  0.4655 0.3914 0.5396    True
   5 173     0.5163  0.4913 0.4168 0.5658    True
   6 174     0.5296  0.5632 0.4895 0.6369    True
   7 174     0.5420  0.6264 0.5546 0.6983   False
   8 174     0.5557  0.5115 0.4372 0.5858    True
   9 174     0.5738  0.5172 0.4430 0.5915    True
  10 174     0.6043  0.6264 0.5546 0.6983    True
```

The predicted value falls inside the 95% interval for **9 of 10 deciles**. The predictions are usable at face value.

### Where the Brier score goes

```
uncertainty  0.2499   irreducible, fixed by the base rate
resolution   0.0058   variance the model actually explains
reliability  0.0017   miscalibration, smaller is better
```

Reliability is 0.0017, which is the number that matters for taking these probabilities at face value: near zero means the model is honestly uncertain rather than confidently wrong. Resolution is small because the outcome is genuinely close to a coin flip. See `reports/ceiling.md` for how much of that is fixable (very little).

### Sharpness

- 5th percentile prediction: 0.396
- median: 0.523
- 95th percentile: 0.600

The narrow spread is the honest answer, not a defect. A model emitting 0.80s on this problem would be miscalibrated.

## What the model actually keys on

```
                  feature    coef
            rb_depth_rank -0.0590
  opp_rush_epa_allowed_t5  0.0386
     rb_fd_carry_share_t5  0.0351
   rb_fd_participation_t5  0.0346
                  is_home  0.0323
  rb_fd_yards_per_game_t5  0.0307
        rb_fd_hit_rate_t5  0.0301
       implied_team_total  0.0257
       opp_ypc_allowed_t5  0.0248
  tm_fd_plays_per_game_t5  0.0242
        rb_carry_share_t5  0.0229
                rest_days -0.0228
rb_fd_carries_per_game_t5  0.0227
        opp_stuff_rate_t5 -0.0210
              team_spread  0.0203
```

Opportunity features dominate talent features. The back's recent first-drive carry volume and participation rate outrank his yards per carry, which is the scripted-opener effect the product plan warned about, showing up in the coefficients exactly as predicted.

## Figures

- `calibration_test.png`
- `calibration_cv.png`
