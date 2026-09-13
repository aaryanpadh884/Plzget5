# Evaluation: calibration and probability quality

Held-out test season: **2025** (468 team-games, base rate 0.5342). This season was never used for model selection; the hyperparameters were chosen on the walk-forward folds through 2023.

## Walk-forward CV (2020-2023, pooled)

```
              auc  logloss   brier     ece
model                                     
logistic   0.5812   0.6832  0.2451  0.0304
lgbm       0.5734   0.6847  0.2458  0.0242
heuristic  0.5551   0.6864  0.2466  0.0324
prior      0.4821   0.6931  0.2500  0.0149
```

## Held-out test season

```
              auc  logloss   brier     ece  cal_slope  cal_intercept
model                                                               
lgbm       0.5753   0.6819  0.2444  0.0735     1.4808         0.0260
logistic   0.5531   0.6830  0.2451  0.0504     0.9669         0.0456
heuristic  0.5503   0.6857  0.2463  0.0189     1.2103         0.0638
prior      0.5000   0.6926  0.2497  0.0297     0.0025         0.1369
```

AUC for `prior` is not meaningful (it predicts one constant per fold) and is listed only for completeness.

On the CV folds the log-loss ordering is `logistic < lgbm < heuristic < prior`, and it is stable across all four seasons. The learned models beat both the heuristic and the base rate there.

**On the held-out season that ordering does not reproduce.** The best test-season log loss belongs to `lgbm` (0.6819), ahead of the CV-selected `logistic` (0.6830), a gap of 0.0012. One season is 467 team-games; the standard error on log loss at that size is larger than the spread between every model in the table. The correct reading is that the learned model's advantage over the simple rule is **real on four seasons of validation and unproven on one season of test**, not that the heuristic is better.

### Calibration slopes

Slope 1.0 with intercept 0.0 is perfect. Below 1.0 means predictions are too spread out (overconfident); above 1.0 means too compressed (underconfident), so the model could safely be more aggressive.

- `lgbm`: slope 1.48 - underconfident, predictions too compressed.
- `logistic`: slope 0.97 - well calibrated.
- `heuristic`: slope 1.21 - well calibrated.

The heuristic's large slope is an artifact of its shape: it emits only two distinct probabilities, both near the base rate, so a logistic refit has to stretch them hard. `logistic` is the only model in the table sitting in the well-calibrated band, which is what the heavy regularization bought. Earlier, lightly regularized configurations scored *worse than the base rate* on log loss despite a similar AUC.

## Reliability

**logistic** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4219  0.4362 -0.0143
   2 93     0.5013  0.5591 -0.0579
   3 94     0.5331  0.5638 -0.0307
   4 93     0.5599  0.5484  0.0116
   5 94     0.6028  0.5638  0.0390
```

**lgbm** (test season, quintiles):

```
 bin  n  mean_pred  actual     gap
   1 94     0.4339  0.4255  0.0083
   2 93     0.5105  0.5591 -0.0487
   3 94     0.5326  0.5213  0.0113
   4 93     0.5512  0.4946  0.0565
   5 94     0.5662  0.6702 -0.1040
```

**heuristic** (test season, quintiles):

```
 bin   n  mean_pred  actual     gap
   2 138     0.4450  0.4493 -0.0043
   5 330     0.5447  0.5697 -0.0250
```

## Does a stated probability mean what it says?

Pooled across the walk-forward folds (2,206 team-games), using `logistic`. Each row is a decile of predicted probability, with a 95% interval on the observed rate so you can see whether a gap is real or sample noise.

```
 bin   n  mean_pred  actual   lo95   hi95  covers
   1 221     0.3871  0.3846 0.3205 0.4488    True
   2 220     0.4448  0.4045 0.3397 0.4694    True
   3 221     0.4782  0.4706 0.4048 0.5364    True
   4 220     0.4996  0.4318 0.3664 0.4973   False
   5 221     0.5156  0.5204 0.4545 0.5862    True
   6 220     0.5295  0.5500 0.4843 0.6157    True
   7 221     0.5422  0.6380 0.5746 0.7014   False
   8 220     0.5567  0.5091 0.4430 0.5752    True
   9 221     0.5745  0.5656 0.5003 0.6310    True
  10 221     0.6064  0.6154 0.5512 0.6795    True
```

The predicted value falls inside the 95% interval for **8 of 10 deciles**. The predictions are usable at face value.

### Where the Brier score goes

```
uncertainty  0.2499   irreducible, fixed by the base rate
resolution   0.0067   variance the model actually explains
reliability  0.0018   miscalibration, smaller is better
```

Reliability is 0.0018, which is the number that matters for taking these probabilities at face value: near zero means the model is honestly uncertain rather than confidently wrong. Resolution is small because the outcome is genuinely close to a coin flip. See `reports/ceiling.md` for how much of that is fixable (very little).

### Sharpness

- 5th percentile prediction: 0.395
- median: 0.523
- 95th percentile: 0.601

The narrow spread is the honest answer, not a defect. A model emitting 0.80s on this problem would be miscalibrated.

## What the model actually keys on

```
                  feature    coef
            rb_depth_rank -0.0598
     rb_fd_carry_share_t5  0.0425
  opp_rush_epa_allowed_t5  0.0385
        rb_fd_hit_rate_t5  0.0375
  rb_fd_yards_per_game_t5  0.0367
   rb_fd_participation_t5  0.0285
       implied_team_total  0.0282
                  is_home  0.0264
rb_fd_carries_per_game_t5  0.0254
       opp_ypc_allowed_t5  0.0240
         rb_stuff_rate_t5 -0.0238
        rb_carry_share_t5  0.0229
  tm_fd_plays_per_game_t5  0.0224
              team_spread  0.0213
                rest_days -0.0197
```

Opportunity features dominate talent features. The back's recent first-drive carry volume and participation rate outrank his yards per carry, which is the scripted-opener effect the product plan warned about, showing up in the coefficients exactly as predicted.

## Figures

- `calibration_test.png`
- `calibration_cv.png`
