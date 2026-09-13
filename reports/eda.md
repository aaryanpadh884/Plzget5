# EDA: first-drive RB 5+ rushing yards

## Sample size

- Team-games with a resolvable RB: **4,734**
- Usable after dropping ambiguous starters: **3,802** (80.3%)
- Seasons: 2016-2024

Starter resolution outcomes:

```
confirmed     3788
ambiguous      932
usage_only      14
```

Why team-games were dropped:

```
usage_snap_disagreement    793
tied_carries               129
snap_share_below_floor       8
thin_margin_no_snaps         2
```

## Base rate

- P(first-drive rush yards >= 5) = **0.5047**
- Positives: 1,919 / 3,802

The label is close to a coin flip, so this is *not* an imbalanced classification problem and needs no resampling or class weighting. That is convenient: log loss and Brier score are directly interpretable against a 0.5 baseline.

Base rate by season:

```
          mean  size
season              
2016    0.4988   409
2017    0.5179   390
2018    0.4987   387
2019    0.4765   405
2020    0.5379   409
2021    0.5034   439
2022    0.5114   438
2023    0.4847   458
2024    0.5139   467
```

## The starter who never touches the ball

- Team-games where the resolved starter had **zero** carries on the opening drive: **17.3%**
- Base rate conditional on at least one carry: **0.6100**

This is the single most important structural fact in the dataset. The label is a compound event: the starter has to be given the ball on the opener *and* the carries have to total 5+ yards. Roughly one in six negatives is decided before a single rushing play happens, by play-calling rather than by the back. Any model that ignores opener participation is trying to predict yardage on drives where the back was never involved.

Distribution of first-drive carries by the starter:

```
0     656
1    1151
2     882
3     583
4     278
5     155
6      55
7      26
8      11
9       3
```

## Distribution of first-drive rushing yards

```
count    3802.000
mean        8.092
std        10.677
min       -14.000
25%         0.000
50%         5.000
75%        12.000
max        87.000
```

- Median is 5 yards, sitting right on the 5-yard line, which is why the base rate lands near 0.5.
- 25.4% of team-games land between 3 and 7 yards, so a large share of outcomes are decided by a single yard or two. That caps how sharp any model can be here and is the strongest argument for optimizing calibration over accuracy.

## Strongest linear correlates of the label

```
rb_fd_carries_per_game_t5       0.1245
rb_fd_hit_rate_t5               0.1229
rb_fd_yards_per_game_t5         0.1173
rb_fd_participation_t5          0.1156
rb_carries_per_game_t5          0.1066
rb_fd_participation_season      0.0985
opp_rush_epa_allowed_t5         0.0628
implied_team_total              0.0599
opp_ypc_allowed_t5              0.0566
tm_fd_plays_per_game_t5         0.0562
tm_fd_rush_yards_per_game_t5    0.0474
total_line                      0.0431
team_spread                     0.0423
opp_stuff_rate_t5              -0.0405
rb_epa_per_carry_t5             0.0355
```

Every correlation is small. Nothing in the pregame feature set comes close to determining the outcome, which is what you would expect for a single-drive, few-carry event.

## Figures

- `eda_yards_hist.png`
- `eda_base_rate_by_season.png`
- `eda_label_by_carries.png`
