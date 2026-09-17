# EDA: first-drive RB 5+ rushing yards

## Sample size

- Team-games with a resolvable RB: **6,605**
- Usable after dropping ambiguous starters: **5,338** (80.8%)
- Seasons: 2013-2026

Starter resolution outcomes:

```
confirmed     5310
ambiguous     1267
usage_only      28
```

Why team-games were dropped:

```
usage_snap_disagreement    1074
tied_carries                174
snap_share_below_floor       15
thin_margin_no_snaps          4
```

## Base rate

- P(first-drive rush yards >= 5) = **0.5082**
- Positives: 2,713 / 5,338

The label is close to a coin flip, so this is *not* an imbalanced classification problem and needs no resampling or class weighting. That is convenient: log loss and Brier score are directly interpretable against a 0.5 baseline.

Base rate by season:

```
          mean  size
season              
2013    0.5000   364
2014    0.4842   349
2015    0.5274   328
2016    0.4988   409
2017    0.5179   390
2018    0.4987   387
2019    0.4765   405
2020    0.5379   409
2021    0.5034   439
2022    0.5114   438
2023    0.4847   458
2024    0.5139   467
2025    0.5362   470
2026    0.7200    25
```

## The starter who never touches the ball

- Team-games where the resolved starter had **zero** carries on the opening drive: **17.3%**
- Base rate conditional on at least one carry: **0.6142**

This is the single most important structural fact in the dataset. The label is a compound event: the starter has to be given the ball on the opener *and* the carries have to total 5+ yards. Roughly one in six negatives is decided before a single rushing play happens, by play-calling rather than by the back. Any model that ignores opener participation is trying to predict yardage on drives where the back was never involved.

Distribution of first-drive carries by the starter:

```
0     921
1    1564
2    1256
3     804
4     420
5     216
6      94
7      43
8      12
9       6
```

## Distribution of first-drive rushing yards

```
count    5338.000
mean        8.236
std        10.799
min       -14.000
25%         0.000
50%         5.000
75%        12.000
max        87.000
```

- Median is 5 yards, sitting right on the 5-yard line, which is why the base rate lands near 0.5.
- 25.0% of team-games land between 3 and 7 yards, so a large share of outcomes are decided by a single yard or two. That caps how sharp any model can be here and is the strongest argument for optimizing calibration over accuracy.

## Strongest linear correlates of the label

```
rb_depth_rank                -0.1234
rb_fd_carry_share_t5          0.1204
rb_fd_participation_t5        0.1170
rb_fd_carries_per_game_t5     0.1161
rb_fd_hit_rate_t5             0.1157
rb_fd_yards_per_game_t5       0.1086
rb_fd_participation_season    0.1041
rb_carry_share_t5             0.1026
rb_carries_per_game_t5        0.0994
rb_snap_share_t5              0.0939
implied_team_total            0.0574
opp_rush_epa_allowed_t5       0.0505
team_spread                   0.0446
tm_fd_plays_per_game_t5       0.0445
opp_ypc_allowed_t5            0.0401
```

Every correlation is small. Nothing in the pregame feature set comes close to determining the outcome, which is what you would expect for a single-drive, few-carry event.

## Figures

- `eda_yards_hist.png`
- `eda_base_rate_by_season.png`
- `eda_label_by_carries.png`
