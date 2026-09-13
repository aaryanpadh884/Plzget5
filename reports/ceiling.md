# How accurate can this model get?

Every attempt to squeeze more out of this problem lands in the same place, so the useful question is not 'can we do better' but 'how much is knowable at all'. This report measures that.

## The experiment

Give a model the one thing it could never know before kickoff: how many carries the back actually took on the opening drive. Compare.

| information available | AUC | log loss | Brier |
|---|---|---|---|
| pregame features only (the real model) | 0.5730 | 0.6842 | 0.2456 |
| **oracle: carry count alone** | **0.8920** | **0.5467** | **0.1802** |
| oracle carries + all pregame features | 0.8807 | 0.5444 | 0.1796 |

Two things fall out of this table.

**Carry count is very nearly the whole answer.** Knowing only how many times the back carried gives AUC 0.892. Every pregame feature we built gives 0.573.

**Pregame features add almost nothing on top of it.** Going from carry count alone to carry count plus all 41 pregame features moves log loss only 0.0023. Once you know the opportunity, the back's talent, the matchup, and the game script are nearly irrelevant.

## Why the outcome is so close to a coin flip

```
                         mean  size
carries on first drive             
0                       0.000   832
1                       0.249  1431
2                       0.670  1122
3                       0.894   714
4                       0.949   374
5                       0.973   188
6                       0.986   140
```

The threshold sits exactly where the carry distribution is densest. One carry is a 25% proposition, two carries is 67%. The average opening drive gives the starter 1.91 carries, so most team-games land on the steepest part of that curve, where a single extra handoff flips the answer.

## What pregame data does and does not know

```
                       question  base rate    AUC  log loss
          P(at least 1 carries)     0.8267 0.6556    0.4510
          P(at least 2 carries)     0.5286 0.5880    0.6782
          P(at least 3 carries)     0.2949 0.5947    0.5857
P(5+ rushing yards) [the label]     0.5045 0.5730    0.6842
```

This is the crux. Pregame data predicts **whether** the back touches the ball reasonably well, because that is a question about his role and teams telegraph roles. It predicts **how many times** much worse, because carry count depends on how long the drive lasts, and drive length is decided by the drive itself: a third-down conversion, a holding penalty, an interception. None of that is knowable on Saturday night.

## Where the Brier score actually goes

Brier = reliability - resolution + uncertainty.

```
uncertainty  0.2499   irreducible; fixed by the 0.5 base rate
resolution   0.0058   how far the model correctly moves off the base rate
reliability  0.0017   miscalibration; smaller is better
-> Brier     0.2458
```

Resolution of 0.0058 against uncertainty of 0.2499 means the model explains roughly 2.3% of the available variance. That is small, and it is small for a real reason rather than a fixable one.

Reliability is near zero, which is the part that matters for your use case: the probabilities the model emits are close to the frequencies they describe. It is not confidently wrong, it is honestly uncertain.

## Practical read

The model's spread of predictions is narrow on purpose. Most team-games come back between 0.40 and 0.60, and that narrowness is the correct answer, not a limitation to engineer away. A model that confidently said 0.75 here would be lying.

If you want materially sharper probabilities, the only lever that would move them is better opportunity forecasting: projected snap share and game-script-conditional carry projections, ideally a beat-reporter or projection feed. More RB efficiency features will not do it, and this table is why.
