# How accurate can this model get?

Every attempt to squeeze more out of this problem lands in the same place, so the useful question is not 'can we do better' but 'how much is knowable at all'. This report measures that.

## The experiment

Give a model the one thing it could never know before kickoff: how many carries the back actually took on the opening drive. Compare.

| information available | AUC | log loss | Brier |
|---|---|---|---|
| pregame features only (the real model) | 0.5807 | 0.6831 | 0.2450 |
| **oracle: carry count alone** | **0.8926** | **0.5410** | **0.1778** |
| oracle carries + all pregame features | 0.8845 | 0.5381 | 0.1769 |

Two things fall out of this table.

**Carry count is very nearly the whole answer.** Knowing only how many times the back carried gives AUC 0.893. Every pregame feature we built gives 0.581.

**Pregame features add almost nothing on top of it.** Going from carry count alone to carry count plus all 41 pregame features moves log loss only 0.0028. Once you know the opportunity, the back's talent, the matchup, and the game script are nearly irrelevant.

## Why the outcome is so close to a coin flip

```
                         mean  size
carries on first drive             
0                       0.000   912
1                       0.248  1546
2                       0.671  1234
3                       0.889   795
4                       0.952   416
5                       0.972   213
6                       0.987   155
```

The threshold sits exactly where the carry distribution is densest. One carry is a 25% proposition, two carries is 67%. The average opening drive gives the starter 1.93 carries, so most team-games land on the steepest part of that curve, where a single extra handoff flips the answer.

## What pregame data does and does not know

```
                       question  base rate    AUC  log loss
          P(at least 1 carries)     0.8270 0.6428    0.4503
          P(at least 2 carries)     0.5337 0.5846    0.6800
          P(at least 3 carries)     0.2996 0.5902    0.5916
P(5+ rushing yards) [the label]     0.5073 0.5807    0.6831
```

This is the crux. Pregame data predicts **whether** the back touches the ball reasonably well, because that is a question about his role and teams telegraph roles. It predicts **how many times** much worse, because carry count depends on how long the drive lasts, and drive length is decided by the drive itself: a third-down conversion, a holding penalty, an interception. None of that is knowable on Saturday night.

## Where the Brier score actually goes

Brier = reliability - resolution + uncertainty.

```
uncertainty  0.2499   irreducible; fixed by the 0.5 base rate
resolution   0.0067   how far the model correctly moves off the base rate
reliability  0.0018   miscalibration; smaller is better
-> Brier     0.2451
```

Resolution of 0.0067 against uncertainty of 0.2499 means the model explains roughly 2.7% of the available variance. That is small, and it is small for a real reason rather than a fixable one.

Reliability is near zero, which is the part that matters for your use case: the probabilities the model emits are close to the frequencies they describe. It is not confidently wrong, it is honestly uncertain.

## Practical read

The model's spread of predictions is narrow on purpose. Most team-games come back between 0.39 and 0.60, and that narrowness is the correct answer, not a limitation to engineer away. A model that confidently said 0.75 here would be lying.

If you want materially sharper probabilities, the only lever that would move them is better opportunity forecasting: projected snap share and game-script-conditional carry projections, ideally a beat-reporter or projection feed. More RB efficiency features will not do it, and this table is why.
