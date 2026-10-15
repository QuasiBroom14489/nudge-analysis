# What makes a nudge useful? (baseline)

1,725 nudges raised; 343 judged by hand: 120 useful, 35 wrong, 188 noise (**35% useful**; 380 verdict events, the latest per nudge counts). Median time to a verdict: 251.9 min.

Caveat: judged nudges are the ones I chose to answer, not a random sample.

## By category

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| email | 256 | 90.0 | 0.352 | 14.0 | 152.0 |
| due | 31 | 13.0 | 0.419 | 6.0 | 12.0 |
| reminder | 27 | 6.0 | 0.222 | 12.0 | 9.0 |
| overdue | 18 | 8.0 | 0.444 | 1.0 | 9.0 |
| unmatched | 6 | 0.0 | 0.0 | 2.0 | 4.0 |

## By urgency

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| medium | 319 | 112.0 | 0.351 | 26.0 | 181.0 |
| high | 20 | 5.0 | 0.25 | 9.0 | 6.0 |

## By conf_band

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| 0.80-0.89 | 99 | 23.0 | 0.232 | 12.0 | 64.0 |
| 0.90-0.99 | 64 | 23.0 | 0.359 | 15.0 | 26.0 |
| 1.00 | 156 | 60.0 | 0.385 | 6.0 | 90.0 |
| <0.80 | 24 | 14.0 | 0.583 | 2.0 | 8.0 |

## By time_of_day

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| before 9 | 40 | 20.0 | 0.5 | 4.0 | 16.0 |
| 9-12 | 46 | 10.0 | 0.217 | 4.0 | 32.0 |
| 12-17 | 127 | 41.0 | 0.323 | 7.0 | 79.0 |
| 17-21 | 74 | 30.0 | 0.405 | 8.0 | 36.0 |
| 21+ | 56 | 19.0 | 0.339 | 12.0 | 25.0 |

## By repeat_band

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| first | 34 | 13.0 | 0.382 | 5.0 | 16.0 |
| 2nd-3rd | 76 | 44.0 | 0.579 | 9.0 | 23.0 |
| 4th-10th | 96 | 42.0 | 0.438 | 14.0 | 40.0 |
| 11th+ | 137 | 21.0 | 0.153 | 7.0 | 109.0 |

## By week

| level | judged | useful | useful_rate | wrong | noise |
|---|---|---|---|---|---|
| 1 | 11 | 7.0 | 0.636 | 1.0 | 3.0 |
| 2 | 176 | 90.0 | 0.511 | 31.0 | 55.0 |
| 4 | 140 | 19.0 | 0.136 | 3.0 | 118.0 |
| 5 | 6 | 2.0 | 0.333 | 0.0 | 4.0 |
| 6 | 6 | 2.0 | 0.333 | 0.0 | 4.0 |

## What if one rule could only fire N times? (in-sample replay)

| cap | raised kept | judged kept | useful kept | share of useful kept | precision |
|---|---|---|---|---|---|
| 3 | 417 | 110 | 57 | 48% | 52% |
| 5 | 580 | 152 | 75 | 62% | 49% |
| 10 | 799 | 206 | 99 | 82% | 48% |
| 20 | 1085 | 255 | 110 | 92% | 43% |
| none | 1725 | 343 | 120 | 100% | 35% |

## Held-out check (time-ordered 70/30 split)

- Trained on the first 240 judged nudges, scored the next 103 (test base rate 11%).
- AUC: model 0.694, the intuition model's own confidence alone 0.568.
- Brier: model 0.1529, vs. 0.2161 for always predicting the training base rate and 0.0954 for a constant that already knows the test window's rate. The model ranks better than chance but does not out-predict that oracle constant: usefulness drifted down over time (see By week).
- If only the model's top third had been delivered, 18% would have been useful.

## Largest coefficients

| term | coef |
|---|---|
| cat__urgency_high | -0.59 |
| cat__category_reminder | -0.48 |
| num__prior_fires | -0.41 |
| cat__time_of_day_9-12 | -0.37 |
| cat__category_infrequent_sklearn | -0.21 |
| cat__time_of_day_before 9 | +0.17 |
| num__evidence_n | +0.20 |
| cat__time_of_day_17-21 | +0.29 |
| cat__urgency_infrequent_sklearn | +0.66 |
| cat__category_email | +0.74 |
