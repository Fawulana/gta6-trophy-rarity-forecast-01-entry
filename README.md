# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fawulana/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Fawulana/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, its earn
rate on Exophase at the case's target, as five quantiles per trophy
(5%, 25%, 50%, 75%, 95%), scored by pinball loss on the logit scale.

## How it works

`main.ts` (Node 22, standard library only, no network, deterministic) reads
every `<id>.in` case file from the input directory and writes
`<id>.out` to the output directory.

The key observation is that Exophase refreshes a game's counts on its own
schedule, not continuously: on launch night the page sat unchanged for hours
(638 players tracked, every rate flat), then one refresh jumped to 3,868
players and every rate but the 100% one fell. The refreshed totals
(4,276 trophies earned over 3,868 players) pin the drop: the 54 non-100%
trophies sum to 10.55 rate points, down from 47.78 — a factor of about 0.22.

So the forecaster has two regimes, chosen per trophy by how much the
history actually moved:

- **Flat history** (the page is a stale snapshot): the median is the last
  rate times a dilution factor `1 / (1 + S / 1.25)`, where `S` is the hours
  from the first read to the target, with a 0.6-logit spread.
- **Moving history** (the final's two days of reads): a blend, in logit
  space, of a saturating-exponential curve fit
  `R(t) = Rb + (C - Rb) * (1 - exp(-k t))` — the shape a growing, diluting
  player base produces — and the log-time linear fit over the last 24 hours,
  anchored at the last level with its slope shrunk by its own standard
  error. Their disagreement widens the spread.

A trophy already at 100% is forecast at 100% with a hair of lower tail.

## Practice results (local evaluation)

The practice answers are not in the kit, so the practice is scored against
a totals-consistent refresh model: non-100% rates scaled by ~0.22, the
100% trophy unchanged (this reproduces the house's stated persistence
figures: typical miss 0.68pp, coverage 0).

| entry | loss | typical miss | 90% coverage |
| --- | --- | --- | --- |
| persistence baseline | 1.405 | 0.68 pp | 0.00 |
| this forecaster (p00) | 0.126 | 0.01 pp | 1.00 |
| this forecaster (p01) | 0.142 | 0.02 pp | 1.00 |

The forecaster also beats persistence under perturbed versions of the
answer model (uniform factor 0.19, and lognormal per-trophy variation):
loss 0.12–0.43 versus persistence 1.40–1.72. On synthetic final-like
histories (two days of reads, saturating and log-linear truth shapes) it
scores 0.19–0.21 versus persistence 0.25–0.54 and the house curve
baseline 0.25–0.32. Those synthetic numbers are indicative only — the
true final shape is unknown, which is why the spread carries an explicit
shape-disagreement term.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```
