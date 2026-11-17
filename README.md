# GTA VI Twitch CCU Forecaster

![Competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
![Rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rangerkit/gta6-forecast/rank.svg)
![Reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/rangerkit/gta6-forecast/reward.svg)

An entry for the make.fun competition
[gta6-twitch-ccu-forecast-01](https://make.fun/competitions/gta6-twitch-ccu-forecast-01):
forecast, hour by hour, how many people will watch Grand Theft Auto VI live
on Twitch over the 24 UTC hours of its release day (2026-11-19), from
everything recorded up to the end of its first hour — five quantiles
(5%, 25%, 50%, 75%, 95%) for each of hours 2–24, scored by pinball loss on
the log scale (lower is better).

## The model

A launch day's audience curve is mostly a *shape*: how the count in hour h
compares with hour 1. That shape is shared across big launches that unlocked
the way GTA VI does — rolling midnight local, consoles first, no early
access — so `main.py` rebuilds it at runtime from the case's own
`comparables` field:

1. Align each comparable launch to its release day and form the log ratios
   `ln(print_h / print_1)` for hours 2..24.
2. Weight each launch by how good an analogue it is for GTA VI:
   early-access-contaminated ×0.25 (the kit's tier label), weak tier ×0.3,
   launches whose hour 1 was still nearly empty because a single global
   unlock landed later in the UTC day ×0.2 (GTA VI's hour 1 already holds
   New Zealand, Asia and Europe), console-only ×1.5, multiplayer-first /
   live-service ×0.5 (GTA VI is a single-player console launch).
3. Per hour, blend the weighted empirical quantiles of the log ratios
   (weight 0.75) with a smoothed normal fit of them (weight 0.25; per-hour
   weighted mean and sd, each smoothed over neighbouring hours), with one
   global spread multiplier of 0.90 on deviations from the mean.
4. Scale by the case's own hour-1 print and write the 23 rows.

If a case carries too few usable comparables, a baked-in table fitted the
same way on the practice cases' comparables is used instead, so the program
still writes a valid forecast. The fitted shape is the rolling-midnight
signature in UTC: a dip over hours 2–5 (to ~0.72× hour 1), then a climb
through Europe's evening and the Americas' unlock and prime time to ~2.4×
at hours 19–21, easing to ~2.1× by hour 24.

The program is a single standard-library Python file: no network, no
randomness, fully deterministic (two runs on the same case are
byte-identical), ~0.2 s per case against the 8 s cap.

## Validation (leave-one-out over the comparable launches)

The configuration was chosen by leave-one-out cross-validation over the
comparable launches themselves — hold one launch out, fit on the rest,
score its hours 2–24 with the competition's exact pinball loss — restricted
to the 19 launches of GTA VI's class (rolling midnight console unlocks, not
early-access contaminated, hour 1 already substantial), with all 37 usable
launches as a robustness check. Run through the actual entrypoint
(`tools/test_entry.py`, synthetic cases, separate processes):

| forecaster | LOO, target class (19) | LOO, all usable (37) |
| --- | --- | --- |
| house persistence baseline (kit starter) | 0.4026 | 0.9297 |
| **this entry** | **0.3179** | **0.7951** |

For reference, the house baseline scores ≈0.387 on the sealed practice
cases. `tools/calibrate.py` holds the offline variant comparison behind
these numbers (uniform vs tiered weights, raw empirical quantiles vs the
shrunk blend, the shrink × spread grid).

## Repository layout

- `main.py` — the entrypoint (the whole model, standard library only).
- `tools/calibrate.py` — comparable extraction, weighting schemes and the
  leave-one-out harness used to tune the configuration.
- `tools/test_entry.py` — strict output-contract validation, byte-identity
  and timing checks on the practice inputs, and leave-one-out scoring
  through the entrypoint itself.
- `.makefun/owner.json` — the entrant wallet for this entry.

## Running it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with .in cases> MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Each `<id>.in` in the input directory produces `<id>.out` in the output
directory: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`
with 23 rows for hours 2–24.

## License

MIT — see `LICENSE`.
