"""GTA VI Twitch CCU forecaster (make.fun gta6-twitch-ccu-forecast-01).

The model, in one paragraph: a launch day's audience curve, hour by hour, is
mostly a *shape* — how the count in hour h compares with hour 1 — and that
shape is shared across big launches that unlocked the way GTA VI does
(rolling midnight local, consoles, no early access). So this program reads
the case's own `comparables` (past launches' hourly curves), aligns each one
to its release day, forms the log ratios ln(print_h / print_1) for hours
2..24, and pools them with analogue weights: early-access-contaminated
launches count 0.25, weak-tier ones 0.3, launches whose hour 1 was still
nearly empty because a single global unlock landed later in the UTC day
count 0.2 (GTA VI's hour 1 already holds New Zealand, Asia and Europe),
console-only launches count 1.5 and multiplayer-first / live-service ones
0.5 (GTA VI is a single-player console launch). For each hour the forecast's
log quantiles are a shrinkage blend (0.75 / 0.25) of the weighted empirical
quantiles of those log ratios and a smoothed normal fit of them (per-hour
weighted mean and sd, each smoothed over neighbouring hours), with one
global spread multiplier of 0.90 on the deviations from the mean. The blend
is scaled by the case's own hour-1 print. The configuration was chosen by
leave-one-out cross-validation over the comparable launches (see
tools/calibrate.py): it scores about 0.32 there, against 0.40 for the house
persistence baseline on the same target-class launches.

If a case carries too few usable comparables, the program falls back to a
baked-in quantile table fitted the same way on the practice cases'
comparables (FALLBACK_LOGQ below), so it still writes a valid forecast.

Determinism: no randomness, no wall clock, iteration only over sorted file
names and explicitly sorted values; two runs on the same case write the same
bytes. Standard library only, no network.

Run by the grader as `python3 -P -s /bot/main.py`; locally:
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, in case a
# module shipped beside this file is ever imported.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

# --- Calibrated configuration (tools/calibrate.py, leave-one-out) ----------
EARLY_ACCESS_FACTOR = 0.25   # kit tier "early-access-contaminated"
WEAK_TIER_FACTOR = 0.3       # kit tier "weak"
LATE_UNLOCK_FACTOR = 0.2     # hour 1 < 5% of the day's peak: single global unlock
CONSOLE_ONLY_FACTOR = 1.5    # no PC in the launch platforms
MULTIPLAYER_FACTOR = 0.5     # multiplayer-first / live-service franchise
SHRINK = 0.75                # weight on the empirical quantiles in the blend
SPREAD = 0.90                # global multiplier on deviations from the mean
SMOOTH_RADIUS = 2            # hours either side in the mean/sd smoothing
MIN_LAUNCHES = 8             # below this, the baked-in fallback table is used

# Multiplayer-first / live-service franchises (weaker analogues for a
# single-player launch), matched on the comparable's name and research text.
MULTIPLAYER_NAMES = (
    "call of duty",
    "battlefield",
    "halo infinite",
    "marathon",
    "arc raiders",
    "marvel rivals",
    "helldivers",
)

# Baked-in fallback: per-hour log quantiles of print_h / print_1, produced by
# this file's own fit_log_quantiles over the practice cases' 38 comparables
# (37 usable), so the fallback is the same model with the practice pool when
# a case's own comparables are missing or degenerate. The shape is the
# rolling-midnight signature in UTC: a dip over hours 2-5 (to ~0.72x hour 1),
# then a climb through Europe's evening and the Americas' unlock and prime
# time to ~2.4x at hours 19-21, easing to ~2.1x by hour 24.
FALLBACK_LOGQ = {
    2: [-0.513171, -0.317627, -0.195847, -0.069081, 0.260283],
    3: [-1.123169, -0.564221, -0.313414, -0.137161, 0.166625],
    4: [-0.843226, -0.591875, -0.330282, -0.170751, 0.088488],
    5: [-0.761811, -0.460029, -0.234041, -0.049106, 0.248563],
    6: [-0.906245, -0.450422, -0.153082, 0.087191, 0.861804],
    7: [-0.816622, -0.476656, -0.092497, 0.151740, 0.960185],
    8: [-0.754790, -0.524872, -0.079076, 0.186401, 1.173733],
    9: [-0.840125, -0.579678, -0.103031, 0.265518, 1.090382],
    10: [-0.891930, -0.569536, -0.114230, 0.227264, 1.199855],
    11: [-0.875655, -0.563598, -0.058513, 0.352431, 1.340333],
    12: [-0.836047, -0.449141, 0.047472, 0.476379, 1.429001],
    13: [-0.731403, -0.302754, 0.170904, 0.696058, 1.639380],
    14: [-0.770060, -0.198370, 0.310610, 0.871926, 1.861240],
    15: [-0.537583, -0.042109, 0.490621, 1.034362, 2.038768],
    16: [-0.545254, 0.020404, 0.574779, 1.104594, 2.138901],
    17: [-0.479711, 0.140523, 0.644334, 1.120655, 2.237035],
    18: [-0.318226, 0.223944, 0.730955, 1.185079, 2.330217],
    19: [-0.207971, 0.295103, 0.820475, 1.357664, 2.447874],
    20: [-0.223501, 0.317971, 0.885807, 1.352001, 2.492589],
    21: [-0.217292, 0.284455, 0.884974, 1.365796, 2.450725],
    22: [-0.251224, 0.222492, 0.847153, 1.329213, 2.346040],
    23: [-0.345755, 0.206287, 0.815479, 1.261298, 2.270973],
    24: [-0.441334, 0.217140, 0.738708, 1.161881, 2.182190],
}


def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round2(x):
    """x rounded to 2 decimals, half up (as the house starter rounds).

    A whole number comes back as an int, so json.dumps writes 1240, not
    1240.0; either way the bytes are deterministic.
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def launch_weight(name, unlock, platforms, p1, peak):
    """The analogue weight of one comparable launch for a GTA VI forecast."""
    text = (unlock or "").lower()
    lname = (name or "").lower()
    w = 1.0
    if "tier: early-access-contaminated" in text:
        w *= EARLY_ACCESS_FACTOR
    if "tier: weak" in text:
        w *= WEAK_TIER_FACTOR
    if peak and p1 / peak < 0.05:
        w *= LATE_UNLOCK_FACTOR
    if "pc" not in (platforms or "").lower():
        w *= CONSOLE_ONLY_FACTOR
    if (
        any(lname.startswith(m) for m in MULTIPLAYER_NAMES)
        or "live-service" in text
        or "free-to-play" in text
        or "f2p" in text
    ):
        w *= MULTIPLAYER_FACTOR
    return w


def launch_curve(comp):
    """One comparable -> (weight, {hour: log ratio to hour 1}) or None.

    The comparable's release day is aligned to its `day_start`: hour h is the
    hourly point starting at day_start + (h - 1) hours. A comparable with no
    positive hour-1 print (the GTA VI category itself before release) cannot
    anchor ratios and returns None.
    """
    try:
        day_start = parse_ts(comp["day_start"])
    except (KeyError, TypeError, ValueError):
        return None
    by_hour = {}
    for point in comp.get("hourly") or []:
        try:
            by_hour[parse_ts(point["from"])] = point.get("viewers")
        except (KeyError, TypeError, ValueError):
            continue
    prints = {}
    for h in range(1, 25):
        v = by_hour.get(day_start + timedelta(hours=h - 1))
        if v is not None and v > 0:
            prints[h] = float(v)
    if 1 not in prints:
        return None
    p1 = prints[1]
    peak = max(prints.values())
    ratios = {h: math.log(v / p1) for h, v in prints.items() if h >= 2}
    if not ratios:
        return None
    weight = launch_weight(comp.get("name"), comp.get("unlock"), comp.get("platforms"), p1, peak)
    return weight, ratios


def weighted_quantile(sorted_pairs, tau):
    """Weighted quantile of (value, weight) pairs sorted by value: the inverse
    of the weighted empirical CDF, linearly interpolated between points."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    target = tau * total
    acc = 0.0
    prev_v, prev_acc = sorted_pairs[0][0], 0.0
    for v, w in sorted_pairs:
        acc += w
        if acc >= target:
            if acc == prev_acc:
                return v
            frac = (target - prev_acc) / (acc - prev_acc)
            return prev_v + frac * (v - prev_v)
        prev_v, prev_acc = v, acc
    return sorted_pairs[-1][0]


def weighted_mean_sd(pairs):
    total = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total
    var = sum(w * (v - mean) ** 2 for v, w in pairs) / total
    return mean, math.sqrt(max(var, 0.0))


def smooth(values, radius):
    """Centered moving average over the hour axis."""
    out = {}
    for h in HOURS:
        lo, hi = max(2, h - radius), min(24, h + radius)
        out[h] = sum(values[k] for k in range(lo, hi + 1) if k in values) / sum(
            1 for k in range(lo, hi + 1) if k in values
        )
    return out


def fit_log_quantiles(comparables):
    """Pool the comparables' log-ratio curves into per-hour log quantiles.

    Returns {hour: [log q_tau / p1 for tau in LEVELS]}, or None when too few
    usable comparables remain (the caller then uses FALLBACK_LOGQ).
    """
    curves = []
    for comp in comparables or []:
        curve = launch_curve(comp)
        if curve is not None:
            curves.append(curve)
    if len(curves) < MIN_LAUNCHES:
        return None
    means, sds, emp = {}, {}, {}
    for h in HOURS:
        pairs = sorted((ratios[h], w) for w, ratios in curves if h in ratios)
        if len(pairs) < MIN_LAUNCHES:
            continue
        means[h], sds[h] = weighted_mean_sd(pairs)
        emp[h] = [weighted_quantile(pairs, tau) for tau in LEVELS]
    if len(emp) < len(HOURS):
        return None
    means = smooth(means, SMOOTH_RADIUS)
    sds = smooth(sds, SMOOTH_RADIUS)
    logq = {}
    for h in HOURS:
        row = []
        for i in range(len(LEVELS)):
            normal_q = means[h] + Z[i] * sds[h] * SPREAD
            emp_q = means[h] + (emp[h][i] - means[h]) * SPREAD
            row.append((1.0 - SHRINK) * normal_q + SHRINK * emp_q)
        logq[h] = row
    return logq


def hour1_print(case):
    """The case's hour-1 print (at least 1), from its `known` row(s)."""
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    for row in rows:
        if isinstance(row, dict) and row.get("hour") == 1:
            try:
                return max(1.0, float(row["viewers"]))
            except (KeyError, TypeError, ValueError):
                return 1.0
    return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    logq = fit_log_quantiles(case.get("comparables")) or FALLBACK_LOGQ
    hours = []
    rows = case.get("forecast") or [{"hour": h} for h in HOURS]
    for row in rows:
        h = row["hour"]
        qs = []
        for v in logq.get(h, FALLBACK_LOGQ[h]):
            q = p1 * math.exp(max(-20.0, min(v, math.log(1e9 / p1))))
            qs.append(max(0.01, min(1e9, round2(q))))
        qs.sort()
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
