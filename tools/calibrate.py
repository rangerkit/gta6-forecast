"""Calibration and leave-one-out validation for the GTA VI Twitch CCU forecaster.

Reads the practice cases from the competition kit, pulls the comparable launches
out of them (every practice case carries the same 38 comparables), and evaluates
forecaster variants by leave-one-out cross-validation over the comparable
launches themselves: for each held-out launch, fit on the rest and score hours
2..24 against that launch's true day-1 prints with the competition's exact
pinball loss on the log scale,

    loss(item) = 2 * max(tau * d, (tau - 1) * d),  d = ln(y) - ln(q_tau)

with the print y floored at 1 viewer, averaged over every (launch, hour, level).

Usage:
    python3 tools/calibrate.py [--kit <kit input dir>] [--grid]

The kit input dir defaults to the downloaded kit's input/ folder.
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]
HOURS = list(range(2, 25))

DEFAULT_KIT = (
    Path.home()
    / "workspace/makefun-kits/gta6-twitch-ccu-forecast-01-kit"
    / "gta6-twitch-ccu-forecast-01-kit/input"
)

# Multiplayer-first / live-service franchises: GTA VI is a single-player launch,
# so these launches' day-1 audience shapes are weaker analogues.
MULTIPLAYER_NAMES = (
    "call of duty",
    "battlefield",
    "halo infinite",
    "marathon",
    "arc raiders",
    "marvel rivals",
    "helldivers",
)


def parse_ts(ts):
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


class Launch:
    """One comparable launch: its release-day hourly prints and research flags."""

    def __init__(self, comp):
        self.name = comp["name"]
        self.unlock = comp.get("unlock") or ""
        self.platforms = comp.get("platforms") or ""
        day_start = parse_ts(comp["day_start"])
        by_hour = {}
        for point in comp.get("hourly") or []:
            by_hour[parse_ts(point["from"])] = point.get("viewers")
        # prints[h] = the print of hour h (1..24) of the release day, or None.
        self.prints = {}
        for h in range(1, 25):
            v = by_hour.get(day_start + timedelta(hours=h - 1))
            self.prints[h] = v if (v is not None and v > 0) else None
        self.usable = self.prints[1] is not None and all(
            self.prints[h] is not None for h in HOURS
        )
        # log ratios r_h = print_h / print_1 for the scored hours.
        self.log_ratios = {}
        if self.prints[1] is not None:
            p1 = self.prints[1]
            for h in HOURS:
                if self.prints[h] is not None:
                    self.log_ratios[h] = math.log(self.prints[h] / p1)
        self.flags = self._flags()

    def _flags(self):
        text = self.unlock.lower()
        name = self.name.lower()
        day = [self.prints[h] for h in range(1, 25) if self.prints[h] is not None]
        peak = max(day) if day else 0
        return {
            # The kit's own tier label only: several clean launches' notes say
            # "no early access", which a bare keyword match would misread.
            "early_access": "tier: early-access-contaminated" in text,
            "weak_tier": "tier: weak" in text,
            "simultaneous_text": ("simultaneous" in text) or ("single global unlock" in text),
            # Hour 1 still nearly empty although the day gets big: the unlock
            # lands later in the UTC day (a single global unlock time), unlike
            # GTA VI's rolling midnight.
            "late_unlock_shape": bool(
                self.prints[1] is not None and peak > 0 and self.prints[1] / peak < 0.05
            ),
            "console_only": "pc" not in self.platforms.lower(),
            "multiplayer": any(name.startswith(m) for m in MULTIPLAYER_NAMES)
            or "live-service" in text
            or "free-to-play" in text
            or "f2p" in text,
        }


def load_launches(kit_dir):
    """The comparable launches shared by all practice cases (deduplicated)."""
    cases = sorted(Path(kit_dir).glob("p*.in"))
    if not cases:
        raise SystemExit(f"no practice cases under {kit_dir}")
    first = json.loads(cases[0].read_text(encoding="utf-8"))
    launches = [Launch(c) for c in first["comparables"]]
    # The other practice cases must carry the same comparables.
    for path in cases[1:]:
        other = json.loads(path.read_text(encoding="utf-8"))
        names = [c["name"] for c in other["comparables"]]
        if names != [c["name"] for c in first["comparables"]]:
            raise SystemExit(f"{path.name}: comparables differ from {cases[0].name}")
    return launches


# Named weighting schemes as explicit factor sets, so the grid can vary one
# factor at a time. A launch's weight is the product of the factors whose flag
# it carries. "simultaneous_text" turned out to mislabel the best analogues
# (ELDEN RING, Cyberpunk 2077, Tears of the Kingdom all mention a simultaneous
# sub-launch while their console unlock rolls at midnight and hour 1 is
# already their peak or near it), so the shape test that matters is the
# data-driven late_unlock_shape: hour 1 still holds <5% of the day's peak.
SCHEMES = {
    "uniform": {},
    "tier": {"early_access": 0.25, "weak_tier": 0.3},
    "tier_sim": {"early_access": 0.25, "weak_tier": 0.3, "late_unlock_shape": 0.2},
    "full": {
        "early_access": 0.25,
        "weak_tier": 0.3,
        "late_unlock_shape": 0.2,
        "console_only": 1.5,
        "multiplayer": 0.5,
    },
}


def weight_of(launch, scheme):
    """Deterministic analogue weight of one launch under a named scheme."""
    factors = SCHEMES[scheme] if isinstance(scheme, str) else scheme
    w = 1.0
    for flag, factor in factors.items():
        if launch.flags.get(flag):
            w *= factor
    return w


def weighted_quantile(sorted_pairs, tau):
    """Weighted quantile of (value, weight) pairs already sorted by value.

    Uses the inverse of the weighted empirical CDF with linear interpolation
    between the bracketing points (the "type 7"-style weighted analogue).
    """
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
    """Centered moving average over the hour axis (values maps hour -> float)."""
    out = {}
    for h in HOURS:
        lo, hi = max(2, h - radius), min(24, h + radius)
        out[h] = sum(values[k] for k in range(lo, hi + 1)) / (hi - lo + 1)
    return out


class EmpiricalForecaster:
    """Per-hour weighted empirical quantiles of the log ratios, shrunk toward a
    smoothed normal fit of the same ratios; one global spread multiplier."""

    def __init__(
        self, scheme="full", shrink=0.5, spread=1.0, smooth_radius=2, smooth_emp=False
    ):
        self.scheme = scheme
        self.shrink = shrink
        self.spread = spread
        self.smooth_radius = smooth_radius
        self.smooth_emp = smooth_emp
        self.quantiles = {}  # hour -> [log q_tau / p1 for each level]

    def fit(self, launches):
        train = [l for l in launches if l.usable]
        weights = {l.name: weight_of(l, self.scheme) for l in train}
        means, sds, emp = {}, {}, {}
        for h in HOURS:
            pairs = sorted(
                (l.log_ratios[h], weights[l.name]) for l in train if h in l.log_ratios
            )
            if not pairs:
                continue
            means[h], sds[h] = weighted_mean_sd(pairs)
            emp[h] = [weighted_quantile(pairs, tau) for tau in LEVELS]
        means = smooth(means, self.smooth_radius)
        sds = smooth(sds, self.smooth_radius)
        if self.smooth_emp:
            for i in range(len(LEVELS)):
                col = smooth({h: emp[h][i] for h in emp}, self.smooth_radius)
                for h in emp:
                    emp[h][i] = col[h]
        for h in HOURS:
            if h not in emp:
                continue
            row = []
            for i, tau in enumerate(LEVELS):
                normal_q = means[h] + Z[i] * sds[h] * self.spread
                emp_q = emp[h][i]
                # Widen the empirical quantile away from the smoothed mean by
                # the same spread multiplier before blending.
                emp_q = means[h] + (emp_q - means[h]) * self.spread
                row.append((1.0 - self.shrink) * normal_q + self.shrink * emp_q)
            self.quantiles[h] = row
        return self

    def forecast(self, p1):
        return {
            h: [p1 * math.exp(v) for v in self.quantiles[h]] for h in HOURS
        }


def baseline_forecast(p1):
    """The house baseline (the kit's starter), number for number."""
    return {
        h: [p1 * math.exp(z * 0.15 * math.sqrt(h - 1)) for z in Z] for h in HOURS
    }


def pinball_loss(forecast, launch):
    """Mean pinball loss of a forecast {hour: [q...]} against a launch's prints."""
    total, n = 0.0, 0
    for h in HOURS:
        y = max(1.0, launch.prints[h])
        log_y = math.log(y)
        for tau, q in zip(LEVELS, forecast[h]):
            d = log_y - math.log(max(q, 1e-12))
            total += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return total / n


def target_class(launch):
    """The launches whose day-1 shape is the same class as GTA VI's: a rolling
    midnight-local console unlock (so hour 1 already holds a real audience),
    not early-access contaminated, not a weak-tier analogue."""
    f = launch.flags
    return not (f["early_access"] or f["weak_tier"] or f["late_unlock_shape"])


def loo_eval(launches, make_forecaster, targets=None):
    """Leave-one-out: fit on all usable launches but one, score the held one.

    `targets` optionally restricts which launches are held out (the fit still
    sees every other usable launch, weighted as the forecaster chooses).
    """
    usable = [l for l in launches if l.usable]
    held_set = targets if targets is not None else usable
    losses = {}
    for held in held_set:
        train = [l for l in usable if l is not held]
        forecaster = make_forecaster(train)
        losses[held.name] = pinball_loss(forecaster(held.prints[1]), held)
    mean = sum(losses.values()) / len(losses)
    return mean, losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(DEFAULT_KIT))
    ap.add_argument("--grid", action="store_true", help="run the tuning grid")
    args = ap.parse_args()

    launches = load_launches(args.kit)
    usable = [l for l in launches if l.usable]
    print(f"comparables: {len(launches)}, usable launches: {len(usable)}")
    for l in launches:
        if not l.usable:
            print(f"  not usable (no full release day): {l.name}")
    flagged = [l.name for l in usable if l.flags["early_access"]]
    print(f"early-access contaminated ({len(flagged)}): {', '.join(flagged)}")
    late = [l.name for l in usable if l.flags["late_unlock_shape"]]
    print(f"late-unlock shape ({len(late)}): {', '.join(late)}")

    variants = [
        ("house baseline formula", lambda train: baseline_forecast),
        ("empirical, uniform weights", lambda train: EmpiricalForecaster("uniform", 1.0, 1.0).fit(train).forecast),
        ("empirical, tier weights", lambda train: EmpiricalForecaster("tier", 1.0, 1.0).fit(train).forecast),
        ("empirical, tier+sim weights", lambda train: EmpiricalForecaster("tier_sim", 1.0, 1.0).fit(train).forecast),
        ("empirical, full weights", lambda train: EmpiricalForecaster("full", 1.0, 1.0).fit(train).forecast),
        ("normal fit, full weights", lambda train: EmpiricalForecaster("full", 0.0, 1.0).fit(train).forecast),
        ("shrunk 0.5, full weights", lambda train: EmpiricalForecaster("full", 0.5, 1.0).fit(train).forecast),
    ]
    targets = [l for l in usable if target_class(l)]
    print(f"\ntarget class (GTA-like rolling console launches): {len(targets)}")
    for l in targets:
        print(f"  {l.name}")

    print("\nleave-one-out, all usable targets (exact pinball, lower better):")
    for name, make in variants:
        mean, _ = loo_eval(launches, make)
        print(f"  {name:34s} {mean:.4f}")
    print("\nleave-one-out, target-class targets only (the honest compass):")
    target_results = {}
    for name, make in variants:
        mean, _ = loo_eval(launches, make, targets=targets)
        target_results[name] = mean
        print(f"  {name:34s} {mean:.4f}")

    if args.grid:
        print("\nfactor-set comparison (shrink 0.75, spread 0.85), target-class targets:")
        factor_sets = {
            "tier only": {"early_access": 0.25, "weak_tier": 0.3},
            "tier + late 0.2": {"early_access": 0.25, "weak_tier": 0.3, "late_unlock_shape": 0.2},
            "tier + late 0.1": {"early_access": 0.25, "weak_tier": 0.3, "late_unlock_shape": 0.1},
            "+ console 1.5": {"early_access": 0.25, "weak_tier": 0.3, "late_unlock_shape": 0.2, "console_only": 1.5},
            "+ multiplayer 0.5 (= full)": dict(SCHEMES["full"]),
            "full, no console boost": {"early_access": 0.25, "weak_tier": 0.3, "late_unlock_shape": 0.2, "multiplayer": 0.5},
            "full, emp smoothed": dict(SCHEMES["full"]),
        }
        for name, factors in factor_sets.items():
            mean, _ = loo_eval(
                launches,
                lambda train, f=factors, se=(name == "full, emp smoothed"): EmpiricalForecaster(
                    f, 0.75, 0.85, smooth_emp=se
                ).fit(train).forecast,
                targets=targets,
            )
            print(f"  {name:28s} {mean:.4f}")

        print("\ngrid over shrink x spread (full weights), target-class targets:")
        best = None
        for shrink in (0.5, 0.65, 0.75, 0.85, 1.0):
            row = []
            for spread in (0.75, 0.85, 0.95, 1.05):
                mean, _ = loo_eval(
                    launches,
                    lambda train, s=shrink, g=spread: EmpiricalForecaster(
                        "full", s, g
                    ).fit(train).forecast,
                    targets=targets,
                )
                row.append(f"{mean:.4f}")
                if best is None or mean < best[0]:
                    best = (mean, shrink, spread)
            print(f"  shrink={shrink:<5} spread 0.75/0.85/0.95/1.05: {'  '.join(row)}")
        print(f"best: loss {best[0]:.4f} at shrink={best[1]}, spread={best[2]}")

        print("\nper-launch losses of the best configuration (target-class targets):")
        _, losses = loo_eval(
            launches,
            lambda train: EmpiricalForecaster("full", best[1], best[2]).fit(train).forecast,
            targets=targets,
        )
        for name in sorted(losses, key=losses.get):
            print(f"  {losses[name]:.4f}  {name}")


if __name__ == "__main__":
    main()
