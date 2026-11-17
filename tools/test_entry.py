"""End-to-end tests for the entrypoint (main.py at the repository root).

1. Practice run: run main.py the way the grader does (python3 -P -s,
   PYTHONHASHSEED=0) on the kit's four practice inputs; validate every
   output strictly against the forecast contract (one `hours` key, 23 rows,
   hours 2..24 each once, five finite quantiles in (0, 1e9], non-decreasing,
   at most 65,536 bytes); run it twice and require byte-identical outputs;
   time each single-case run against the 8 s cap.
2. Leave-one-out through the entrypoint: for every usable comparable
   launch, build a synthetic case whose hour-1 print is that launch's and
   whose comparables are the other launches, run BOTH this entry and the
   kit's starter (the house baseline) on it as separate processes, and score
   both with the competition's exact pinball loss against the launch's true
   hours 2..24. This is the honest number for the entrypoint logic itself,
   not just for the offline fit in calibrate.py.

Usage: python3 tools/test_entry.py [--kit <kit dir>]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibrate import DEFAULT_KIT, LEVELS, Launch, load_launches, pinball_loss, target_class

REPO = Path(__file__).resolve().parent.parent
ENTRY = REPO / "main.py"


def run_program(program, case, workdir):
    """Run a program once on a single synthetic/practice case; return (ms, output dict)."""
    in_dir = Path(workdir) / "in"
    out_dir = Path(workdir) / "out"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    (in_dir / f"{case['id']}.in").write_text(json.dumps(case), encoding="utf-8")
    env = dict(os.environ)
    env["MAKEFUN_INPUT_DIR"] = str(in_dir)
    env["MAKEFUN_OUTPUT_DIR"] = str(out_dir)
    env["PYTHONHASHSEED"] = "0"
    start = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-P", "-s", str(program)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    ms = (time.perf_counter() - start) * 1000
    if proc.returncode != 0:
        raise RuntimeError(f"{program} failed on {case['id']}: {proc.stderr.strip()}")
    out_path = out_dir / f"{case['id']}.out"
    return ms, out_path.read_bytes()


def validate_output(raw, case_id):
    """Strict contract check of one output file's bytes."""
    assert len(raw) <= 65536, f"{case_id}: output too large ({len(raw)} bytes)"
    doc = json.loads(raw.decode("utf-8"))
    assert set(doc.keys()) == {"hours"}, f"{case_id}: top-level keys {set(doc.keys())}"
    rows = doc["hours"]
    assert len(rows) == 23, f"{case_id}: {len(rows)} rows, want 23"
    seen = set()
    for row in rows:
        assert set(row.keys()) == {"hour", "q"}, f"{case_id}: row keys {set(row.keys())}"
        h = row["hour"]
        assert 2 <= h <= 24 and h not in seen, f"{case_id}: bad/duplicate hour {h}"
        seen.add(h)
        q = row["q"]
        assert len(q) == 5, f"{case_id} h{h}: {len(q)} quantiles"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), f"{case_id} h{h}: {v}"
            assert 0 < v <= 1e9, f"{case_id} h{h}: q out of bounds: {v}"
        assert all(q[i] <= q[i + 1] for i in range(4)), f"{case_id} h{h}: not monotone"
    assert seen == set(range(2, 25)), f"{case_id}: hours {sorted(seen)}"
    return doc


def synthetic_case(held, all_comps):
    """A case that looks like a practice case for the held-out launch."""
    from calibrate import parse_ts

    day_start = parse_ts(held_comp_day_start(all_comps, held.name))
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return {
        "schema": "makefun/twitch-ccu-case/v1",
        "id": "loo",
        "game": {"name": held.name, "twitch_id": "0"},
        "day_start": day_start.strftime(fmt),
        "checkpoint": (day_start + timedelta(hours=1)).strftime(fmt),
        "levels": LEVELS,
        "known": [
            {
                "hour": 1,
                "from": day_start.strftime(fmt),
                "until": (day_start + timedelta(hours=1)).strftime(fmt),
                "viewers": held.prints[1],
                "channels": None,
                "reads": 1,
            }
        ],
        "forecast": [
            {
                "hour": h,
                "from": (day_start + timedelta(hours=h - 1)).strftime(fmt),
                "until": (day_start + timedelta(hours=h)).strftime(fmt),
            }
            for h in range(2, 25)
        ],
        "history": [],
        "history_hourly": [],
        "streams": None,
        "categories_hourly": None,
        "comparables": [c for c in all_comps if c["name"] != held.name],
        "notes": "synthetic leave-one-out case built by tools/test_entry.py",
    }


def held_comp_day_start(all_comps, name):
    for c in all_comps:
        if c["name"] == name:
            return c["day_start"]
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", default=str(DEFAULT_KIT.parent))
    args = ap.parse_args()
    kit_dir = Path(args.kit)
    starter = kit_dir / "starter" / "main.py"

    # ---- 1. practice run: shape, determinism, timing ----------------------
    practice = json.loads((kit_dir / "input" / "p00.in").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as td:
        runs = []
        for run_no in (1, 2):
            outputs = {}
            for pid in ("p00", "p01", "p02", "p03"):
                case = json.loads((kit_dir / "input" / f"{pid}.in").read_text(encoding="utf-8"))
                ms, raw = run_program(ENTRY, case, Path(td) / f"r{run_no}-{pid}")
                doc = validate_output(raw, pid)
                outputs[pid] = raw
                if run_no == 1:
                    print(f"practice {pid}: valid output, {len(raw)} bytes, run {ms:.0f} ms")
            runs.append(outputs)
        for pid in outputs:
            assert runs[0][pid] == runs[1][pid], f"{pid}: two runs differ"
        print("byte-identity: all 4 practice outputs identical across two runs")

    # ---- 2. leave-one-out through the entrypoint --------------------------
    launches = load_launches(kit_dir / "input")
    usable = [l for l in launches if l.usable]
    all_comps = practice["comparables"]
    ours, house = {}, {}
    with tempfile.TemporaryDirectory() as td:
        for i, held in enumerate(usable):
            case = synthetic_case(held, all_comps)
            _, raw = run_program(ENTRY, case, Path(td) / f"o{i}")
            doc = validate_output(raw, f"loo-{held.name}")
            ours[held.name] = pinball_loss(
                {row["hour"]: row["q"] for row in doc["hours"]}, held
            )
            _, raw = run_program(starter, case, Path(td) / f"h{i}")
            doc = validate_output(raw, f"loo-house-{held.name}")
            house[held.name] = pinball_loss(
                {row["hour"]: row["q"] for row in doc["hours"]}, held
            )
    targets = [l.name for l in usable if target_class(l)]
    for label, names in (("all usable", [l.name for l in usable]), ("target class", targets)):
        mo = sum(ours[n] for n in names) / len(names)
        mh = sum(house[n] for n in names) / len(names)
        print(f"LOO through entrypoint, {label} targets ({len(names)}): ours {mo:.4f} | house starter {mh:.4f}")
    print("\nworst 5 target-class launches for our entry:")
    for name in sorted(targets, key=lambda n: ours[n])[-5:]:
        print(f"  {ours[name]:.4f}  {name} (house {house[name]:.4f})")


if __name__ == "__main__":
    main()
