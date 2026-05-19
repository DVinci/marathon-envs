#!/usr/bin/env python3
"""Find optimal --num-envs and --num-spawn-envs for each Marathon Envs build.

Tests a grid of (num_envs, num_spawn_envs) combinations, measures steady-state
steps/sec throughput for each, and saves the best combination per environment to
config/optimal_spawn_envs.json for automatic use by train_all_envs.py.

Grid defaults:
  num_envs:      1, 2, 4
  num_spawn_envs: 4, 10, 25, 50, 100, 200

Each trial runs 10k steps (summary at 5k and 10k). Throughput is measured from
the 5k->10k interval to exclude initialisation overhead. The inner spawn loop
stops when improvement drops below 5% (plateau) or the run times out.

Usage:
    python calibrate_envs.py
    python calibrate_envs.py --envs Walker2d-v0 Ant-v0
    python calibrate_envs.py --num-envs-levels 1 2 4
    python calibrate_envs.py --spawn-levels 4 10 50 100 200
    python calibrate_envs.py --run-prefix calib_0519

Expected runtime: ~45-75 minutes for all 16 environments.
Delete artifacts afterwards:
    Remove-Item -Recurse results\\calib_* summaries\\calib_*
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Force UTF-8 output so Unicode in ML-Agents logs doesn't crash on Windows cp1252
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

ENVS = [
    "Hopper-v0",
    "Walker2d-v0",
    "Ant-v0",
    "MarathonMan-v0",
    "MarathonManSparse-v0",
    "TerrainHopper-v0",
    "TerrainWalker2d-v0",
    "TerrainAnt-v0",
    "TerrainMarathonMan-v0",
    "MarathonManWalking-v0",
    "MarathonManRunning-v0",
    "MarathonManJazzDancing-v0",
    "MarathonManMMAKick-v0",
    "MarathonManPunchingBag-v0",
    "MarathonManBackflip-v0",
    "ControllerMarathonMan-v0",
]

CALIB_CONFIG = "config/marathon_envs_calibration_config.yaml"
BUILDS_DIR   = Path("builds")
EXE_NAME     = "Marathon Environments.exe"
OPTIMAL_FILE = Path("config/optimal_spawn_envs.json")

DEFAULT_NUM_ENVS_LEVELS = [1, 2, 4]
DEFAULT_SPAWN_LEVELS    = [4, 10, 25, 50, 100, 200]
PLATEAU_THRESHOLD       = 0.05  # stop if < 5% throughput gain vs previous level

_THROUGHPUT_RE = re.compile(r"Step:\s+(\d+)\..*?Time Elapsed:\s+([\d.]+)\s+s")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _kill_unity() -> None:
    """Kill any lingering Unity environment processes to free ports."""
    subprocess.run(
        ["taskkill", "/F", "/IM", EXE_NAME],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["taskkill", "/F", "/IM", "mlagents-learn.exe"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _run(env_id: str, run_id: str, num_envs: int, num_spawn_envs: int) -> tuple[float, int]:
    """Run one calibration trial. Returns (steps_per_sec, return_code)."""
    _kill_unity()

    exe = BUILDS_DIR / env_id / EXE_NAME

    cmd = [
        "mlagents-learn", CALIB_CONFIG,
        f"--run-id={run_id}",
        f"--env={exe}",
        "--no-graphics",
        "--force",
        "--timeout-wait", "120",
        f"--num-envs={num_envs}",
        "--env-args", f"--spawn-env={env_id}", f"--num-spawn-envs={num_spawn_envs}",
    ]

    total = num_envs * num_spawn_envs
    print(f"    num_envs={num_envs}  spawn={num_spawn_envs}  (total={total}) ...", flush=True)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )

    data_points: list[tuple[int, float]] = []

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        m = _THROUGHPUT_RE.search(line)
        if m:
            data_points.append((int(m.group(1)), float(m.group(2))))

    proc.wait()

    if len(data_points) >= 2:
        # Use last two intervals to avoid init overhead
        p1, p2 = data_points[-2], data_points[-1]
        dt = p2[1] - p1[1]
        tput = (p2[0] - p1[0]) / dt if dt > 0 else 0.0
    elif data_points:
        # Only one point - use cumulative rate (includes init overhead)
        tput = data_points[0][0] / data_points[0][1] if data_points[0][1] > 0 else 0.0
    else:
        tput = 0.0

    status = f"{tput:.1f} steps/s" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"    -> {status}", flush=True)
    return tput, proc.returncode


# ---------------------------------------------------------------------------
# Per-environment calibration
# ---------------------------------------------------------------------------

def _calibrate_env(
    env_id: str,
    prefix: str,
    num_envs_levels: list[int],
    spawn_levels: list[int],
) -> dict | None:
    exe = BUILDS_DIR / env_id / EXE_NAME
    if not exe.exists():
        print(f"  [SKIP] {env_id} - build not found: {exe}")
        return None

    best: dict = {"num_envs": num_envs_levels[0], "num_spawn_envs": spawn_levels[0], "steps_per_sec": 0.0}

    for num_envs in num_envs_levels:
        print(f"\n  Testing num_envs={num_envs} ...")
        prev_tput = 0.0
        best_spawn_tput = 0.0
        best_spawn = spawn_levels[0]

        for spawn in spawn_levels:
            run_id = f"{prefix}-{env_id}-n{num_envs}s{spawn}"
            tput, rc = _run(env_id, run_id, num_envs, spawn)

            if rc != 0:
                print(f"  -> timeout/error - stopping num_envs={num_envs} at spawn={spawn}")
                break

            if tput > best_spawn_tput:
                best_spawn_tput = tput
                best_spawn = spawn

            if prev_tput > 0 and tput <= prev_tput * (1 + PLATEAU_THRESHOLD):
                print(f"  -> plateau - stopping num_envs={num_envs} at spawn={spawn}")
                break

            prev_tput = tput

        if best_spawn_tput > best["steps_per_sec"]:
            best = {
                "num_envs": num_envs,
                "num_spawn_envs": best_spawn,
                "steps_per_sec": round(best_spawn_tput, 1),
            }

    return best


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate optimal parallelism parameters for each Marathon Envs build."
    )
    parser.add_argument(
        "--envs", nargs="+", metavar="ENV_ID",
        help="Calibrate only these environments (default: all 16).",
    )
    parser.add_argument(
        "--num-envs-levels", nargs="+", type=int, default=DEFAULT_NUM_ENVS_LEVELS, metavar="N",
        help=f"num_envs values to test (default: {DEFAULT_NUM_ENVS_LEVELS}).",
    )
    parser.add_argument(
        "--spawn-levels", nargs="+", type=int, default=DEFAULT_SPAWN_LEVELS, metavar="S",
        help=f"num_spawn_envs values to test (default: {DEFAULT_SPAWN_LEVELS}).",
    )
    parser.add_argument(
        "--run-prefix", default=None, metavar="PREFIX",
        help="Prefix for run IDs (default: calib_YYYYMMDD).",
    )
    args = parser.parse_args()

    prefix = args.run_prefix or ("calib_" + datetime.date.today().strftime("%Y%m%d"))
    envs = args.envs or ENVS

    unknown = [e for e in envs if e not in ENVS]
    if unknown:
        parser.error(f"Unknown environment IDs: {', '.join(unknown)}")

    total_combinations = len(envs) * len(args.num_envs_levels) * len(args.spawn_levels)
    print(f"\n{'='*60}")
    print(f"  Marathon Envs - Parallelism Calibration")
    print(f"{'='*60}")
    print(f"  num_envs levels : {args.num_envs_levels}")
    print(f"  spawn levels    : {args.spawn_levels}")
    print(f"  environments    : {len(envs)}")
    print(f"  max combinations: {total_combinations}  (early stopping will cut this)")
    print(f"  run prefix      : {prefix}")
    print(f"{'='*60}\n")

    results: dict[str, dict] = {}

    for env_id in envs:
        print(f"\n{'='*60}")
        print(f"  Calibrating {env_id}")
        print(f"{'='*60}")
        best = _calibrate_env(env_id, prefix, args.num_envs_levels, args.spawn_levels)
        if best is not None:
            results[env_id] = best
            print(
                f"\n  [BEST] {env_id}: "
                f"num_envs={best['num_envs']}  spawn={best['num_spawn_envs']}  "
                f"-> {best['steps_per_sec']:.1f} steps/s"
            )

    # Merge with any existing results so per-env runs accumulate
    if results:
        existing: dict = {}
        if OPTIMAL_FILE.exists():
            existing = json.loads(OPTIMAL_FILE.read_text())
        existing.update(results)
        OPTIMAL_FILE.write_text(json.dumps(existing, indent=2))

    # Print summary table
    divider = "=" * 72
    thin    = "-" * 72
    print(f"\n{divider}")
    print("CALIBRATION RESULTS")
    print(divider)
    print(f"  {'Environment':<32} {'num_envs':>8}  {'spawn':>6}  {'steps/s':>9}")
    print(f"  {thin}")
    for env_id in envs:
        if env_id in results:
            r = results[env_id]
            print(f"  {env_id:<32} {r['num_envs']:>8}  {r['num_spawn_envs']:>6}  {r['steps_per_sec']:>9.1f}")
        else:
            print(f"  {env_id:<32}   SKIPPED (build not found)")
    print(divider)

    if results:
        print(f"\n  Results saved to {OPTIMAL_FILE}")
        print("  Run full training:  python train_all_envs.py\n")
    print("  Delete calibration artifacts when no longer needed:")
    print("    Remove-Item -Recurse results\\calib_* summaries\\calib_*\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
