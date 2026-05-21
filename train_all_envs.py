#!/usr/bin/env python3
"""Train all (or a subset of) Marathon Envs builds sequentially.

Each build in builds/<envId>/ is run through mlagents-learn in turn.
Builds that don't exist yet are skipped with a warning.
Best-model tracking (same as train_ppo.py) is applied to every run.

Usage:
    python train_all_envs.py
    python train_all_envs.py --run-prefix run01 --num-spawn-envs 8
    python train_all_envs.py --envs Hopper-v0 Walker2d-v0 Ant-v0
    python train_all_envs.py --resume          # continue existing runs
    python train_all_envs.py --no-graphics     # already default; use --graphics to disable
"""

import argparse
import datetime
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

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

CONFIG       = "config/marathon_envs_config.yaml"
BUILDS_DIR   = Path("builds")
EXE_NAME     = "Marathon Environments.exe"
OPTIMAL_FILE = Path("config/optimal_spawn_envs.json")
STATE_FILE   = Path("train_all_envs.last_run")

_REWARD_RE = re.compile(r"Mean Reward:\s+([-\d]+(?:\.\d+)?)")
_EXPORT_RE = re.compile(r"Exported\s+(.+\.onnx)")


def _train_one(
    env_id: str, run_id: str, args: argparse.Namespace,
    num_envs: int, num_spawn_envs: int,
) -> bool | None:
    exe = BUILDS_DIR / env_id / EXE_NAME
    if not exe.exists():
        print(f"[SKIP] {env_id} - build not found: {exe}")
        return None

    cmd = [
        "mlagents-learn", CONFIG,
        f"--run-id={run_id}",
        f"--env={exe}",
    ]
    if not args.graphics:
        cmd.append("--no-graphics")
    if num_envs > 1:
        cmd.append(f"--num-envs={num_envs}")
    if args.resume:
        cmd.append("--resume")
    cmd += ["--env-args", f"--spawn-env={env_id}", f"--num-spawn-envs={num_spawn_envs}"]

    divider = "=" * 64
    print(f"\n{divider}")
    print(f"[START] {env_id}   run-id: {run_id}")
    print(f"  {' '.join(str(c) for c in cmd)}")
    print(f"{divider}\n", flush=True)

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    global_best = None  # type: float | None
    window_best = float("-inf")

    try:
        for line in proc.stdout:
            sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
            sys.stdout.flush()

            m = _REWARD_RE.search(line)
            if m:
                reward = float(m.group(1))
                if reward > window_best:
                    window_best = reward

            m = _EXPORT_RE.search(line)
            if m:
                onnx_path = Path(m.group(1).strip())
                if global_best is None or window_best > global_best:
                    best_path = onnx_path.parent / "best_model.onnx"
                    shutil.copy2(onnx_path, best_path)
                    global_best = window_best
                    print(f"  -> New best: {global_best:.1f}  (saved {best_path})", flush=True)
                window_best = float("-inf")

    except KeyboardInterrupt:
        print(f"\n[INTERRUPTED] {env_id} - terminating subprocess...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        raise

    proc.wait()
    ok = proc.returncode == 0
    tag = "DONE" if ok else f"FAILED (exit {proc.returncode})"
    print(f"\n[{tag}] {env_id}\n", flush=True)
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train all Marathon Envs builds sequentially with mlagents-learn (PPO)."
    )
    parser.add_argument(
        "--run-prefix",
        default=None,
        help="Prefix for run IDs (default: today's date, e.g. 20260519). "
             "Each run becomes <prefix>-<envId>.",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        metavar="ENV_ID",
        help="Train only these environments (default: all 16). "
             "When resuming, omitting --envs restores the original env list.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel Unity processes per environment. "
             "Overrides per-env value from config/optimal_spawn_envs.json. Default: per-env optimal or 1.",
    )
    parser.add_argument(
        "--num-spawn-envs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel environment instances inside each Unity process. "
             "Overrides per-env value from config/optimal_spawn_envs.json. Default: per-env optimal or 50.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pass --resume to mlagents-learn to continue an existing run.",
    )
    parser.add_argument(
        "--graphics",
        action="store_true",
        help="Enable graphics (by default --no-graphics is passed for headless training).",
    )
    args = parser.parse_args()

    # Resolve prefix and env list, restoring from state file when resuming
    saved_state: dict = {}
    if args.resume and STATE_FILE.exists():
        try:
            saved_state = json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    if args.run_prefix:
        prefix = args.run_prefix
    elif saved_state.get("prefix"):
        prefix = saved_state["prefix"]
        print(f"Resuming batch with prefix: {prefix}  (from {STATE_FILE})")
    else:
        prefix = datetime.date.today().strftime("%Y%m%d")

    if args.envs:
        envs = args.envs
    elif args.resume and saved_state.get("envs"):
        envs = saved_state["envs"]
        print(f"Resuming env list: {', '.join(envs)}")
    else:
        envs = ENVS

    unknown = [e for e in envs if e not in ENVS]
    if unknown:
        parser.error(f"Unknown environment IDs: {', '.join(unknown)}")

    # Always load optimal params; CLI flags override per-field
    optimal: dict = {}
    if OPTIMAL_FILE.exists():
        optimal = json.loads(OPTIMAL_FILE.read_text())
        print(f"Loaded per-environment optimal params from {OPTIMAL_FILE}")

    STATE_FILE.write_text(json.dumps({"prefix": prefix, "envs": envs}))
    print(f"Run prefix: {prefix}  (saved to {STATE_FILE})")
    print(f"To resume this batch:  python train_all_envs.py --resume\n")

    results: dict[str, bool | None] = {}
    try:
        for env_id in envs:
            entry = optimal.get(env_id, {})
            num_envs  = args.num_envs       if args.num_envs       is not None else entry.get("num_envs", 1)
            num_spawn = args.num_spawn_envs if args.num_spawn_envs is not None else entry.get("num_spawn_envs", 50)
            results[env_id] = _train_one(env_id, f"{prefix}-{env_id}", args, num_envs, num_spawn)
    except KeyboardInterrupt:
        remaining = [e for e in envs if e not in results]
        print(f"\n[INTERRUPTED] Batch cancelled. Remaining: {', '.join(remaining)}", flush=True)

    divider = "=" * 64
    done    = [e for e, r in results.items() if r is True]
    failed  = [e for e, r in results.items() if r is False]
    skipped = [e for e, r in results.items() if r is None]

    print(f"\n{divider}")
    print("TRAINING SUMMARY")
    print(divider)
    if done:
        print(f"  Completed ({len(done):2d}): {', '.join(done)}")
    if failed:
        print(f"  Failed    ({len(failed):2d}): {', '.join(failed)}")
    if skipped:
        print(f"  Skipped   ({len(skipped):2d}): {', '.join(skipped)}")
    print(divider)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
