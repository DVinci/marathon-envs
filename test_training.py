#!/usr/bin/env python3
"""Smoke-tests the batch training infrastructure before committing to full runs.

Phase 1 — Connection & output (all 16 envs):
  Runs each build with the minimal test config (~15k steps, 3 checkpoints).
  Verifies: exit code 0, at least one .onnx checkpoint, best_model.onnx, TensorBoard events.

Phase 2 — Resume (Walker2d-v0 only):
  Runs to 5k steps, then resumes to 10k steps.
  Verifies the step counter continued from ~5k rather than restarting from 0.

Usage:
    python test_training.py
    python test_training.py --num-spawn-envs 4   # more parallel envs per process
    python test_training.py --phase1-only        # skip resume test
    python test_training.py --phase2-only        # skip the 16-env sweep
"""

import argparse
import datetime
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
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

TEST_CONFIG  = "config/marathon_envs_test_config.yaml"
BUILDS_DIR   = Path("builds")
EXE_NAME     = "Marathon Environments.exe"
RESULTS_DIR  = Path("results")

_REWARD_RE = re.compile(r"Mean Reward:\s+([-\d]+(?:\.\d+)?)")
_EXPORT_RE = re.compile(r"Exported\s+(.+\.onnx)")
_STEP_RE   = re.compile(r"Step:\s+(\d+)\.")
_RESUME_RE = re.compile(r"[Rr]esuming")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def _run(env_id: str, run_id: str, config: str, num_envs: int, num_spawn_envs: int,
         resume: bool = False) -> tuple[int, list[str]]:
    exe = BUILDS_DIR / env_id / EXE_NAME
    if not exe.exists():
        print(f"  [SKIP] {env_id} — build not found: {exe}")
        return -1, []

    cmd = [
        "mlagents-learn", config,
        f"--run-id={run_id}",
        f"--env={exe}",
        "--no-graphics",
        f"--num-envs={num_envs}",
    ]
    if resume:
        cmd.append("--resume")
    cmd += ["--env-args", f"--spawn-env={env_id}", f"--num-spawn-envs={num_spawn_envs}"]

    divider = "-" * 56
    print(f"\n{divider}")
    print(f"  {env_id}  |  run-id: {run_id}{'  [RESUME]' if resume else ''}")
    print(f"{divider}")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )

    global_best = None  # type: float | None  — None until first checkpoint saved
    window_best = float("-inf")
    lines: list[str] = []

    for line in proc.stdout:
        sys.stdout.buffer.write(line.encode(sys.stdout.encoding or "utf-8", errors="replace"))
        sys.stdout.flush()
        lines.append(line)

        m = _REWARD_RE.search(line)
        if m:
            reward = float(m.group(1))
            if reward > window_best:
                window_best = reward

        m = _EXPORT_RE.search(line)
        if m:
            onnx_path = Path(m.group(1).strip())
            if global_best is None or window_best > global_best:
                best_path = onnx_path.parent / f"{run_id}_best.onnx"
                shutil.copy2(onnx_path, best_path)
                global_best = window_best
                print(f"  -> New best: {global_best:.1f}  (saved {best_path})", flush=True)
            window_best = float("-inf")

    proc.wait()
    return proc.returncode, lines


# ---------------------------------------------------------------------------
# Output checkers
# ---------------------------------------------------------------------------

def _check_run_outputs(run_id: str, env_id: str) -> list[tuple[str, bool, str]]:
    run_dir = RESULTS_DIR / run_id

    # Style-transfer envs report BehaviorName="My Behavior" at runtime, so their
    # model subdir is "My Behavior" rather than the env_id.
    env_dir = run_dir / env_id
    if not env_dir.exists() and run_dir.exists():
        subdirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if subdirs:
            env_dir = subdirs[0]

    checks = []

    onnx_checkpoints = [
        f for f in (env_dir.glob("*.onnx") if env_dir.exists() else [])
        if f.name != "best_model.onnx"
    ]
    checks.append((
        "checkpoint .onnx saved",
        len(onnx_checkpoints) >= 1,
        f"{len(onnx_checkpoints)} checkpoint(s)",
    ))

    best_files = list(env_dir.glob("*_best.onnx")) if env_dir.exists() else []
    checks.append((
        "best model saved",
        len(best_files) >= 1,
        best_files[0].name if best_files else "MISSING",
    ))

    # ml-agents writes tfevents into results/<run_id>/<behavior>/, not a separate summaries dir
    tb_files = list(env_dir.glob("events.out.tfevents.*")) if env_dir.exists() else []
    checks.append((
        "TensorBoard events written",
        len(tb_files) >= 1,
        f"{len(tb_files)} file(s)",
    ))

    return checks


def _check_resume(run_id_p1: str, env_id: str, resume_lines: list[str]) -> list[tuple[str, bool, str]]:
    checks = []

    env_dir = RESULTS_DIR / run_id_p1 / env_id
    onnx_checkpoints = list(env_dir.glob(f"{env_id}-*.onnx")) if env_dir.exists() else []
    checks.append((
        "Phase 1 checkpoint saved",
        len(onnx_checkpoints) >= 1,
        f"{len(onnx_checkpoints)} checkpoint(s)",
    ))

    resumed_msg = any(_RESUME_RE.search(l) for l in resume_lines)
    step_matches = [_STEP_RE.search(l) for l in resume_lines if _STEP_RE.search(l)]
    first_step = int(step_matches[0].group(1)) if step_matches else 0
    resumed_from_checkpoint = resumed_msg and first_step >= 4000

    checks.append((
        "Phase 2 resumed from checkpoint (not step 0)",
        resumed_from_checkpoint,
        f"first Step in log: {first_step},  'Resuming' in log: {resumed_msg}",
    ))

    return checks


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------

def phase1(prefix: str, num_envs: int, num_spawn_envs: int) -> dict[str, list[tuple[str, bool, str]]]:
    results = {}
    for env_id in ENVS:
        run_id = f"{prefix}-{env_id}"
        rc, _ = _run(env_id, run_id, TEST_CONFIG, num_envs, num_spawn_envs)
        if rc == -1:
            results[env_id] = [("build exists", False, "exe not found — skipped")]
            continue
        if rc != 0:
            results[env_id] = [("mlagents-learn exit code", False, f"exit {rc}")]
            continue
        results[env_id] = _check_run_outputs(run_id, env_id)
    return results


# ---------------------------------------------------------------------------
# Phase 2 — resume test
# ---------------------------------------------------------------------------

_RESUME_CONFIG_TEMPLATE = textwrap.dedent("""\
    engine_settings:
      time_scale: 20
      target_frame_rate: -1
      quality_level: 0

    behaviors:
      Walker2d-v0:
        trainer_type: ppo
        hyperparameters:
          batch_size: 64
          buffer_size: 512
          learning_rate: 1.0e-3
          beta: 3.0e-3
          epsilon: 0.2
          lambd: 0.95
          num_epoch: 3
          learning_rate_schedule: linear
        network_settings:
          normalize: true
          hidden_units: 64
          num_layers: 2
          vis_encode_type: simple
        reward_signals:
          extrinsic:
            gamma: 0.99
            strength: 1.0
        max_steps: {max_steps}
        time_horizon: 100
        summary_freq: 5000
        checkpoint_interval: 5000
        keep_checkpoints: 10
""")


def phase2_resume(prefix: str, num_envs: int, num_spawn_envs: int) -> list[tuple[str, bool, str]]:
    env_id   = "Walker2d-v0"
    run_id   = f"{prefix}-resume-walker"
    tmp_dir  = Path(tempfile.mkdtemp(prefix="marathon_resume_test_"))

    try:
        # Phase 2a: run to 5000 steps
        cfg_p1 = tmp_dir / "resume_phase1.yaml"
        cfg_p1.write_text(_RESUME_CONFIG_TEMPLATE.format(max_steps=5000))

        print("\n=== Phase 2a: initial run to 5k steps ===")
        rc, _ = _run(env_id, run_id, str(cfg_p1), num_envs, num_spawn_envs, resume=False)
        p1_checks = _check_resume(run_id, env_id, [])

        if not p1_checks[0][1]:
            return p1_checks + [("Phase 2 resume", False, "skipped — Phase 1 checkpoint missing")]

        # Phase 2b: resume to 10000 steps
        cfg_p2 = tmp_dir / "resume_phase2.yaml"
        cfg_p2.write_text(_RESUME_CONFIG_TEMPLATE.format(max_steps=10000))

        print("\n=== Phase 2b: resume to 10k steps ===")
        rc, resume_lines = _run(env_id, run_id, str(cfg_p2), num_envs, num_spawn_envs, resume=True)

        return _check_resume(run_id, env_id, resume_lines)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _print_report(phase1_results: dict, phase2_results: list | None) -> int:
    divider = "=" * 60
    thin    = "-" * 60
    passed  = 0
    failed  = 0

    print(f"\n{divider}")
    print("TEST RESULTS")
    print(divider)

    if phase1_results is not None:
        print("\nPhase 1 — Connection & output (all 16 envs)")
        print(thin)
        for env_id, checks in phase1_results.items():
            env_ok = all(ok for _, ok, _ in checks)
            tag = "PASS" if env_ok else "FAIL"
            detail = "  |  ".join(f"{name}: {det}" for name, ok, det in checks if not ok) or ""
            suffix = f"  <- {detail}" if detail else ""
            print(f"  [{tag}]  {env_id:<32}{suffix}")
            if env_ok:
                passed += 1
            else:
                failed += 1

    if phase2_results is not None:
        print(f"\nPhase 2 — Resume (Walker2d-v0)")
        print(thin)
        for name, ok, detail in phase2_results:
            tag = "PASS" if ok else "FAIL"
            print(f"  [{tag}]  {name:<40}  {detail}")
            if ok:
                passed += 1
            else:
                failed += 1

    total = passed + failed
    print(f"\n{divider}")
    print(f"  {passed}/{total} passed")
    print(divider)

    if failed == 0:
        print("\n  All checks passed. Safe to start full training:")
        print("    python train_all_envs.py --num-spawn-envs 50\n")
    else:
        print(f"\n  {failed} check(s) failed — investigate before starting full training.\n")

    print("  Test artifacts left in results/ under run-prefix 'test_*'.")
    print("  Delete when no longer needed:\n")
    print("    Remove-Item -Recurse results\\test_*\n")

    return 0 if failed == 0 else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Marathon Envs batch training infrastructure."
    )
    parser.add_argument("--num-envs",       type=int, default=1,  metavar="N")
    parser.add_argument("--num-spawn-envs", type=int, default=4,  metavar="N")
    parser.add_argument("--phase1-only",    action="store_true")
    parser.add_argument("--phase2-only",    action="store_true")
    args = parser.parse_args()

    prefix = "test_" + datetime.date.today().strftime("%Y%m%d")

    phase1_results = None
    phase2_results = None

    if not args.phase2_only:
        print(f"\n{'='*60}")
        print(f"Phase 1 — running all 16 environments  (prefix: {prefix})")
        print(f"{'='*60}")
        phase1_results = phase1(prefix, args.num_envs, args.num_spawn_envs)

    if not args.phase1_only:
        print(f"\n{'='*60}")
        print(f"Phase 2 — resume test  (Walker2d-v0)")
        print(f"{'='*60}")
        phase2_results = phase2_resume(prefix, args.num_envs, args.num_spawn_envs)

    return _print_report(phase1_results, phase2_results)


if __name__ == "__main__":
    sys.exit(main())
