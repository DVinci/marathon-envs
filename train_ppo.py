#!/usr/bin/env python3
"""
Wrapper around mlagents-learn that keeps a best_model.onnx alongside
the periodic checkpoints. Run exactly like mlagents-learn:

  python train_ppo.py config/marathon_envs_config.yaml --run-id=walker_04 \
      --no-graphics --env="builds/Walker2d-v0/Marathon Environments.exe" \
      --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

_REWARD_RE = re.compile(r"Mean Reward:\s+([-\d]+(?:\.\d+)?)")
_EXPORT_RE = re.compile(r"Exported\s+(.+\.onnx)")


def main() -> int:
    proc = subprocess.Popen(
        ["mlagents-learn"] + sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    global_best = float("-inf")
    window_best = float("-inf")  # max reward seen since last checkpoint export

    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        m = _REWARD_RE.search(line)
        if m:
            reward = float(m.group(1))
            if reward > window_best:
                window_best = reward

        m = _EXPORT_RE.search(line)
        if m:
            onnx_path = Path(m.group(1).strip())
            if window_best > global_best:
                best_path = onnx_path.parent / "best_model.onnx"
                shutil.copy2(onnx_path, best_path)
                global_best = window_best
                print(f"  → New best: {global_best:.1f}  (saved {best_path})", flush=True)
            window_best = float("-inf")  # reset for next checkpoint window

    proc.wait()
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
