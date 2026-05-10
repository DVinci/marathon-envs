# Training Guide

## Commands

### Editor training (no build needed)
```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_name>
```
Then press **Ctrl+P** in Unity to Play. Select environment in popup → GO.
Press **Ctrl+P** again to stop.

### Headless executable training
```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_name> --no-graphics --env="builds/<env>/Marathon Environments.exe" --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=<N>
```

### Resume training
```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_name> --resume --no-graphics --env="builds/<env>/Marathon Environments.exe" --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=<N>
```

### Multiple parallel Unity processes (better CPU utilization)
```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_name> --no-graphics --env="builds/<env>/Marathon Environments.exe" --num-envs=4 --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=5
```
`--num-envs` = number of Unity processes (one per CPU core)
`--num-spawn-envs` = environments per process

### Monitor training
```bash
tensorboard --logdir=summaries
```

---

## Environment Selection

`envIdDefault` in the scene's EnvSpawner Inspector controls which environment spawns when the trainer is connected (training mode skips the picker dialog by design).

`trainingNumEnvsDefault` controls how many parallel copies spawn. Use `1` for Editor play mode. Use higher values (4–20) for headless builds.

`--spawn-env` and `--num-spawn-envs` are **Unity-side arguments** passed via `--env-args`, not mlagents-learn flags.

---

## Performance

### Bottleneck analysis
- PhysX ragdoll simulation is **single-threaded** — the fundamental limit
- GPU (neural network updates) is NOT the bottleneck — it hits 100% waiting for Unity
- Adding more environments in one process doesn't help past the PhysX limit
- Use `--num-envs` to spawn multiple Unity processes across CPU cores

### Observed performance (RTX 2070 Super, 20 environments)
- ~60 steps/second regardless of Editor or headless build
- Walker2d-v0 (1M steps): ~4.5 hours
- Rendering overhead (Editor vs headless) is negligible vs physics cost

### To go faster
- More CPU cores → more `--num-envs` processes
- GPU-accelerated physics (Isaac Lab) → thousands of envs on GPU
- Better algorithms (SAC) → fewer steps needed for same quality

---

## CPU vs GPU Detection
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Building Headless Executable (Unity 6)

1. File → **Build Profiles** (Ctrl+Shift+B)
2. Select **Windows Server** in left panel
3. Click **Build** → save to `builds/<EnvName>/`

The `.exe` is launched automatically by mlagents-learn via `--env=`.

---

## Expected Rewards (Walker2d-v0)

| Steps | Expected Reward | Notes |
|---|---|---|
| 0–100k | 13–120 | Early stumbling |
| 100k–400k | 100–500 | Learning basic gait |
| 400k–1M | 300–700 | Refining locomotion |
| 1M+ | 700–1000+ | Competent walking |
| 5–10M | 1000–2000+ | Good locomotion |
| 10M+ | 2000–3000 | Strong performance |

Reward is noisy — look at the trend over 100k+ steps, not individual values.

---

## Resuming vs Starting Over

- `--resume` loads from `.pt` checkpoint files in `results/<run_id>/`
- `.onnx` files are export-only — cannot resume from them
- `.pt` → `.onnx` is one-way (optimizer state and training metadata are lost)
- Keep the `results/` folder to preserve resume capability

---

## Deploying Trained Models in Unity

1. Find the exported `.onnx` in `results/<run_id>/<BehaviorName>.onnx`
2. In Unity Inspector, select the agent prefab
3. **Behavior Parameters → Model** → assign the `.onnx`
4. **Behavior Parameters → Behavior Type** → set to `Inference Only`
5. Unity Sentis runs the model natively in C# (no Python needed)
