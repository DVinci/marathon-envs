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

## Hyperparameter Configuration

All hyperparameters live in `config/marathon_envs_config.yaml`, one block per behavior name (e.g. `Walker2d-v0`). There are no global defaults — each environment declares all its values explicitly.

### Parameter reference

| Parameter | What it controls |
|---|---|
| `batch_size` | Samples per gradient update — larger = more stable, slower |
| `buffer_size` | Experiences collected before any update — should be >> `batch_size` |
| `learning_rate` | Step size — start at `3e-4`; lower to `1e-4` for style transfer |
| `beta` | Entropy bonus — keeps exploration alive; set near zero (`1e-5`) once style is found |
| `epsilon` | PPO clip ratio — limits how much one update changes the policy; keep at `0.2` |
| `lambd` | GAE lambda — bias-variance tradeoff for advantage estimation; `0.95` is standard |
| `num_epoch` | Passes over the buffer per update — lower (3) for complex envs, higher (10) for simple |
| `normalize` | Online observation normalization — enable for complex bodies (MarathonMan, terrain) |
| `hidden_units` | Neurons per layer — `64` for classical, `256` for style transfer |
| `gamma` | Discount factor — raise to `0.9999` for sparse reward (agent must plan far ahead) |
| `time_horizon` | Steps before forced episode cut — raise to `1000` for terrain (longer episodes needed) |

### Profiles by environment group

| Group | `batch_size` | `lr` | `hidden_units` | `max_steps` | Notable settings |
| --- | --- | --- | --- | --- | --- |
| Classical (Hopper, Walker, Ant) | 16–32 | 1e-3 | 64 | 1M | Defaults; fast convergence |
| MarathonMan-v0 | 64 | 3e-4 | 64 | 10M | `normalize: true` |
| Style transfer (Walking/Running/Dancing) | 768 | 1e-4 | 256 | 64–128M | `beta: 1e-5`, `num_epoch: 3`; very long runs |
| Terrain variants | 32 | 3e-4 | 64 | 50M | `beta: 0.1` (high exploration), `time_horizon: 1000` |

**Why terrain uses `beta: 0.1`:** agents need to explore more aggressively to discover footing on uneven surfaces. High entropy keeps them trying varied strategies.

**Why style transfer uses `beta: 1e-5`:** once the policy finds the reference pose, you want it to commit and refine rather than keep experimenting. Near-zero entropy is intentional.

### Starting point for a new environment

Copy the `Walker2d-v0` block (ML-Agents defaults). Increase `max_steps` as needed. Add `normalize: true` if the observation space is large or has mixed-scale values.

### Network size: bigger is not better in RL

Larger networks do not automatically produce better policies for the same task. This is a key difference from supervised deep learning.

**Why bigger hurts in RL:**

- Sample efficiency drops — more parameters need more gradient steps; PPO's trust-region constraint limits how much the policy can change per update, so a large network may be undertrained at the same step budget
- RL gradients are noisy; small networks are easier to optimize under that noise
- A 64-unit MLP cannot memorize noise — it's forced to learn general locomotion structure; a 512-unit network can fit noise and produce a fragile policy

**When larger networks do help:**

- Style transfer uses 256 units because matching a motion capture pose requires learning complex correlations across many joints simultaneously
- Visual/pixel observations (CNNs need more capacity)
- Long-horizon memory (larger LSTM hidden state)

**Rule of thumb:** start at 64 units / 2 layers. Only scale up if the reward curve plateaus well below the expected ceiling. The bottleneck in this project is almost always sample count (physics throughput), not model capacity.

---

## Deploying Trained Models in Unity

1. Find the exported `.onnx` in `results/<run_id>/<BehaviorName>.onnx`
2. In Unity Inspector, select the agent prefab
3. **Behavior Parameters → Model** → assign the `.onnx`
4. **Behavior Parameters → Behavior Type** → set to `Inference Only`
5. Unity Sentis runs the model natively in C# (no Python needed)
