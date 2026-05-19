# Training Guide

## PPO Training (ML-Agents)

### Recommended command

Use `train_ppo.py` instead of `mlagents-learn` directly — it passes all arguments through unchanged and additionally saves `best_model.onnx` whenever a new peak reward is reached at a checkpoint boundary.

```bash
python train_ppo.py config/marathon_envs_config.yaml --run-id=<run_name> \
  --no-graphics --env="builds/<env>/Marathon Environments.exe" \
  --num-envs=4 --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=50
```

### Resume training

```bash
python train_ppo.py config/marathon_envs_config.yaml --run-id=<run_name> --resume \
  --no-graphics --env="builds/<env>/Marathon Environments.exe" \
  --num-envs=4 --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=50
```

### Editor training (no build needed)

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_name>
```

Then press **Ctrl+P** in Unity to Play. Select environment in popup → GO.
Press **Ctrl+P** again to stop. (`train_ppo.py` also works here.)

### Key flags

| Flag | Meaning |
| --- | --- |
| `--num-envs` | Number of Unity processes — one per CPU core |
| `--num-spawn-envs` | Environments per process — sweet spot is 50 on i7-3770S |
| `--no-graphics` | Headless mode; required for multi-process runs |
| `--resume` | Resume from `.pt` checkpoints in `results/<run_id>/` |

### Monitor training

```bash
tensorboard --logdir=summaries
```

Open `http://localhost:6006` in a browser. TensorBoard is written to `summaries/<run_id>/`.

---

## Environment Selection

`envIdDefault` in the scene's EnvSpawner Inspector controls which environment spawns when the trainer is connected (training mode skips the picker dialog by design).

`trainingNumEnvsDefault` controls how many parallel copies spawn. Use `1` for Editor play mode. Use higher values (4–20) for headless builds.

`--spawn-env` and `--num-spawn-envs` are **Unity-side arguments** passed via `--env-args`, not mlagents-learn flags.

---

## SAC Training (Stable Baselines 3)

SAC is an off-policy algorithm that reuses experience from a replay buffer, making it far more sample-efficient than PPO. It typically reaches higher final reward in fewer steps, at the cost of running with fewer parallel environments.

### Setup

```bash
cd marathon-envs
pip install -e .
pip install stable-baselines3
```

### Run

```bash
python stable_baselines_sac_train.py
```

Edit the constants at the top of the script to change environment, run ID, or step budget:

```python
ENV_NAME = "Walker2d-v0"
ENV_PATH = os.path.join("builds", "Walker2d-v0", "Marathon Environments.exe")
RUN_ID   = "sac_walker_01"
TOTAL_TIMESTEPS = 2_000_000
```

### Resume SAC training

Stop the script at any time (Ctrl+C). The next run resumes automatically:

```bash
python stable_baselines_sac_train.py
```

The script detects `results/<RUN_ID>/Walker2d-v0_checkpoint.zip` on startup and loads both the model weights and the replay buffer. Training continues from the saved step count; TensorBoard graphs stay continuous.

The replay buffer is what makes resume meaningful for SAC — without it, the agent would have to re-explore from scratch and lose its Q-value estimates. The first checkpoint (and thus first resumable point) is saved at `SAVE_FREQ` steps (default 100k).

### Output files

All saved to `results/<RUN_ID>/`:

| File | Contents |
| --- | --- |
| `best_model.zip` | Best policy by mean episode reward — updated in-place |
| `<EnvName>_checkpoint.zip` | Latest resumable checkpoint (model + optimizer state) |
| `<EnvName>_checkpoint_replay_buffer.pkl` | Replay buffer paired with the checkpoint |
| `<EnvName>_<step>.zip` | Periodic snapshot every `SAVE_FREQ` steps (model only) |
| `<EnvName>_final.zip` | Policy state at end of training |
| `monitor.csv` | Per-episode reward and length log |

TensorBoard logs go to `summaries/<RUN_ID>/`.

### Performance note

With a single environment, SAC's bottleneck is the Unity gRPC round-trip (~20–30ms per step), not the GPU. The script uses `N_ENVS=4` (four Unity processes via `SubprocVecEnv`) and `train_freq=4, gradient_steps=4`. This gives ~700 fps before gradient updates start, settling to ~200–400 fps once training is running — roughly 10–15× faster than a single-env run.

The mlagents conda environment includes CUDA-enabled PyTorch; SB3 SAC automatically uses `device=cuda`.

### Reading the training output

Each stats block has two sections:

**rollout/** — environment statistics:

| Field | Meaning |
| --- | --- |
| `ep_len_mean` | Average episode length in steps. Grows as the agent learns to stay upright. |
| `ep_rew_mean` | Average total reward per episode. The main metric to watch. |
| `episodes` | Total episodes completed across all envs since start. |
| `fps` | Environment steps per second. Drops when gradient updates start. |
| `time_elapsed` | Seconds since training started. |
| `total_timesteps` | Total steps collected across all envs. |

**train/** — neural network statistics:

| Field | Meaning |
| --- | --- |
| `actor_loss` | Policy network loss. Negative is normal — actor finds actions the critic scores highly. |
| `critic_loss` | Q-value prediction error. Should decrease over time. |
| `ent_coef` | Entropy coefficient — controls exploration. Auto-tuned from ~1.0 down to ~0.001. High = exploring, low = exploiting. |
| `ent_coef_loss` | Signal driving `ent_coef` adjustment. Negative = policy is less random than target entropy. |
| `learning_rate` | Fixed at 3e-4 (no decay schedule, unlike PPO). |
| `n_updates` | Gradient steps taken so far. |

### PPO vs SAC comparison

| | PPO (ML-Agents) | SAC (Stable Baselines 3) |
| --- | --- | --- |
| Algorithm type | On-policy | Off-policy (replay buffer) |
| Parallel envs | 200 (4 × 50) | 4 (1 per process) |
| Steps/second | ~1300 | ~200–700 |
| Steps to reward ~800 | ~4M | ~100k–200k |
| Final policy quality | ~828 at 5M steps | expected 1500–3000+ at 1M steps |
| Resume support | Yes (`.pt` checkpoints) | Yes (`.zip` + replay buffer) |

PPO parallelizes well across CPU cores. SAC needs far fewer steps to reach a good policy and runs on GPU for gradient computation.

---

## Performance

### Bottleneck

PhysX ragdoll simulation is **single-threaded per Unity process** — the hard limit. The GPU is not the bottleneck; it sits idle waiting for Unity to deliver observations.

**Critical setting:** without `engine_settings.time_scale`, Unity runs at 1× real-time (~60 steps/s regardless of environment count). Always set:

```yaml
engine_settings:
  time_scale: 20
  target_frame_rate: -1
  quality_level: 0
```

This is already set in `config/marathon_envs_config.yaml`.

### Observed performance (i7-3770S, RTX 2070 Super)

| Config | Throughput | Time to 1M steps |
| --- | --- | --- |
| 1 process, 20 envs, no time_scale | 61 steps/s | 4.6 h |
| 4 processes, 20 envs/proc, time_scale=20 | 1192 steps/s | 14 min |
| 4 processes, 50 envs/proc, time_scale=20 | **1506 steps/s** | **11 min** |
| 4 processes, 100 envs/proc, time_scale=20 | 1435 steps/s | 11.6 min |

**50 envs/process is the confirmed sweet spot** for a 4-core CPU. Beyond that, per-core PhysX overhead exceeds the benefit of more data.

### To go faster

- More CPU cores → increase `--num-envs` (one process per core)
- GPU-accelerated physics (Isaac Lab) → thousands of envs on GPU
- Switch to SAC → fewer total steps needed for same or better quality

---

## CPU vs GPU Detection

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

---

## Building Executables

### Single build (manual)

1. File → **Build Profiles** (Ctrl+Shift+B)
2. Select **Windows Server** in left panel
3. Click **Build** → save to `builds/<EnvName>/`

The `.exe` is launched automatically by mlagents-learn via `--env=`.

### Batch build — all 16 environments

The Unity Editor menu **Marathon Envs → Build All Environments (Windows)** builds one executable per environment automatically. Each build has `envIdDefault` pre-set so training can start without passing `--spawn-env`.

**Menu:** `Marathon Envs → Build All Environments (Windows)`

Output: `builds/<envId>/Marathon Environments.exe` × 16

A progress bar tracks completion. The scene's `envIdDefault` is restored to its original value after all builds finish. There is also a **Build Single Environment (Windows)** menu item for rebuilding just one environment after a change.

Source: [UnitySDK/Assets/Editor/BatchBuild.cs](../UnitySDK/Assets/Editor/BatchBuild.cs)

---

## Batch Training — All Environments

`train_all_envs.py` trains every pre-built environment in sequence. Each environment gets its own `mlagents-learn` run with a dated run-ID, and best-model tracking (same as `train_ppo.py`) is applied automatically.

### Basic usage

```bash
# Train all 16 environments with defaults (headless, 1 Unity process, 1 env per process)
python train_all_envs.py

# Recommended: 4 parallel instances per build (good for 4-core CPU)
python train_all_envs.py --num-spawn-envs 4

# Train only a subset
python train_all_envs.py --envs Hopper-v0 Walker2d-v0 Ant-v0

# Resume a batch that was interrupted (prefix is read from train_all_envs.last_prefix automatically)
python train_all_envs.py --resume
```

### Run IDs

Each environment gets the run-ID `<prefix>-<envId>`. The prefix defaults to today's date (`YYYYMMDD`), e.g. `20260519-Walker2d-v0`. Summaries and checkpoints land in the usual `summaries/` and `results/` folders.

### Flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `--run-prefix PREFIX` | today's date | Prefix for all run-IDs in this batch |
| `--envs ENV …` | all 16 | Train only the listed environments |
| `--num-envs N` | 1 | Parallel Unity processes per build |
| `--num-spawn-envs N` | 1 | Parallel env instances inside each process |
| `--resume` | off | Pass `--resume` to mlagents-learn (continue existing runs) |
| `--graphics` | off | Enable graphics (headless by default) |

### Build prerequisite

`train_all_envs.py` skips any environment whose build folder doesn't exist yet — it will print `[SKIP] <envId> — build not found`. Run **Marathon Envs → Build All Environments (Windows)** in the Unity Editor first to generate all builds.

### Output

At the end of the batch, a summary is printed:

```text
================================================================
TRAINING SUMMARY
================================================================
  Completed (14): Hopper-v0, Walker2d-v0, ...
  Skipped   ( 2): MarathonManJazzDancing-v0, MarathonManBackflip-v0
================================================================
```

Exits with code `1` if any run failed, `0` otherwise — safe to use in scripts.

---

## Expected Rewards (Walker2d-v0)

Values are from actual training runs on this machine. `normalize: true` raises the ceiling at every stage.

| Steps | normalize: false | normalize: true | Notes |
| --- | --- | --- | --- |
| 0–500k | 13–458 | 13–458 | Early stumbling; no difference yet |
| 500k–1M | 300–700 | 400–740 | normalize advantage becomes visible |
| 1M–2M | 600–680 | 740–782 | normalize: true ~+100 reward |
| 2M–3M | 680–731 | 782–809 | ceiling rising, gap ~70–80 |
| 3M–5M | 731–762 | 809–828 | normalize: true plateau is ~828 at 5M |
| 5–10M | ~1000–2000 | expected | PPO ceiling; switch to SAC for 3000+ |

**Observed ceilings on this machine (i7-3770S, 4×50 envs, 5M steps):**

- `normalize: false`: 762 (walker_03) — bimodal instability, dips to 515
- `normalize: true`: 828 (walker_04) — stable, std=1–7 throughout

Reward is noisy — look at the trend over 100k+ steps, not individual values.

---

## Evaluating Training Success

There is no fixed reward ceiling — success is judged by multiple signals together.

### Reward curve shape

A healthy run shows a clear upward trend followed by a plateau. The plateau means the policy has converged near its local optimum. A curve that never rises, or rises and then collapses, indicates a problem (bad hyperparameters, environment bug, or reward shaping issue).

### Reward plateau + low standard deviation

When mean reward stabilizes **and** standard deviation drops, the agent is doing the right thing consistently — not just getting lucky in some episodes. High mean + high std = sometimes lucky. High mean + low std = reliable policy.

### Visual inspection

Watch the agent in Unity play mode. This is the real test. A smooth, stable gait at reward 800 is better than wild lurching at reward 1200. Numbers don't capture style, energy efficiency, or fall recovery.

### Entropy decay (TensorBoard)

`Policy/Entropy` should decrease over training as the policy commits to specific actions. If entropy stays high throughout, the agent is still exploring randomly. If it collapses to near zero very early, exploration died and the policy may be stuck.

### Perturbation test

Push the agent in play mode. A well-trained walker recovers from nudges. A fragile policy (overfit to lucky episodes) falls immediately.

### Reference benchmarks

Walker2d-v0 (MuJoCo reference, equivalent task):

- PPO: ~1500–3000 at convergence
- SAC: ~3000–5000 at convergence

This project's Walker2d-v0 is a faithful port — those are the rough targets for a fully trained policy.

### Quick decision guide

| Reward | Status | Action |
| --- | --- | --- |
| < 500 | Still learning | Keep training |
| 500–1000 | Basic locomotion | Evaluate visually |
| 1000–2000 | Competent | Compare to reference video |
| 2000+ | Strong | Visual + perturbation test |
| Plateau 500k+ steps, low std | Converged | Done (or tune and retrain) |

---

## Can Training for Too Long Be Harmful?

Yes — in several distinct ways.

### Entropy collapse → local optimum lock-in

As training continues, entropy (exploration) naturally drops. Once the policy commits hard to one strategy, it stops exploring alternatives. If that strategy is a local optimum (e.g., shuffle forward with tiny steps), more training makes the policy *more* committed to the wrong behavior. No amount of additional steps escapes a collapsed local optimum.

### Reward hacking gets reinforced

The agent finds ways to score reward that don't match the intended behavior — a walker that leans forward and barely catches itself, or spins in place because the reward formula has a gap. Early in training this is fleeting; after 10M+ steps the policy has deeply optimized for the exploit and it is very hard to unlearn.

### Policy gradient instability (rare with PPO)

PPO's clipping prevents catastrophic updates, but over a very long run with too high a learning rate, the policy can drift and degrade. Reward may climb to 1500 then slowly erode back to 800. `learning_rate_schedule: linear` decays the LR toward zero over `max_steps` specifically to prevent this — it is one reason the schedule matters.

### Style transfer environments are most fragile

Environments using motion capture reference (Walking, Running, Dancing, etc.) have complex reward shaping. Running past the plateau can degrade style quality even while raw reward stays flat, because the policy drifts toward exploiting the reward formula rather than matching the reference pose.

### The practical answer for this project

With PPO + linear LR decay, "too long" usually means **wasted compute, not broken training**. A run that plateaued at 15M and continues to 20M hasn't been harmed — it just burned CPU for nothing. Active degradation is uncommon.

**Rule of thumb:** if the reward curve has been flat for 20–30% of `max_steps`, stop and export. You are done.

---

## Checkpoint Selection

Not every checkpoint is equal. The best checkpoint to deploy is the one with the **highest mean reward and lowest standard deviation** — meaning the policy performs well *consistently*, not just on lucky episodes.

### What the metrics mean

- **High mean + low std**: agent completes most episodes without falling; reliable for deployment
- **High mean + high std**: sometimes great, sometimes falls early — noisy policy
- **Low std alone**: consistent but potentially consistently mediocre — check the mean too

When they conflict (e.g., checkpoint A: mean=730 std=150 vs checkpoint B: mean=700 std=10), prefer B for inference. Prefer A only if you intend to keep training.

### best_model.onnx / best_model.zip

Both training scripts track this automatically:

- **`train_ppo.py`** saves `results/<run_id>/<BehaviorName>/best_model.onnx` — updated whenever mean reward improves at a checkpoint boundary
- **`stable_baselines_sac_train.py`** saves `results/<run_id>/best_model.zip` — updated whenever mean episode reward improves during training

The "best" here is peak mean reward in a checkpoint window, not peak at the exact summary moment. The policy at the checkpoint boundary is the closest available snapshot.

### Checkpoint interval and file sizes

ML-Agents checkpoints are tiny because the network is small (64 units × 2 layers):

| File | Size |
| --- | --- |
| `.pt` (resume checkpoint) | ~189 KB |
| `.onnx` (inference model) | ~34 KB |

To capture finer-grained best moments, lower `checkpoint_interval` in the config. The default is 500,000 steps. Setting it to 50,000 costs ~19 MB over a 5M-step run with `keep_checkpoints: 5` (only 5 `.pt` files kept at any time).

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
| --- | --- |
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
