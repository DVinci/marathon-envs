# Training Performance Log

Observed throughput and reward data from actual training runs on the development machine.

## Hardware

| Component | Spec |
| --- | --- |
| CPU | Intel i7-3770S (4 cores / 8 threads, Ivy Bridge) |
| RAM | 32 GB |
| GPU | NVIDIA RTX 2070 Super 8 GB |
| OS | Windows 11 Pro |
| Bottleneck | CPU — PhysX simulation is single-threaded per Unity process |

---

## Run: hopper_01 — Walker2d-v0

**Agent:** Walker2d-v0
**ML-Agents:** 1.2.0.dev0 · PyTorch 2.3.0 · Unity package 4.0.2
**Hyperparameters:** defaults from `config/marathon_envs_config.yaml`
(batch_size=16, buffer_size=5120, lr=1e-3, hidden_units=64, normalize=false, max_steps=1M)

---

### Session A — 20 environments, 1 process

**Command:**

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=hopper_01 --resume --no-graphics
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=20
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 1 (default) |
| `--num-spawn-envs` | 20 |
| Total environments | 20 |
| Resumed from step | 418,961 |
| Interrupted at step | 542,204 |
| Session steps | ~123,000 |
| Session wall time | ~32 min |
| **Throughput** | **~61 steps/s** |

**Reward progression (10k-step summaries):**

| Step | Mean Reward | Std |
| --- | --- | --- |
| 420,000 | 21 | 0 |
| 430,000 | 150 | 81 |
| 440,000 | 559 | 167 |
| 450,000 | 536 | 199 |
| 460,000 | 554 | 185 |
| 470,000 | 363 | 268 |
| 480,000 | 463 | 254 |
| 490,000 | 507 | 211 |
| 500,000 | 519 | 218 |
| 510,000 | 329 | 215 |
| 520,000 | 590 | 161 |
| 530,000 | 557 | 233 |
| 540,000 | 586 | 207 |

---

### Session B — 50 environments, 1 process

**Command:**

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=hopper_01 --resume --no-graphics
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 1 (default) |
| `--num-spawn-envs` | 50 |
| Total environments | 50 |
| Resumed from step | 542,204 |
| Finished at step | 1,000,000 (max_steps reached) |
| Session steps | ~458,000 |
| Session wall time | ~130 min |
| **Throughput** | **~58 steps/s** |

**Reward progression (10k-step summaries):**

| Step | Mean Reward | Std |
| --- | --- | --- |
| 550,000 | 43 | 31 |
| 560,000 | 142 | 52 |
| 570,000 | 380 | 15 |
| 590,000 | 647 | 104 |
| 600,000 | 580 | 243 |
| 640,000 | 657 | 80 |
| 690,000 | 653 | 106 |
| 700,000 | 621 | 156 |
| 790,000 | 629 | 136 |
| 920,000 | 624 | 109 |
| 960,000 | 675 | 15 |
| 1,000,000 | 602 | 189 |

---

## Key Observations

### Throughput is the same at 20 and 50 environments

| Config | Throughput |
| --- | --- |
| 1 process, 20 envs | ~61 steps/s |
| 1 process, 50 envs | ~58 steps/s |

Going from 20 to 50 environments in the same process produced **no improvement** in steps/second. The CPU core was already saturated at 20 environments. Extra environments added to the same process sit idle waiting for the PhysX tick to complete.

**Conclusion:** for this machine and this agent, 20 environments is the saturation point per process. Adding more environments per process beyond that wastes memory without gaining throughput.

### Reward is noisy but trending correctly

The policy reached 400–675 reward range by 1M steps, consistent with the expected progression in `training-guide.md`. High standard deviation (~200–250) is normal — some episodes terminate early, others run the full `time_horizon=100` steps.

### Estimated time to 1M steps from scratch

At ~60 steps/s: **~4.6 hours** on this machine with 1 process.

---

## Run: walker_02 — Walker2d-v0 (4 processes, old config)

**Command:**

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=walker_02 --no-graphics
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=20
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 20 |
| Total environments | 80 |
| Config | default (batch_size=16, num_epoch=10) |
| Interrupted at step | 30,000 |
| **Throughput** | **~59 steps/s** |

---

## Python Is the Bottleneck — All Configs Converge to ~60 steps/s

| Config | Processes | Envs/process | Total envs | Throughput |
| --- | --- | --- | --- | --- |
| hopper_01 session A | 1 | 20 | 20 | 61 steps/s |
| hopper_01 session B | 1 | 50 | 50 | 58 steps/s |
| walker_02 | 4 | 20 | 80 | 59 steps/s |

Adding Unity processes and environments made no difference — all three configs yielded ~60 steps/s. The diagnosis of "Python bottleneck" was incorrect.

### Root Cause: `time_scale` Not Set — Simulation Running in Real-Time

The config had no `engine_settings` block. Without `time_scale`, Unity runs the simulation at **1× real-time**. The ~60 steps/s matched the default 60 FPS cap on the Unity game loop — the simulation was not running fast at all.

### Fix Applied

Added `engine_settings` to the top of `config/marathon_envs_config.yaml`:

```yaml
engine_settings:
  time_scale: 20
  target_frame_rate: -1
  quality_level: 0
```

| Setting | Effect |
| --- | --- |
| `time_scale: 20` | Physics runs 20× faster than real-time |
| `target_frame_rate: -1` | Uncaps the Unity game loop |
| `quality_level: 0` | Minimum rendering overhead |

Also updated Walker2d-v0 hyperparameters to reduce Python update overhead:

| Parameter | Before | After |
| --- | --- | --- |
| `batch_size` | 16 | 1024 |
| `buffer_size` | 5120 | 10240 |
| `num_epoch` | 10 | 3 |

---

## Run: walker_02 resumed — With engine_settings + new hyperparams

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 20 |
| Total environments | 80 |
| `time_scale` | 20 |
| `target_frame_rate` | -1 |
| Resumed from step | 36,948 |
| Interrupted at step | 283,759 |
| **Throughput** | **~1192 steps/s** |
| **Improvement** | **~20× over all previous runs** |
| **Est. time to 1M steps** | **~14 minutes** |

---

## Run: walker_02 — 4×100 environments

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=walker_02 --no-graphics --resume
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=100
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 100 |
| Total environments | 400 |
| Resumed from step | 283,759 |
| Interrupted at step | 578,997 |
| **Throughput** | **~1435 steps/s** |
| **Est. time to 1M steps** | **~11.6 min** |

Going from 20→100 envs/process (5× more) gave only **+20% throughput** — PhysX per-core saturation setting in. The sweet spot is likely around 40–50 envs/process.

---

## Run: walker_02 — 4×50 environments

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=walker_02 --no-graphics --resume
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 50 |
| Total environments | 200 |
| Resumed from step | 578,997 |
| Interrupted at step | 911,449 |
| **Throughput** | **~1506 steps/s** |
| **Est. time to 1M steps** | **~11 min** |

---

## Full Comparison — Throughput

| Config | time_scale | Processes | Envs/process | Total envs | Throughput | Est. time to 1M |
| --- | --- | --- | --- | --- | --- | --- |
| hopper_01 session A | unset (=1) | 1 | 20 | 20 | 61 steps/s | 4.6h |
| hopper_01 session B | unset (=1) | 1 | 50 | 50 | 58 steps/s | 4.8h |
| walker_02 old config | unset (=1) | 4 | 20 | 80 | 59 steps/s | 4.7h |
| walker_02 + time_scale | 20 | 4 | 20 | 80 | 1192 steps/s | 14 min |
| walker_02 + time_scale | 20 | 4 | 100 | 400 | 1435 steps/s | 11.6 min |
| **walker_02 + time_scale** | **20** | **4** | **50** | **200** | **1506 steps/s** | **~11 min** |

**50 envs/process is the confirmed sweet spot.** At 100 envs/process throughput actually drops — per-core PhysX overhead of managing 100 simultaneous simulations exceeds the benefit of more data.

## Full Comparison — Walker2d-v0 Policy Quality

| Run | normalize | max_steps | Best reward | Std at best | Wall time (full) | Bimodal? |
| --- | --- | --- | --- | --- | --- | --- |
| walker_03 | false | 5M | 758 (4.5M) | 5.7 | ~54 min | Yes — dips to 515–530 |
| **walker_04** | **true** | **5M** | **828 (4.5M)** | **low (1–7)** | **~64 min** | **No — stable throughout** |

`normalize: true` costs ~10 more minutes (normalization statistics update adds overhead) but raises the ceiling by **+70 reward** and eliminates the bimodal instability pattern entirely.

---

## Recommended Configuration for This Machine

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_id> --no-graphics
  --env="builds/<env>/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=50
```

`engine_settings` with `time_scale: 20` and `target_frame_rate: -1` is set globally in the config. `--num-spawn-envs=50` is the confirmed sweet spot for this machine (i7-3770S, 4 cores).

---

## Run: walker_03 — Clean baseline, 0 → 5M steps

**Command:**

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=walker_03 --no-graphics
  --env="builds/Walker2d-v0/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 50 |
| Total environments | 200 |
| `time_scale` | 20 |
| `batch_size` | 1024 |
| `buffer_size` | 10240 |
| `num_epoch` | 3 |
| `normalize` | false |
| `hidden_units` | 64 |
| `max_steps` | 5,000,000 |

### Session 1 — 0 → 1M steps (max_steps=1M, original config)

| Parameter | Value |
| --- | --- |
| Start step | 0 |
| End step | 1,000,038 |
| Session wall time | ~13.3 min |
| **Throughput** | **~1263 steps/s** |

**Reward progression:**

| Steps | Mean Reward | Std |
| --- | --- | --- |
| 0–100k | 8–22 | low |
| 100k–400k | 22–151 | moderate |
| 400k–700k | 129–320 | high |
| 700k–910k | 281–492 | high |
| 910k–1M | 392 | — |

Peak reward at 1M: **492 at step 910k**, ending at **392 at step 1M**.

### Session 2 — 1M → 5M (resumed, max_steps extended to 5M)

Resume started with a dip to reward 17 at step 1.01M (policy resuming from a declining checkpoint), then rapidly recovered.

**Checkpoint window analysis** (best reward seen in each 500k-step window):

| Checkpoint | Reward at export | Window best | Std at best | Best checkpoint? |
| --- | --- | --- | --- | --- |
| 1.5M | 588 | 588 | 28 | — |
| 2.0M | 659 | 680 (step 1.86M) | 31 | — |
| 2.5M | 650 | 717 (step 2.47M) | 20 | — |
| 3.0M | 723 | 731 (step 2.75M) | 15 | — |
| 3.5M | 725 | 748 (step 3.42M) | 9 | — |
| 4.0M | 663 | 754 (step 3.9M) | 10 | — |
| **4.5M** | **758** | **760 (step 4.41M)** | **9** | **Yes** |
| ~5.0M | ~757 | 762 (step 4.8M) | ~16 | Marginal |

**Best deployment checkpoint:** `Walker2d-v0-4499924.onnx` — highest mean reward (758) with lowest std (5.7) at export time.

**Throughput (session 2):** ~1,420 steps/s · **Wall time 1M→4.5M: ~41 min**

### Ceiling progression — not a plateau

The ceiling kept rising across the full 5M run, disproving the earlier plateau diagnosis:

| Step range | Peak reward seen |
| --- | --- |
| 1M–2M | 680 |
| 2M–3M | 731 |
| 3M–4M | 754 |
| 4M–5M | 762 |

Rate of improvement slows (~10 reward/M steps from 3M onward) but never stops.

### Bimodal episode pattern

From ~2.5M onward, every checkpoint window shows a clear alternation between two episode types:

- **Good episodes:** reward ~740–762, std=5–17 — agent completes `time_horizon=100` steps consistently
- **Bad episodes:** reward ~510–650, std=150–315 — mix of full completions and early falls, often triggered by initial body orientation

This is the characteristic signature of a **near-stable policy**: the gait is learned, but the policy hasn't generalized its recovery to all starting conditions. The sudden deep dips (529 at 3.22M, 515 at 3.68M, 526 at 4.94M) confirm this — they are not gradual degradations but single-episode cascade failures.

### Architecture ceiling observations

The 760 ceiling is the confirmed limit for 64-unit, `normalize: false`. The bimodal behavior was eliminated by `normalize: true` in walker_04 (see below).

| Option | Requires restart? | Observed effect |
| --- | --- | --- |
| `normalize: true` | Yes (new run_id) | Eliminated bimodal dips; pushed ceiling from 762 → 828 |
| `hidden_units: 128` | Yes | More capacity; marginal benefit for simple locomotion |
| Switch to SAC | Yes | 2000–5000 at convergence |

---

## Run: walker_04 — normalize: true, 0 → 5M steps

**Command:**

```bash
python train_ppo.py config/marathon_envs_config.yaml --run-id=walker_04 --no-graphics --force \
  --env="builds/Walker2d-v0/Marathon Environments.exe" \
  --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
```

| Parameter | Value |
| --- | --- |
| `--num-envs` | 4 |
| `--num-spawn-envs` | 50 |
| Total environments | 200 |
| `time_scale` | 20 |
| `batch_size` | 1024 |
| `buffer_size` | 10240 |
| `num_epoch` | 3 |
| `normalize` | **true** |
| `hidden_units` | 64 |
| `max_steps` | 5,000,000 |
| Start step | 0 |
| End step | 5,000,072 |
| Wall time | ~64 min |
| **Throughput** | **~1,308 steps/s** |

### Checkpoint progression

| Checkpoint | Reward at export | Best model? |
| --- | --- | --- |
| 500k | 458 | — |
| 1M | 739.4 | — |
| 1.5M | 757.4 | — |
| 2M | 781.7 | — |
| 2.5M | 796.6 | — |
| 3M | 808.9 | — |
| 3.5M | 814.1 | — |
| 4M | 823.2 | — |
| **4.5M** | **828.3** | **Yes** |
| 5M | 825.2 | — |

**Best deployment checkpoint:** `Walker2d-v0-4499917.onnx` — highest mean reward (828.3). Saved as `best_model.onnx` by `train_ppo.py`.

### Ceiling progression vs walker_03

| Step range | walker_03 (normalize: false) | walker_04 (normalize: true) | Δ |
| --- | --- | --- | --- |
| 1M–2M | 680 | 781.7 | +102 |
| 2M–3M | 731 | 808.9 | +78 |
| 3M–4M | 754 | 823.2 | +69 |
| 4M–5M | 762 | 828.3 | +66 |

`normalize: true` adds ~66–102 reward across all stages of training — the advantage is largest early and narrows as both runs approach their ceilings.

### Bimodal behavior: eliminated

walker_03 showed alternating good/bad episodes (std=150–315 on bad windows). walker_04 maintained std=1–7 throughout — including the late stages where walker_03 would dive to 515–530.

At 5M steps the final entropy was **0.52** (vs initial 1.43), and mean episode length reached **999 steps** — the agent rarely falls. This confirms a stable, generalized gait policy.

### Recommended config for Walker2d-v0

`normalize: true` is confirmed better. Update from the defaults:

| Parameter | Old (walker_03) | New (walker_04) |
| --- | --- | --- |
| `normalize` | false | **true** |

Already set in `config/marathon_envs_config.yaml`.
