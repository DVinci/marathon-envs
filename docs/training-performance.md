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
```
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
```
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

## Full Comparison

| Config | time_scale | Processes | Total envs | Throughput | Est. time to 1M |
| --- | --- | --- | --- | --- | --- |
| hopper_01 session A | unset (=1) | 1 | 20 | 61 steps/s | 4.6h |
| hopper_01 session B | unset (=1) | 1 | 50 | 58 steps/s | 4.8h |
| walker_02 (old config) | unset (=1) | 4 | 80 | 59 steps/s | 4.7h |
| **walker_02 resumed** | **20** | **4** | **80** | **~1192 steps/s** | **~14 min** |

---

## Recommended Configuration for This Machine

```bash
mlagents-learn config/marathon_envs_config.yaml --run-id=<run_id> --no-graphics
  --env="builds/<env>/Marathon Environments.exe"
  --num-envs=4 --env-args --spawn-env=<EnvName-v0> --num-spawn-envs=20
```

`engine_settings` with `time_scale: 20` and `target_frame_rate: -1` is now set globally in the config and applies to all environments.
