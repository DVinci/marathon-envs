# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Marathon Environments** is a set of high-dimensional continuous control benchmarks for reinforcement learning using Unity's ML-Agents Toolkit and PhysX physics engine. It provides 16 environments from classical locomotion (Hopper, Walker, Ant, MarathonMan) to style transfer (DeepMimic-based) and procedural terrain variants.

- Unity version: 6000.3.15f1+
- ML-Agents: 4.0.2 (via UPM; replaces local 0.14.1 patch)
- Python: 3.10+ with Farama Gymnasium interface (`gymnasium>=0.29.1`, `mlagents-envs==1.1.0`)

## Repository Layout

```
marathon-envs/
├── UnitySDK/Assets/MarathonEnvs/     # C# environments, agents, scenes
│   ├── Scripts/ActiveRagdoll002/     # Core physics scripts (MarathonAgent, BodyManager002, Muscle002)
│   ├── Agents/Scripts/               # Environment-specific agent implementations
│   ├── Agents/Xml/                   # DeepMind/OpenAI-reference physics definitions
│   ├── Agents/Models/                # Original pre-trained .nn models (from repo)
│   └── Agents/Brains/                # Our trained .onnx models (deployed for inference)
├── UnitySDK/Assets/SpawnableEnvs/    # EnvSpawner, SpawnableEnv, SelectEnvToSpawn
├── marathon-envs/marathon_envs/      # Python Gymnasium wrapper package
│   ├── envs/__init__.py              # MarathonEnvs Gymnasium wrapper (main Python interface)
│   └── tests/test_gym.py             # Gymnasium wrapper unit tests (mock UnityEnvironment)
├── builds/                           # Built Unity executables, one folder per environment
├── config/marathon_envs_config.yaml  # PPO hyperparameters for all 16 environments
├── config/optimal_spawn_envs.json    # Calibrated num_envs + num_spawn_envs per environment
├── train_ppo.py                      # Single-env training wrapper with best_model tracking
├── train_all_envs.py                 # Sequential multi-env training with resume support
├── calibrate_envs.py                 # Find optimal parallelism per build (run once per machine)
├── stable_baselines_sac_train.py     # SAC training (alternative to PPO)
├── stable_baselines_sac_run.py       # SAC inference runner
├── Training.md                       # Per-environment training commands (all platforms)
├── Changes.md                        # ML-Agents patch migration guide
└── Dockerfile                        # Ubuntu 16.04 + Python 3.6.4 headless training image
```

## C# Agent Class Hierarchy

```
Agent (ml-agents base)
└── MarathonAgent          # Scripts/ActiveRagdoll002/MarathonAgent.cs
    ├── BodyManager002     # Central physics manager — muscles, sensors, frame rewards, FixedDeltaTime
    ├── RagDoll002         # Ragdoll character configuration
    ├── Muscle002          # Joint motor actuator controller
    └── BodyPart002        # Physical body segment representation
        └── DeepMindHopperAgent   # Agents/Scripts/DeepMindHopperAgent.cs
        └── DeepMindWalkerAgent
        └── OpenAIAntAgent
        └── MarathonManAgent
        └── TerrainHopperAgent / TerrainWalkerAgent / ...
```

`BodyManager002` is the central coordinator — it manages the `Muscles`, `BodyParts`, and `Observations` collections, tracks `DistanceTraveled` and terrain contact flags, and controls physics simulation frequency (`FixedDeltaTime`). Each environment-specific agent class extends `MarathonAgent` and overrides reward/observation/termination logic.

## Key Systems

**EnvSpawner** (`UnitySDK/Assets/SpawnableEnvs/`) — Dynamically spawns multiple parallel environment instances during training. Injected into ml-agents via `--spawn-env` and `--num-spawn-envs` CLI args (not native ml-agents; documented in `Changes.md`).

**Style Transfer / Phase Imitation** — MarathonMan variants use motion capture data (`Assets/MarathonEnvs/Animations/`) with phase-based reward signals to learn human-like movement styles.

**Python Gym Wrapper** — `MarathonEnvs` class in `marathon-envs/marathon_envs/envs/__init__.py` wraps `UnityEnvironment` from `mlagents_envs`. Sets time scale to 20× and quality level to 0 for training speed. Provides `reset()`, `step()`, and `render()` following OpenAI Gym conventions.

**XML Physics Definitions** — `Agents/Xml/` contains both original DeepMind/OpenAI reference XML files (e.g., `dm_hopper.xml`, `unity_oai_ant.xml`) and Unity-adapted variants used to match the benchmark physics specifications.

**MarathonSpawner** (`Scripts/ActiveRagdoll002/MarathonSpawner.cs`) — Parses the XML at runtime and builds the ragdoll hierarchy (Rigidbodies, ConfigurableJoints, capsule colliders). Key details:

- MuJoCo → Unity coordinate transform: `RightToLeft(X,Y,Z) = (-X, Z, -Y)` (positions); joint axes use `(x, -z, y)`
- ConfigurableJoint axes are **negated** vs MuJoCo hinge axes (line ~1230 in `ToConfigurable()`)
- `init_qpos` from XML is **never applied** — all joints spawn at 0° regardless
- Episode resets restore saved spawn transforms and zero all velocities (first episode always spawns from XML)

## Available Environments

| Category | Environments |
|---|---|
| Classical | Hopper-v0, Walker2d-v0, ~~Ant-v0~~ (defective), MarathonMan-v0 |
| Style Transfer | MarathonManWalking/Running/JazzDancing/MMAKick/PunchingBag/Backflip-v0 |
| Procedural Terrain | TerrainHopper/Walker2d/Ant/MarathonMan-v0 |
| Controller (DReCon) | ControllerMarathonMan-v0 (preview) |
| Sparse Reward | MarathonManSparse-v0 |

**Ant-v0 is defective** — `MarathonSpawner` ignores `init_qpos`, spawning all legs horizontal. Episodes terminate in ~5–20 steps before the torso falls enough for feet to contact the ground. The agent learns belly-sliding, not walking. TerrainAnt-v0 likely shares the same issue.

## Commands

### Python setup

```bash
cd marathon-envs
pip install -e .
```

### Run tests

```bash
python -m pytest marathon-envs/marathon_envs/tests/test_gym.py
```

### PPO training — normal workflow

**Step 1 — calibrate (once per machine/build):**
Finds the optimal `num_envs` × `num_spawn_envs` for each build and writes results to `config/optimal_spawn_envs.json`. Takes ~45–75 min for all 16 environments.

```bash
python calibrate_envs.py
python calibrate_envs.py --envs Walker2d-v0 Ant-v0   # subset only
```

Afterwards, clean up calibration artifacts:

```bash
Remove-Item -Recurse results\calib_* summaries\calib_*
```

**Step 2a — train one environment** (uses `optimal_spawn_envs.json` automatically):

```bash
python train_ppo.py config/marathon_envs_config.yaml --run-id=20260524-Walker2d-v0 --env="builds\Walker2d-v0\Marathon Environments.exe" --no-graphics --num-envs=4 --env-args --spawn-env=Walker2d-v0 --num-spawn-envs=50
```

`train_ppo.py` wraps `mlagents-learn` and saves `<run_id>_best.onnx` next to checkpoints whenever a new reward high is reached.

**Step 2b — train all environments sequentially:**

```bash
python train_all_envs.py
python train_all_envs.py --envs MarathonManSparse-v0 TerrainHopper-v0   # subset
python train_all_envs.py --resume                                         # resume interrupted batch
```

Uses today's date as run prefix (e.g. `20260524`), reads `optimal_spawn_envs.json` for per-env parallelism, and saves `<prefix>-<envId>_best.onnx` alongside checkpoints. Batch state is saved to `train_all_envs.last_run` for `--resume`.

**From Unity Editor** (no built exe needed):

```bash
mlagents-learn config/marathon_envs_config.yaml --train --run-id=<run_name>
# Wait for "Listening on port 5004", then press Play in Unity Editor.
# A popup appears — select the environment and click GO.
```

### Monitor training

```bash
tensorboard --logdir=summaries
```

### SAC alternative (Stable Baselines3)

```bash
python stable_baselines_sac_train.py   # train; saves to models/
python stable_baselines_sac_run.py     # inference
```

### Docker

```bash
docker build -t marathon-envs .
```

## PPO Hyperparameter Notes

`config/marathon_envs_config.yaml` contains per-environment overrides. Key differences:

- **Classical** (Hopper/Walker/Ant): `max_steps=5M`, `batch_size=1024`, `lr=1e-3`, `time_horizon=100–1000`
- **MarathonMan-v0**: `max_steps=50M`, `lr=3e-4` — complex humanoid needs ~10× more steps than Hopper/Walker
- **Terrain variants**: `max_steps=50M`, `lr=3e-4`, `beta=0.1`
- **Style Transfer** (Walking/Running/Dancing/MMAKick/PunchingBag): `max_steps=64M`, `batch_size=768`, `lr=1e-4`
- **MarathonManBackflip-v0** / **ControllerMarathonMan-v0**: `max_steps=128M` — longest runs
- **`learning_rate_schedule: linear`** on all envs means LR decays to ~0 by `max_steps`. Resuming beyond the original `max_steps` causes policy regression — start a fresh run instead.

## ML-Agents Migration Notes

The project was upgraded from ML-Agents 0.14.1 (local patch) to 4.0.2 (UPM). `Changes.md` documents the original 0.14.1 patches that are now superseded. Key C# API changes made during the 4.0.2 migration:

- `using MLAgents` → `using Unity.MLAgents` / `Unity.MLAgents.Sensors` / `Unity.MLAgents.Actuators`
- `AgentReset()` → `OnEpisodeBegin()`
- `CollectObservations()` → `CollectObservations(VectorSensor sensor)`
- `AgentAction(float[] v)` → `OnActionReceived(ActionBuffers actions)`
- `AddVectorObs(x)` → `sensor.AddObservation(x)`
- `Done()` → `EndEpisode()`
- `GetStepCount()` → `StepCount` (property)
- `GetCumulativeReward()` → `CumulativeReward` (property)
- Barracuda 0.6.1-preview → Unity Sentis 2.2.1 (inference engine replacement)
- `Agent.Instantiate(...)` → `Object.Instantiate(...)`

## Development Workflow

1. Modify C# environment logic in `UnitySDK/Assets/MarathonEnvs/Scripts/` or `Agents/Scripts/`
2. Build Unity 6000.3.15f1 project for target platform (Windows `.exe`, Linux `x86_64`) into `builds/<EnvId>/`
3. Train with `python train_all_envs.py --envs <EnvId>` (reads `optimal_spawn_envs.json` automatically)
4. Copy `results/<run_id>/<EnvId>/<run_id>_best.onnx` into `Agents/Brains/` and assign in the Env prefab's `m_Model` override
5. For Gymnasium/SAC training: use the Python wrapper in `marathon-envs/`

**Model deployment note:** Env prefabs (e.g. `AntEnv-v0.prefab`) contain a `m_Model` serialized override that replaces whatever the character prefab specifies. Always update the Env prefab's override — editing the character prefab alone has no effect during normal play.

## Branch Workflow

| Branch | Purpose |
| --- | --- |
| `master` | Protected. Merged from `develop` via PR only. |
| `develop` | Default PR target. All work lands here first. |
| `feature/<name>` | Branch from `develop` for new work. |
| `hotfix/<name>` | Branch from `master`; merge to both `master` and `develop`. |

## Test Suite (npm)

```bash
npm test                  # Run all tests (YAML + consistency + link existence)
npm run test:format       # YAML format and schema validation only
npm run test:consistency  # Cross-file consistency checks
npm run test:links        # External URL check (requires LINK_CHECK=true)
npm run test:ui           # Playwright visual debug mode
```

## GitHub Actions

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `validate.yml` | PR + develop push | Runs full test suite |
| `link-check.yml` | Weekly Mon 9am UTC | Checks all external URLs in markdown files |
| `pr-labeler.yml` | Every PR | Auto-applies labels by changed file paths |
| `sync-upstream.yml` | Daily 7am UTC | Syncs from Unity-Technologies/marathon-envs |

## Documentation Directive

The `docs/` folder is a living knowledge base built from Q&A sessions with the project owner. It contains:

- `docs/training-guide.md` — practical training commands, performance, builds, deployment
- `docs/rl-concepts.md` — RL theory relevant to this project
- `docs/ecosystem.md` — tools, formats, ecosystem comparisons

**ALWAYS** update or extend these documents (or create new ones) whenever the user asks questions that produce knowledge relevant to this project — RL concepts, training insights, architecture decisions, tool comparisons, Unity/ML-Agents specifics. Commit the changes to the `develop` branch. The goal is that future sessions can retrieve this knowledge by reading `docs/` rather than rediscovering it through conversation.
