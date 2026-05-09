# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Marathon Environments** is a set of high-dimensional continuous control benchmarks for reinforcement learning using Unity's ML-Agents Toolkit and PhysX physics engine. It provides 16 environments from classical locomotion (Hopper, Walker, Ant, MarathonMan) to style transfer (DeepMimic-based) and procedural terrain variants.

- Unity version: 2020.1+
- ML-Agents: 0.14.1 (with custom patches — see `Changes.md`)
- Python: 3.6+ with OpenAI Gym interface

## Repository Layout

```
marathon-envs/
├── UnitySDK/Assets/MarathonEnvs/     # C# environments, agents, scenes
│   ├── Scripts/ActiveRagdoll002/     # Core physics scripts (MarathonAgent, BodyManager002, Muscle002)
│   ├── Agents/Scripts/               # Environment-specific agent implementations
│   └── Agents/Xml/                   # DeepMind/OpenAI-reference physics definitions
├── UnitySDK/Assets/SpawnableEnvs/    # EnvSpawner, SpawnableEnv, SelectEnvToSpawn
├── com.unity.ml-agents/              # ML-Agents package with custom modifications
│   ├── Runtime/                      # Agent, Academy, Brain, Sensor interfaces
│   └── Plugins/                      # gRPC, Protobuf, Barracuda
├── marathon-envs/marathon_envs/      # Python Gym wrapper package
│   ├── envs/__init__.py              # MarathonEnvs Gym wrapper (main Python interface)
│   └── tests/test_gym.py             # Gym wrapper unit tests (mock UnityEnvironment)
├── config/marathon_envs_config.yaml  # PPO hyperparameters for all 16 environments
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

## Available Environments

| Category | Environments |
|---|---|
| Classical | Hopper-v0, Walker2d-v0, Ant-v0, MarathonMan-v0 |
| Style Transfer | MarathonManWalking/Running/JazzDancing/MMAKick/PunchingBag/Backflip-v0 |
| Procedural Terrain | TerrainHopper/Walker2d/Ant/MarathonMan-v0 |
| Controller (DReCon) | ControllerMarathonMan-v0 (preview) |
| Sparse Reward | MarathonManSparse-v0 |

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

### PPO training (ml-agents)

```bash
mlagents-learn config/marathon_envs_config.yaml --train \
  --env="envs\MarathonEnvs\Unity Environment.exe" \
  --run-id=<run_name> \
  --spawn-env=<EnvironmentName-v0> \
  --num-spawn-envs=<N>
```

### Train from Unity Editor (development — no built exe needed)

```bash
mlagents-learn config/marathon_envs_config.yaml --train --run-id=<run_name> --spawn-env=<EnvironmentName-v0>
# Then press Play in Unity Editor (connects on port 5004)
```

### Headless / server training

Add `--no-graphics` to any `mlagents-learn` command.

### Monitor training

```bash
tensorboard --logdir=summaries
```

### SAC alternative (Stable Baselines)

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

- **Style Transfer** (Walking/Running/Dancing/Backflip): `max_steps=64–128M`, `batch_size=768`, `lr=1e-4` — very long training runs
- **Classical** (Hopper/Walker/Ant): `max_steps=1M`, `batch_size=16–32`
- **Terrain variants**: `max_steps=50M`, `batch_size=32`

## ML-Agents Custom Patches

`Changes.md` is the authoritative migration guide when updating ml-agents versions. Key patches applied to 0.14.1:

- CLI args `--spawn-env`, `--num-spawn-envs` added to `Academy.cs`
- `EnvSpawner`, `SpawnableEnv`, `SelectEnvToSpawn` integration
- Physics simulation frequency control
- `Agent.OnDestroy` cleanup for proper lifecycle management
- Optional action skipping for physics efficiency

## Development Workflow

1. Modify C# environment logic in `UnitySDK/Assets/MarathonEnvs/Scripts/` or `Agents/Scripts/`
2. Build Unity project for target platform (Windows `.exe`, Linux `x86_64`)
3. Run training with `mlagents-learn` against the built executable
4. Deploy trained `.nn` model back into Unity for inference testing
5. For Gym/SAC training: use the Python wrapper in `marathon-envs/`

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
