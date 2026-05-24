# Ecosystem: Tools, Formats, and Comparisons

## Model Formats

| Format | Used by | Key trait |
|---|---|---|
| `.pt` / `.pth` | PyTorch | Native training format, includes optimizer state |
| `.safetensors` | HuggingFace | Safe version of `.pt` — can't execute arbitrary code on load |
| `.onnx` | Cross-platform | Inference standard, works across frameworks (PyTorch, TF, Unity Sentis) |
| `.gguf` | llama.cpp | Optimized for CPU inference, supports quantization (4-bit, 8-bit) |
| `.nn` | Unity Barracuda (old) | Unity's proprietary format, replaced by ONNX + Sentis in Unity 6 |
| `.pb` | TensorFlow | TF frozen graph |

**General flow for LLMs:**
```
Train (.pt) → export (.safetensors) → quantize (.gguf) → run locally
```

**For this repo:**
```
Train (mlagents → .pt checkpoints) → export (.onnx) → deploy (Unity Sentis)
```

`.nn` files from ML-Agents 0.14.1 (TensorFlow-based) cannot be used with Unity Sentis or ML-Agents 4.0.2. Converting them requires TensorFlow + tf2onnx, but the converted ONNX models are unlikely to be compatible with ML-Agents 4.0.2's expected input/output node naming conventions. **Retraining is the practical path.**

---

## Unity ML-Agents vs Isaac Lab

| | Unity ML-Agents | Isaac Lab |
|---|---|---|
| Setup time | Hours | Days/weeks |
| Visual tooling | Excellent (editor) | Minimal (Python + USD) |
| Training speed | ~60 steps/s (PhysX) | ~5,000–15,000 steps/s (GPU physics) |
| Target user | Game devs / hobbyists | Robotics researchers |
| Environment format | GameObjects/prefabs | USD/URDF/MJCF |
| Deployment | Unity Sentis (native) | ROS2 or custom |
| Hardware | Any CPU | NVIDIA GPU required |

**No official Unity ↔ Isaac Lab bridge exists.** The ONNX export path is theoretically possible but undocumented — observation/action space conventions differ between ecosystems.

**Isaac Lab GPU physics** runs thousands of parallel environments on a single GPU. An RTX 2070 Super (8GB VRAM) could run ~hundreds of Walker2d environments vs 20 in Unity.

**Why Unity is still the right choice for games:** visual editor, prefab workflow, seamless Sentis inference integration, and the training speed is acceptable for non-research use.

---

## ECS/DOTS for Physics Performance

Unity's ECS (Entity Component System) + Burst Compiler + Job System would address the exact bottleneck (single-threaded PhysX):

- **Burst Compiler**: SIMD-vectorized native code → faster physics per step
- **Job System**: distributes physics across all CPU cores
- **Unity Physics (DOTS)**: designed for massive parallelism

Estimated improvement: **5–20× more environments** at the same CPU budget.

**Why it hasn't been done:** requires rewriting every MonoBehaviour as ECS components, re-implementing ragdoll articulation in DOTS, and ML-Agents has no official DOTS support. Months of work.

---

## Torch.NET / TorchSharp (Replacing Python Backend)

**Torch.NET** (SciSharp): essentially abandoned, still requires Python+PyTorch, not production-ready.

**TorchSharp** (Microsoft): actively maintained, ships LibTorch natively (no Python). Would require reimplementing PPO from scratch in C#.

**Would there be performance gains?** No meaningful ones. The Python ↔ Unity communication overhead is ~5–10% of training time. The bottleneck is PhysX physics simulation, not the Python trainer. GPU was at 100% waiting for Unity even in the current setup.

**For inference-only:** already solved by Unity Sentis (runs ONNX natively in C#, no Python).

---

## LLM Context Windows

LLMs have large context windows (128k–1M tokens) because:
- Language is not Markov — meaning depends on everything written before
- Long-range dependencies (conclusion requires premise from 50 pages earlier)
- In-context learning (examples in prompt teach new tasks without retraining)

**RL agents contrast:** Walker2d needs ~30 numbers (joint angles, velocities) — current physics state is sufficient. No memory of past steps needed because the Markov assumption holds.

Attention scales as **O(n²)** with sequence length — doubling context quadruples compute. This is why large context is expensive.

---

## LLM Model Sizes (Known / Estimated)

| Model | Parameters | Source |
|---|---|---|
| GPT-3 | 175B | Published (2020) |
| LLaMA 3 | 8B, 70B, 405B | Published (open source) |
| GPT-4 | ~1.8T (rumored MoE, 8×220B) | Leaked/estimated |
| Claude 3 Opus | ~200B–500B estimated | Triangulated from benchmarks |
| Gemini Ultra | Undisclosed | — |

Frontier labs (OpenAI, Anthropic, Google) stopped publishing architecture details after GPT-3 as models became commercially valuable. Open model benchmarks allow order-of-magnitude estimates but not precise numbers.

---

## Physics-Based Character Games (Shipped in Unity)

No shipped commercial games use ML-Agents / RL for character locomotion. All successful physics character games use hand-crafted active ragdoll systems.

| Game | Approach | Notes |
|---|---|---|
| Gang Beasts | Active ragdoll, configurable joints | Multi-platform hit |
| Human: Fall Flat | Active ragdoll | 5M+ players |
| Totally Accurate Battle Simulator | Wobbly physics locomotion | Multi-platform |
| Heave Ho | ConfigurableJoint arms | Devolver Digital |
| Very Very Valet | Physics-based controller | Nintendo Switch, PS5 |

**The gap:** nobody has shipped a game with RL-trained physics locomotion. MarathonEnvs demonstrated technical feasibility but the jump to shipped product hasn't happened. Building this would be genuinely unexplored territory.

**Why the gap exists:**
- Training time makes iteration slow (can't "tweak" like animation)
- RL policies can fail unpredictably on edge cases
- Hard to art-direct — you retrain, not animate
- Hand-crafted ragdolls give more deterministic, tunable results

---

## PhysX Optimization Options

PhysX is single-threaded in Unity (one thread per scene). The bottleneck is physics steps per second, not the neural network. Options in ascending order of effort:

| Option | Effort | Gain | Notes |
| --- | --- | --- | --- |
| Reduce solver iterations | Minutes | 5–15% | `Physics.defaultSolverIterations` default is 6; 4 often works |
| Simplified collider geometry | Hours | 10–20% | Convex hulls on ragdoll limbs, remove redundant colliders |
| Increase `FixedDeltaTime` | Minutes | 20–50% | Fewer physics steps/second; risks simulation instability |
| Reduce parallel env count | Minutes | Linear | Fewer spawned environments = more steps/s each |
| DOTS/ECS rewrite | Months | 5–20× | Parallel physics across CPU cores; no ML-Agents support |
| Better hardware | Days (cloud) | 2–4× | More cores ≠ faster per-env; CPU clock speed matters most |

**Quick wins:** reducing solver iterations from 6 to 4 and simplifying collider meshes can recover 15–25% physics throughput with no visible quality loss. The `BodyManager002.cs` `FixedDeltaTime` property is already configurable per-environment.

**DOTS/ECS:** theoretically the right answer but requires rewriting every `MonoBehaviour` as ECS components, reimplementing articulated ragdolls in Unity Physics, and ML-Agents has no official DOTS support. Not a near-term option.

**Real hardware path:** cloud CPU VMs (c5.9xlarge, 36 vCPUs, $1.53/hr) effectively multiply environments proportionally to core count. See [cloud-training.md](cloud-training.md).

---

## Sentis Inference Backend

Unity Sentis (ML-Agents 4.0.2) supports four inference backends controlled by `inferenceDevice` on each Brain `.asset` file:

| Value | Enum | Backend | Best for |
| --- | --- | --- | --- |
| 0 | Default | Auto-selects GPU | Large models (>5MB) |
| 1 | GPU (Compute) | GPU shader dispatch | Large models |
| 2 | CPU (Burst) | SIMD-vectorized CPU | Small MLPs (marathon-envs) |
| 3 | CPU (Fallback) | Plain C# | Debugging only |

**Current state:** all marathon-envs Brain assets use `inferenceDevice: 0` (Default → GPUCompute on RTX 2070 Super).

**Optimal setting for this project:** `2` (Burst). Marathon-envs policy networks are tiny (2–3 hidden layers, 256 units each, <1MB). GPU dispatch overhead dominates at this scale — Burst executes small tensor ops faster because it avoids the CPU→GPU transfer and kernel launch latency. Crossover is around 5MB model size.

To change: open each `.asset` file in `UnitySDK/Assets/MarathonEnvs/Agents/Brains/` and set `m_InferenceDevice: 2`, or edit the Brain asset in the Unity Inspector under "Inference Device → CPU (Burst)". The difference is marginal for training (bottleneck is PhysX, not inference), but matters for deployed real-time inference in a game.
