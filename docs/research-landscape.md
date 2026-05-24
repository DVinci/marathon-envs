# Research Landscape: Physics-Based Character Animation

This document synthesizes a collection of 162 research papers (2012–2026) on physics-based character animation, RL locomotion, motion synthesis, and related fields. It is organized to orient marathon-envs development: which technique families exist, how they connect, what marathon-envs already implements, and where the field has moved beyond the current implementation.

Source collection: `D:\Projetos\Physics Character Controller\Knowledge\` (162 PDFs + catalog).

---

## What Marathon-Envs Already Implements

| Technique | Paper | Marathon-Envs Location |
| --- | --- | --- |
| Physics-based PPO locomotion | OpenAI Gym benchmarks (Hopper, Walker2d, Ant) | All classical environments |
| DeepMimic-style explicit reward | Peng et al., SIGGRAPH 2018 | StyleTransfer002Agent.cs — 7 exp(-distance) components |
| Energy efficiency penalty | Standard in RL locomotion | BodyManager002.GetEffortNormalized() |
| Active ragdoll with PD joints | Common approach | Muscle002.cs, MarathonJoint.cs |
| Data-driven responsive control (preview) | DReCon, Bergamin et al. 2019 | ControllerMarathonMan-v0 |
| SAC training | Haarnoja et al. 2018 | stable_baselines_sac_train.py |
| ONNX/Sentis inference | Unity standard | Via mlagents-learn export |

Marathon-envs' core loop: **RL (PPO/SAC) + explicit pose-matching reward + active ragdoll physics + ONNX inference**. This corresponds to the 2018–2019 state of the art.

---

## Technique Families

### 1. Imitation Learning & Reference Motion Tracking

**What it is:** The agent learns to reproduce motion capture clips by receiving reward proportional to how closely it matches the reference pose, velocity, and end-effector positions at each timestep.

**Marathon-envs relevance: HIGH** — this is the current StyleTransfer family.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| DeepMimic | 2018 | SIGGRAPH | Reference-state initialization (RSI); 7-component exponential reward | [xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic) |
| AMP | 2021 | SIGGRAPH | Replace explicit reward with adversarial discriminator | [xbpeng/amp](https://github.com/xbpeng/amp) |
| ADD | 2023 | SIGGRAPH | Adversarial differential discriminator on velocity | No public code |
| SFV | 2019 | SIGGRAPH Asia | Learn from RGB video instead of MoCap | No public code |
| SuperTrack | 2021 | SIGGRAPH | Supervised learning via differentiable physics (no RL) | Academic (Ubisoft La Forge) |

**Evolution:** DeepMimic → AMP removes hand-crafted reward weights → ADD improves temporal discrimination → SFV removes MoCap dependency entirely.

---

### 2. Reusable Skill Embeddings

**What it is:** Instead of training one policy per behavior, learn a shared latent space where each point represents a distinct skill. A high-level controller selects skills; a low-level decoder executes them. Skills trained once can be reused across many downstream tasks.

**Marathon-envs relevance: HIGH** — natural next step after AMP; allows all 16 environments to share a single pre-trained low-level controller.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| ASE | 2022 | SIGGRAPH | Large-scale unsupervised skill discovery | [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) |
| CALM | 2023 | SIGGRAPH | Conditional latent space + language/goal conditioning | [NVlabs/CALM](https://github.com/NVlabs/CALM) |
| C·ASE | 2023 | SIGGRAPH | Conditional adversarial embeddings on motion subsets | No public code |
| PHC | 2023 | ICCV | Multiplicative primitives + fall recovery | [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) |
| PULSE | 2024 | ICLR | Distilled universal prior from large imitator | [ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE) |
| MaskedMimic | 2024 | SIGGRAPH | Masked motion inpainting — unified multi-task | [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) |
| SONIC | 2024 | SIGGRAPH | Supersizing motion tracking — whole-body control | [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) |
| PLT | 2025 | SIGGRAPH | Per-part codebooks; compositional skill learning without forgetting | [jinseokbae/plt](https://github.com/jinseokbae/plt) |
| AMOR | 2025 | SIGGRAPH | Pareto-front multi-objective RL; post-training reward-weight tuning | No public code |
| UniPhys | 2025 | ICCV | Diffusion-based unified planner + controller, no separate planning | [wuyan01/UniPhys](https://github.com/wuyan01/UniPhys) |

**Evolution:** ASE (unsupervised discovery) → CALM (conditioning on goals) → C·ASE (subset-based training) → PULSE (distillation from large teacher) → MaskedMimic (masking replaces skill selection) → PLT (per-part decomposition avoids catastrophic forgetting) → AMOR (Pareto front enables post-training tuning).

---

### 3. Phase-Based Neural Networks

**What it is:** Motion has periodic structure (strides, cycles). Encoding a phase parameter into the network allows it to generate temporally coherent, smoothly varying locomotion without memory (no RNN needed). The network weights themselves are a function of phase.

**Marathon-envs relevance: MEDIUM** — applicable as a kinematic layer above the current physics controller; strongest benefit for continuous locomotion at variable speeds.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| PFNN | 2017 | SIGGRAPH | Phase-indexed weight interpolation (cubic spline) | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) — MIT |
| Mode-Adaptive NN | 2018 | SIGGRAPH | Gating network selects from expert sub-networks | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) |
| Local Motion Phases | 2020 | SIGGRAPH | Per-body-part phase for contact-rich motion | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) |
| DeepPhase | 2022 | SIGGRAPH | Periodic autoencoder learns phase manifold from data | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) |
| Categorical Codebook | 2024 | SIGGRAPH | VQ-VAE codebook replaces phase | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) |

**Key resource:** All these papers have Unity C# implementations in [sebastianstarke/AI4Animation](https://github.com/sebastianstarke/AI4Animation) (8k+ stars, actively maintained 2025), with ONNX export. This is the highest-value kinematic code repository for Unity.

---

### 4. Motion Matching

**What it is:** At each frame, search a motion capture database for the clip that best matches current pose and desired future trajectory. No neural network needed for the core; neural networks are added for compression or improved search.

**Marathon-envs relevance: MEDIUM** — most valuable as the kinematic "upper layer" feeding trajectory targets into a physics controller (as in DReCon).

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| Learned Motion Matching | 2022 | SIGGRAPH | Neural approximation compresses database 50× | [Ubisoft La Forge](https://github.com/ubisoft-laforge/learned-motion-matching) |
| CAMDM | 2024 | SIGGRAPH | Diffusion model for motion generation with Unity ONNX demo | [AIGAnimation/CAMDM](https://github.com/AIGAnimation/CAMDM) |
| Environment-Aware MM | 2025 | SIGGRAPH Asia | Terrain-aware feature extraction for complex scenes | [UPC ViRVIG](https://github.com/JLPM22/MotionMatching) |
| MAMM | 2025 | SIGGRAPH | Metric-aligning matching from arbitrary control signals (sketches, audio) | — |
| MMVR | 2022 | SCA | Motion matching for VR with sparse sensors | [UPC-ViRVIG/MMVR](https://github.com/UPC-ViRVIG/MMVR) — Unity 6 |

**Key resource:** [JLPM22/MotionMatching](https://github.com/JLPM22/MotionMatching) — production-quality Unity 6, MIT license, 561 stars, ONNX support. Best starting point for a kinematic layer.

---

### 5. Responsive & Interactive Control

**What it is:** Controllers that respond in real-time to user input, direction changes, speed variations, and environmental perturbations. Often hybrid: kinematic planner + physics corrector.

**Marathon-envs relevance: HIGH** — directly addresses the game-dev use case. DReCon is already in marathon-envs as a preview.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| DReCon | 2019 | SIGGRAPH Asia | Motion matching + PPO feedback corrector | In marathon-envs (ControllerMarathonMan-v0) |
| AdaptNet | 2023 | SIGGRAPH Asia | Latent injection for fast style/task adaptation | [Roblox/AdaptNet](https://github.com/Roblox/AdaptNet) |
| PARC | 2025 | SIGGRAPH | Iterative terrain augmentation via RL+diffusion | [msooloo/PARC](https://github.com/msooloo/PARC) |
| CARL | 2020 | SIGGRAPH | Quadruped control with latent conditioning | Public |
| Control Operators | 2023 | SIGGRAPH | Composable operators for interactive animation | — |

---

### 6. Diffusion-Based Motion Generation

**What it is:** Diffusion models (DDPM/DDIM) generate motion by iteratively denoising from random noise, conditioned on text, goal, or past motion. Primarily kinematic (no physics); coupling with physics controllers is an active area.

**Marathon-envs relevance: LOW–MEDIUM** — kinematic output needs a physics tracking layer; high-quality motion generation could replace MoCap pipeline long-term.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| PhysDiff | 2023 | ICCV | Physics-guided diffusion model | — |
| CAMDM | 2024 | SIGGRAPH | 8-step diffusion for real-time control with Unity ONNX demo | [AIGAnimation/CAMDM](https://github.com/AIGAnimation/CAMDM) |
| PDP | 2024 | — | Diffusion policy for physics control | — |
| AAMDM | 2023 | SIGGRAPH Asia | Accelerated auto-regressive diffusion | — |
| MDM | 2023 | ICLR | Base motion diffusion model | [GuyTevet/MDM](https://github.com/GuyTevet/motion-diffusion-model) |

---

### 7. Language-Directed Control

**What it is:** The character controller is conditioned on natural language or semantic labels, enabling text-to-motion or instruction-following behaviors.

**Marathon-envs relevance: LOW** — research-stage, requires large pretrained language models; not applicable in short term.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| PADL | 2023 | SIGGRAPH | CLIP-conditioned physics control | [nv-tlabs/PADL](https://github.com/nv-tlabs/PADL) |
| SuperPADL | 2024 | SIGGRAPH | Progressive distillation for scaling | — |
| InsActor | 2024 | — | Instruction-driven characters | — |
| Eureka | 2023 | NeurIPS | LLM-designed reward functions (GPT-4) | [eureka-research/Eureka](https://github.com/eureka-research/Eureka) — MIT |

**Note:** Eureka is the most immediately practical entry: it uses GPT-4 to write reward functions automatically and then evaluates them in IsaacGym. Could be adapted to automate reward design for new marathon-envs environments.

---

### 8. Contact-Rich Interactions

**What it is:** Characters that interact with objects, other characters, or the environment via contact (grasping, sitting, climbing, fighting).

**Marathon-envs relevance: LOW–MEDIUM** — future direction once robust locomotion is solved.

| Paper | Year | Venue | Key Advance | Code |
| --- | --- | --- | --- | --- |
| InterMimic | 2025 | CVPR | 46k human-object interaction motions | [Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic) |
| Omnigrasp | 2024 | NeurIPS | Grasping diverse objects with physics | [ZhengyiLuo/Omnigrasp](https://github.com/ZhengyiLuo/Omnigrasp) |
| PMP | 2024 | — | Part-wise motion priors for interactions | [jinseokbae/pmp](https://github.com/jinseokbae/pmp) |
| MaskedManipulator | 2024 | — | Whole-body manipulation | — |
| Neural State Machine | 2019 | SIGGRAPH Asia | Character-scene interaction via state machine | [AI4Animation](https://github.com/sebastianstarke/AI4Animation) |

---

### 9A. Musculoskeletal & Biomechanical Simulation

**What it is:** Using anatomically detailed muscle actuators (instead of joint torques) to produce more realistic, emergent biomechanical gaits. The musculoskeletal model provides a strong physical prior that can replace MoCap entirely.

**Marathon-envs relevance: LOW-MEDIUM** — Muscle002.cs uses simplified joint motors, not full musculoskeletal simulation. FreeMusco is notable because it produces locomotion **without any MoCap** — the physics of muscles alone drives gait discovery.

| Paper | Year | Key Advance | Code |
| --- | --- | --- | --- |
| Flexible Muscle-Based Locomotion | 2012 | Bipedal locomotion from spinal reflexes + muscles (SIGGRAPH Asia) | — |
| GaitNet | 2022 | Anatomically-parameterized gait families (SIGGRAPH) | — |
| FreeMusco | 2025 | Motion-free musculoskeletal RL — no MoCap needed (SIGGRAPH Asia) | — |

**FreeMusco note:** Learns bipedal/quadrupedal gaits from energy-aware RL using only the musculoskeletal structure as prior. Downstream goal navigation and path following emerge from latent space conditioning. If marathon-envs ever replaces `Muscle002.cs` joint motors with physiological muscle models, FreeMusco provides the training approach.

---

### 9. Robotics & Humanoid Foundation Models

**What it is:** Training large-scale policies for real robots; lessons on physics fidelity, sim-to-real transfer, and energy efficiency transfer back to game characters.

**Marathon-envs relevance: LOW** — different hardware constraints, but agility and energy efficiency research is transferable.

| Paper | Year | Key Advance | Code |
| --- | --- | --- | --- |
| GR00T N1 | 2025 | NVIDIA foundation model for humanoid robots | [NVIDIA/Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) — Apache 2.0 |
| HOVER | 2025 | Versatile whole-body controller | [NVlabs/HOVER](https://github.com/NVlabs/HOVER) — Apache 2.0 |
| ASAP | 2025 | Sim-to-real transfer for humanoid skills | — |
| Expressive Whole-Body | 2024 | Expressive humanoid control | Public |

---

## How the Technique Families Connect

```
MoCap Data
    │
    ├─► Phase Extraction (PFNN/DeepPhase) ─────► Kinematic Controller
    │                                                    │
    ├─► Motion Matching (DReCon kinematic) ──────────────┤
    │                                                    │
    └─► Diffusion Generation (CAMDM, MDM) ───────────────┤
                                                         │
                                            Trajectory Targets
                                                         │
                                                         ▼
                                          Physics Feedback Controller (PPO)
                                                         │
                                    ┌────────────────────┴──────────────────┐
                                    │                                        │
                        Explicit Reward (DeepMimic)          Adversarial Reward (AMP)
                                          ▲                                  │
                                          │                         Skill Embedding
                                  [marathon-envs                (ASE → CALM → PULSE)
                                   current state]                             │
                                                                    Universal Prior
                                                                  (PHC → MaskedMimic)
```

**Reading the diagram:** Marathon-envs currently lives at "Physics Feedback Controller + Explicit Reward (DeepMimic)." The natural next steps are replacing the explicit reward with AMP, then building a skill embedding library on top.

---

## Relevance Summary

| Family | Marathon-Envs Relevance | Effort | Best Entry Point |
| --- | --- | --- | --- |
| Imitation Learning (DeepMimic) | ★★★ Already implemented | — | Already done |
| Responsive Control (DReCon) | ★★★ Preview exists | Low | Complete ControllerMarathonMan-v0 |
| Adversarial Style (AMP) | ★★★ Direct upgrade | Medium | Replace DeepMimic reward |
| Skill Embeddings (ASE) | ★★★ Major upgrade | High | Pre-train on large MoCap |
| Supervised Tracking (SuperTrack) | ★★★ Training speedup | Medium | Replace PPO with diff physics |
| Policy Adaptation (AdaptNet) | ★★ Style transfer | Medium | Roblox open-source |
| Phase Networks (PFNN/DeepPhase) | ★★ Kinematic layer | Medium | AI4Animation Unity C# |
| Motion Matching | ★★ Kinematic layer | Low | JLPM22/MotionMatching |
| Diffusion Control (CAMDM) | ★★ Future | High | ONNX demo available |
| Skill Libraries (CALM/MaskedMimic) | ★★ Long-term | Very High | NVIDIA ProtoMotions |
| Contact Interactions | ★ Future | Very High | After locomotion solved |
| Language Control | ★ Research | Very High | Not near-term |
| Robotics | ★ Inspiration | N/A | Read for physics insights |

---

## Key Research Groups

| Group | Affiliation | Key Papers | Primary Code |
| --- | --- | --- | --- |
| Xue Bin (Jason) Peng | UC Berkeley | DeepMimic, AMP, ASE | github.com/xbpeng |
| Zhengyi Luo | CMU → NVIDIA | PHC, PULSE, SimXR, Omnigrasp | github.com/ZhengyiLuo |
| NVIDIA Toronto AI Lab | NVIDIA | CALM, MaskedMimic, PADL, Kimodo | github.com/NVlabs, nv-tlabs |
| Sebastian Starke | EA → ETH | PFNN, DeepPhase, AI4Animation | github.com/sebastianstarke |
| Ubisoft La Forge | Ubisoft | DReCon, Learned MM, SuperTrack | github.com/ubisoft-laforge |
| Roblox Research | Roblox | AdaptNet | github.com/Roblox |
| Joan Llobera (JLPM22) | UPC | MotionMatching Unity 6 | github.com/JLPM22 |

---

## Collection Stats

- **Total PDFs:** 162 papers (2012–2026)
- **Papers with public code:** 85 official + 8 community ports
- **Primary venues:** SIGGRAPH (~55%), CVPR/ICCV/NeurIPS (~25%), other (~20%)
- **Full catalog:** `D:\Projetos\Physics Character Controller\Knowledge\Catalog.md`
- **Implementation index:** `D:\Projetos\Physics Character Controller\Knowledge\implementations.md`
- **Unity-specific analysis:** `D:\Projetos\Physics Character Controller\Knowledge\Unity-Game-Engine-Physics-Control-Finds.md`
