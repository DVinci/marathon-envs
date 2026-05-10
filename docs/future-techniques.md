# Future Techniques Roadmap

Ordered implementation roadmap for marathon-envs, based on deep study of 162 research papers. Each entry includes the paper it's based on, what it adds, prerequisites, estimated effort (weeks of focused work on the current i7-3770S + RTX 2070 Super rig), and Unity/ONNX feasibility.

Techniques are organized into three tiers based on implementation complexity and prerequisite depth. Complete Tier 1 before Tier 2, and so on.

---

## Tier 1 — Near-Term (Extend Current Marathon-Envs Directly)

These require no new training infrastructure. They extend or replace components that already exist in the codebase.

---

### T1-A: Complete DReCon Interactive Controller

**Paper:** DReCon (Bergamin et al., SIGGRAPH Asia 2019) — already in marathon-envs as preview  
**What it adds:** A real-time player-interactive physics character that responds to controller input with natural MoCap-derived motion  
**Current state:** `ControllerMarathonMan-v0` exists but the kinematic motion matching layer is a stub

**Implementation steps:**
1. Integrate [JLPM22/MotionMatching](https://github.com/JLPM22/MotionMatching) (Unity 6, MIT, 561 stars) as the kinematic layer — it provides real-time MoCap database query from player trajectory input
2. Connect its output (target pose per frame) to `StyleTransfer002Animator.cs`
3. Train the PPO feedback policy on diverse player input trajectories (random walk, direction changes, speed variations)
4. Test: character follows player controller input with natural gait and physics robustness

**Prerequisites:** None — uses existing marathon-envs PPO training  
**Effort:** ~2 weeks  
**ONNX:** Feedback MLP exports trivially; motion matching runs in C#  
**Benefit:** Turns marathon-envs from a benchmark into an interactive game character demo

---

### T1-B: AMP Adversarial Style Reward

**Paper:** AMP (Peng et al., SIGGRAPH 2021)  
**What it adds:** Replace the 7-component DeepMimic reward in `StyleTransfer002Agent.cs` with a learned adversarial discriminator. No more reward weight tuning.

**Implementation steps:**
1. Add a discriminator network (2-layer MLP, LSGAN loss) that reads (state_t, state_{t+1}) tuples
2. Alternate training: PPO policy update → discriminator update on real MoCap vs policy transitions
3. Reward becomes: `r = 0.5 × r_task + 0.5 × D(s_t, s_{t+1})`
4. r_task can be a simple forward velocity or upright reward — no pose distances needed
5. Reference: [Balint-H/modular-agents](https://github.com/Balint-H/modular-agents) has a Unity ML-Agents compatible AMP reward module

**Prerequisites:** T1-A helpful but not required  
**Effort:** ~3 weeks  
**ONNX:** Policy MLP unchanged; discriminator not needed at inference  
**Benefit:** Eliminates manual reward tuning. One discriminator covers all 6 style-transfer environments.

---

### T1-C: AdaptNet Latent Style Transfer

**Paper:** AdaptNet (Xu et al., Roblox — SIGGRAPH Asia 2023)  
**What it adds:** Adapt any trained marathon-envs policy to a new style (or small morphology change) in 10–30 minutes, without retraining from scratch.

**Implementation steps:**
1. Take any trained marathon-envs policy (Walker2d, MarathonMan, etc.)
2. Add a small latent modifier module (2-layer MLP) that injects style information into the policy's hidden state
3. Freeze the original policy; train only the modifier for 10–30 minutes on the target style
4. Result: the modified policy adopts the new style while keeping the original dynamics knowledge
5. Reference code: [Roblox/AdaptNet](https://github.com/Roblox/AdaptNet) — open source with pre-trained models

**Prerequisites:** Any trained marathon-envs policy  
**Effort:** ~1 week to integrate the modifier architecture  
**ONNX:** Full pipeline exports cleanly  
**Benefit:** New dance styles, new motion styles, or subtle morphology changes without expensive full retraining

---

### T1-D: Supervised Motion Tracking via Differentiable Physics (SuperTrack)

**Paper:** SuperTrack (Fussell et al., SIGGRAPH 2021)  
**What it adds:** Replace the PPO training loop with supervised learning through a learned differentiable physics world model. 3–5× faster convergence, no reward engineering.

**Implementation steps:**
1. Collect state transition data from existing Unity PhysX simulations (run current marathon-envs, log (state_t, action_t, state_{t+1}) tuples)
2. Train a world model W (5-layer MLP) to predict next_state from current_state + PD_targets
3. Train policy Π by backpropagating loss through W — no RL loop needed
4. Loss: pose + velocity + contact MSE against reference MoCap clips
5. Note: PyPhysX wraps the same PhysX SDK Unity uses — world model should generalize

**Prerequisites:** Existing marathon-envs Unity simulation + reference MoCap clips  
**Effort:** ~4 weeks (world model training is new infrastructure)  
**ONNX:** Both world model and policy are pure MLPs — export cleanly  
**Benefit:** 3–5× faster training. Walker2d trains in ~20 GPU hours vs current 100+ hours. No reward weight tuning.

---

## Tier 2 — Medium-Term (New Capabilities Requiring More Infrastructure)

These require either a larger MoCap dataset, new network architectures, or multi-stage training pipelines.

---

### T2-A: ASE Pre-Training for Skill Reuse

**Paper:** ASE (Peng et al., SIGGRAPH 2022)  
**What it adds:** A shared 64-dim latent skill space pre-trained on a large MoCap corpus. All 16 marathon-envs environments become downstream tasks using the same pre-trained low-level controller — new environments train in hours instead of days.

**Implementation steps:**
1. Acquire AMASS dataset (free, academic license, 40+ hours of MoCap) or use existing marathon-envs FBX clips
2. Train ASE pre-training stage: encoder q(z|s,s'), discriminator D, policy π(a|s,z) — uses IsaacGym or equivalent
3. For each marathon-envs environment: train a high-level policy ω(z|s,goal) using the frozen pre-trained low-level controller
4. New environments only require high-level policy training (~hours)
5. Reference: [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) — IsaacGym; adapt training loop

**Prerequisites:** Large MoCap dataset (AMASS recommended), GPU with ≥8GB VRAM (RTX 2070 Super qualifies, but training will be slow — cloud VM recommended)  
**Effort:** ~6–8 weeks total (pre-training is the heavy part; downstream tasks are fast)  
**ONNX:** Encoder offline + policy online — both pure MLPs  
**Benefit:** Dramatic reduction in per-environment training time. Scalable to dozens of new environments.

---

### T2-B: Phase-Based Kinematic Controller (DeepPhase / AI4Animation)

**Papers:** PFNN (Holden 2017), DeepPhase (Starke 2022)  
**What it adds:** A learned phase manifold that drives a kinematic controller with smooth, speed-variable, terrain-aware locomotion — usable as the kinematic layer in DReCon (T1-A).

**Implementation steps:**
1. Clone [sebastianstarke/AI4Animation](https://github.com/sebastianstarke/AI4Animation) — it contains PFNN, DeepPhase, and other phase-based controllers with Unity C# implementations
2. Start with the SIGGRAPH_2022 (DeepPhase) folder — it includes PyTorch training code + Unity demo
3. Train PAE (Periodic Autoencoder) on existing marathon-envs FBX animation files to extract phase channels
4. In Unity: use the PAE-derived phase features to drive joint target generation
5. Connect to DReCon physics feedback policy (T1-A) for a complete hybrid system

**Prerequisites:** T1-A (DReCon)  
**Effort:** ~3–4 weeks  
**ONNX:** PFNN exports trivially; DeepPhase encoder can be precomputed offline  
**Benefit:** Natural gait cycles with automatic phase adaptation. Smooth speed variation (walk → run → sprint) without retraining.

---

### T2-C: PHC Fall Recovery System

**Paper:** PHC (Luo et al., ICCV 2023)  
**What it adds:** Explicit fall state detection and a dedicated recovery controller that returns the agent to standing from any fall state. Episodes never need to reset in interactive scenarios.

**Implementation steps:**
1. Add fall state detection to `BodyManager002.cs`:
   - Monitor head height relative to standing height threshold
   - Monitor contact patterns (body touching ground outside normal gait)
2. Train a separate recovery controller (2-layer MLP) with reward for returning to standing posture
3. Gate between normal locomotion policy and recovery policy based on fall state flag
4. Progressive training: train walking first → intentionally perturb → train recovery → fine-tune jointly

**Prerequisites:** Any trained locomotion policy (Walker2d, MarathonMan)  
**Effort:** ~2–3 weeks  
**ONNX:** Two separate MLPs — gate logic in C#  
**Benefit:** Robust interactive character that recovers from player-applied forces, collisions, and environmental perturbations instead of resetting.

---

### T2-D: CAMDM as Kinematic Layer

**Paper:** CAMDM (SIGGRAPH 2024)  
**What it adds:** A compact 20MB diffusion-based kinematic motion generator covering 100 styles, running at 60+ FPS. Official Unity ONNX demo included.

**Implementation steps:**
1. Clone [AIGAnimation/CAMDM](https://github.com/AIGAnimation/CAMDM) — includes pre-trained ONNX models and Unity C# integration
2. Load the CAMDM ONNX model in Unity Sentis
3. At each physics step: CAMDM generates a kinematic target pose conditioned on past motion (20-frame window) + player trajectory input
4. Connect CAMDM output to the DReCon physics feedback controller (T1-A)
5. Result: diffusion-quality motion with physics robustness at 60+ FPS

**Prerequisites:** T1-A (DReCon physics layer)  
**Effort:** ~2 weeks (most code already in CAMDM repo)  
**ONNX:** Fully supported — official ONNX + Sentis demo included  
**Benefit:** Most immediately practical integration of a generative motion model. Eliminates the need for a large runtime MoCap database.

---

## Tier 3 — Long-Term (Significant R&D Investment)

These require either very long training runs (days to weeks on high-end hardware), multi-stage pipelines, or substantial new infrastructure. They represent the state of the art in 2024–2025.

---

### T3-A: PULSE Universal Prior

**Paper:** PULSE (Luo et al., ICLR 2024 Spotlight)  
**What it adds:** A distilled universal motion VAE that makes downstream task training 10–20× faster. Trained once; reused for all future environments.

**Implementation steps:**
1. Train a teacher policy on diverse marathon-envs environments (Walker2d, MarathonMan, terrain variants)
2. Collect teacher trajectories as distillation data
3. Train student VAE: encoder q(z|motion) + conditional prior p(z|proprioception) + decoder
4. All future downstream tasks train hierarchical RL in latent space (~48 hours per task)
5. Code: [ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE) — dual backend (IsaacGym / MuJoCo MJX)

**Prerequisites:** T2-A (trained teacher policy), large MoCap dataset  
**Effort:** ~8–12 weeks (distillation training is substantial)  
**ONNX:** Encoder + conditional prior — pure MLPs, fully exportable  
**Benefit:** Permanent reduction in per-environment training cost. Pay once, reuse indefinitely.

---

### T3-B: C·ASE Multi-Style Skill Library

**Paper:** C·ASE (SIGGRAPH 2023)  
**What it adds:** A conditional skill embedding system where the 6 style-transfer environments (Walking, Running, JazzDancing, MMAKick, PunchingBag, Backflip) become independently trained subsets that compose under a single high-level controller.

**Implementation steps:**
1. Cluster existing marathon-envs FBX animations into semantic skill subsets
2. Train per-subset conditional policies (parallelizable — one per GPU)
3. Train adversarial discriminator per subset to ensure motion quality
4. Train high-level controller mapping game input → skill selection
5. Result: single agent that switches fluently between all 6 styles

**Prerequisites:** T1-B (AMP discriminator), large enough GPU cluster for parallel training  
**Effort:** ~8–10 weeks  
**ONNX:** Conditional policy + embedding table — feasible with care  
**Benefit:** Interactive game character that transitions smoothly between all known behaviors based on gameplay state.

---

### T3-C: MaskedMimic Unified Framework

**Paper:** MaskedMimic (NVIDIA, SIGGRAPH 2024)  
**What it adds:** A single model that handles all marathon-envs control modes: autonomous locomotion (unconstrained mask), DeepMimic tracking (full-constraint mask), DReCon hybrid (partial mask), and VR control (head+hands constrained).

**Implementation steps:**
1. Implement the two-stage training pipeline: PPO full-body tracker → Conditional VAE with masking
2. Define masks for each marathon-envs environment (classical → unconstrained, style transfer → full, interactive → partial)
3. Train unified model on AMASS + existing FBX clips
4. One ONNX export per mask configuration (or runtime mask input)
5. Code: [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) — NVIDIA's current framework

**Prerequisites:** T2-A (ASE or equivalent pre-training), AMASS dataset, 8-GPU training setup (weeks of training)  
**Effort:** ~12–16 weeks  
**ONNX:** VAE architecture — highly feasible  
**Benefit:** Eliminates all 16 separate environment policies. One universal model with mask control for all scenarios.

---

### T3-D: PARC Automatic Terrain Behavior Synthesis

**Paper:** PARC (SIGGRAPH 2025)  
**What it adds:** Automatically generate terrain-adaptive behaviors (vault, climb, jump) beyond the original MoCap dataset via iterative diffusion + RL augmentation.

**Implementation steps:**
1. Collect 14–20 minutes of terrain-relevant MoCap (or use FBX clips)
2. Train terrain-conditioned diffusion motion generator
3. Run 4 augmentation iterations: generate → track with PPO → add successful motions to dataset
4. Terrain variants (TerrainHopper, TerrainWalker, etc.) train on augmented dataset
5. Code: [msooloo/PARC](https://github.com/msooloo/PARC)

**Prerequisites:** T2-D (CAMDM or equivalent motion generator)  
**Effort:** ~10–12 weeks (1 month training per iteration × 4 iterations; parallelizable)  
**ONNX:** Tracker policy exports; diffusion generator is offline  
**Benefit:** Terrain variants get dramatically richer behavior without manual MoCap collection. Gap jumping, ledge climbing, and compound parkour emerge automatically.

---

## Implementation Priority Summary

| ID | Technique | Effort | Impact | Start After |
| --- | --- | --- | --- | --- |
| T1-A | Complete DReCon | 2 weeks | High — interactive character | Nothing |
| T1-B | AMP adversarial reward | 3 weeks | High — no reward tuning | Nothing |
| T1-C | AdaptNet style transfer | 1 week | Medium — fast style adaption | Trained policy |
| T1-D | SuperTrack diff-physics | 4 weeks | High — 3-5× faster training | Nothing |
| T2-A | ASE skill pre-training | 6–8 weeks | Very High — reuse across all envs | Large MoCap dataset |
| T2-B | DeepPhase kinematic layer | 3–4 weeks | Medium — natural gaits | T1-A |
| T2-C | PHC fall recovery | 2–3 weeks | High — robust interactive control | Trained policy |
| T2-D | CAMDM kinematic layer | 2 weeks | High — best ONNX entry point | T1-A |
| T3-A | PULSE universal prior | 8–12 weeks | Very High — permanent cost reduction | T2-A |
| T3-B | C·ASE skill library | 8–10 weeks | High — multi-style composition | T1-B |
| T3-C | MaskedMimic unified | 12–16 weeks | Very High — replaces all 16 envs | T2-A + T3-B |
| T3-D | PARC terrain synthesis | 10–12 weeks | Medium — automatic terrain behaviors | T2-D |

**Recommended starting point:** T1-A (DReCon completion) and T1-B (AMP reward) can run in parallel — they are independent and both deliver immediate, visible improvements.

---

## Hardware Considerations

All Tier 1 and Tier 2 tasks are feasible on the current i7-3770S + RTX 2070 Super rig. Tier 3 tasks (especially T3-A PULSE and T3-C MaskedMimic) require GPU-accelerated simulation (IsaacGym) with multi-GPU setups — these are candidates for cloud VM training (see [cloud-training.md](cloud-training.md)).

For cloud training: use CPU-heavy VMs for PhysX simulation (c5.9xlarge, 36 vCPUs) not GPU VMs — the bottleneck is physics, not the neural network. See cloud-training.md for details.
