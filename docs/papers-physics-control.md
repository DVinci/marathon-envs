# Physics-Based Control Papers — Deep Study

Detailed technical summaries of the highest-priority research papers for marathon-envs: those directly extending or improving physics-based RL character control with active ragdolls. All PDFs are in `D:\Projetos\Physics Character Controller\Knowledge\Character Motion\`.

**Inclusion criteria:** Uses RL with physics simulation; targets humanoid locomotion; output is joint torques/targets compatible with PhysX; ONNX export is feasible.

---

## 1. DeepMimic — Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills

**Peng et al. — SIGGRAPH 2018**

**Core idea:** Learn physics-based character skills by imitating MoCap clips frame-by-frame. Reference-state initialization (RSI) samples the episode start from a random frame in the clip, forcing the policy to recover from any mid-motion state.

**What makes it special:** RSI is the key insight. Without it, the agent must navigate from an initial standing pose to the correct mid-motion state before learning the target motion — long-horizon credit assignment makes this intractable. RSI bypasses this entirely by starting at any point in the reference.

**Reward function (7 components):**
```
r = w_p × exp(-k_p × ||q_sim - q_ref||²)    # joint pose
  + w_v × exp(-k_v × ||dq_sim - dq_ref||²)  # joint velocity
  + w_e × exp(-k_e × ||e_sim - e_ref||²)    # end-effector positions
  + w_r × exp(-k_r × ||root_sim - root_ref||²) # root orientation/velocity
```

**Training:** PPO + GAE, BulletPhysics, ~30 CPU-hours per skill, standard MoCap libraries.

**Marathon-envs status:** Directly implemented in [StyleTransfer002Agent.cs](UnitySDK/Assets/MarathonEnvs/Scripts/ActiveRagdoll002/StyleTransfer002Agent.cs) — 7 exponential reward components, FBX reference animations in `Assets/MarathonEnvs/Animations/`.

**What AMP (below) enables that DeepMimic doesn't:** AMP eliminates the need to tune the 7 reward weights by learning the reward from data.

**Code:** [xbpeng/DeepMimic](https://github.com/xbpeng/DeepMimic) — BulletPhysics/TF1; conceptually portable.

---

## 2. AMP — Adversarial Motion Priors for Stylized Physics-Based Character Control

**Peng et al. — SIGGRAPH 2021**

**Core idea:** Replace DeepMimic's explicit pose-matching reward with an adversarial discriminator trained to distinguish reference MoCap state transitions from policy-generated transitions. The discriminator output becomes the style reward automatically.

**What makes it special:** The discriminator learns what "natural motion" looks like purely from data. No manual reward engineering. A single AMP discriminator trained on thousands of MoCap clips provides a rich, general style reward — the policy spontaneously adopts natural motion while pursuing task goals.

**Architecture:**
- Policy: 2-layer MLP (1024-512 units, ReLU), 56-dim PD target offsets
- Discriminator: 2-layer MLP, least-squares GAN (LSGAN) loss
- Reward: `r = w_task × r_task + w_style × D(s_t, s_{t+1})`
- w_task = w_style = 0.5 for all tested tasks

**Training:**
- BulletPhysics at 1.2 kHz simulation
- MoCap dataset: up to 1136 hours of clips
- PPO + GAIL objective, 30–140 CPU hours per model

**What this enables vs current marathon-envs:**
- Style from unstructured databases — no need to align FBX clips frame-by-frame
- Arbitrary task + style composition (run fast while looking like zombie motion)
- No per-skill reward weight tuning
- Works on non-humanoid morphologies (quadrupeds, custom rigs)

**ONNX export:** Policy is a pure MLP — trivially exported. Discriminator not needed at inference.

**Code:** [xbpeng/amp](https://github.com/xbpeng/amp) — BulletPhysics; AMP is also the foundation of [Balint-H/modular-agents](https://github.com/Balint-H/modular-agents), a Unity ML-Agents module library.

---

## 3. ASE — Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters

**Peng et al. — SIGGRAPH 2022**

**Core idea:** Extend AMP by learning a latent skill space Z alongside the motion prior. The low-level policy is conditioned on a latent code z — different z values produce different skills. An encoder maps state transitions to latent codes. A high-level policy selects z to accomplish downstream tasks.

**What makes it special:** Skills trained once can be reused across unlimited downstream tasks. Training a new task only requires training the high-level policy (hours), not retraining the entire controller (days). The 64-dim latent space organizes skills semantically.

**Two-stage training:**

Stage 1 — Pre-training:
```
Encoder q(z | s_t, s_{t+1})  — maps transitions to skill latent
Discriminator D(s_t, s_{t+1}) — distinguishes dataset vs policy
Policy π(a | s, z)             — conditioned on skill z
Diversity objective            — KL divergence encourages different z → different behaviors
```

Stage 2 — Downstream task training:
```
High-level policy ω(z | s, goal) — selects which skill z to use
Low-level policy π frozen        — executes z
Training time: hours vs days (80-90% sample reduction)
```

**Training setup:**
- IsaacGym (NVIDIA GPU sim), 187 motion clips (~30 minutes)
- Pre-training: 10 days on V100, 10 billion samples
- Skill space: Z ∈ R^64 (unit hypersphere, prevents mode collapse)

**Key results:**
- 30+ distinct locomotion styles discovered automatically
- 0.31s mean recovery time from perturbations
- 80-90% fewer samples needed for downstream tasks

**Marathon-envs application:**
- Pre-train ASE on existing FBX animations + AMASS dataset
- Each of the 16 marathon-envs environments becomes a downstream task using the same pre-trained low-level controller
- New environments only require high-level policy training (hours, not days)

**ONNX export:** All three networks are pure MLPs — exported cleanly. Encoder runs offline to get skill z; policy runs online at inference.

**Code:** [nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) — IsaacGym; superseded by [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) but still functional.

---

## 4. DReCon — Data-Driven Responsive Control of Physics-Based Characters

**Bergamin et al. — SIGGRAPH Asia 2019**

**Core idea:** Decouple trajectory planning from balance control. A motion matching system finds the best-matching MoCap frame for current user input (kinematic layer). A small PPO policy learns only to output corrective PD offsets to track that kinematic target (physics layer).

**What makes it special:** The physics policy only needs to learn small corrective offsets (~25-dim) instead of the full locomotion problem. This makes training fast and robust. Motion matching provides high-quality natural motion without RL having to discover trajectories from scratch.

**Architecture:**
- Kinematic layer: Motion matching on 10 minutes of unstructured MoCap, 340μs lookup time
- Physics layer: 2-layer MLP (128 units), 25-dim PD offsets
- Observation: 110-dim (CM velocity, pose, joint velocities, contact labels, trajectory features)
- Training: PPO, 8 parallel environments, hours to train

**Key results:**
- Heading change response: 1.2s (vs 2.0s DeepMimic, 5.1s skill graph)
- Total runtime: 340μs (game-engine ready)
- Robust to 0.1–8kg cube impacts at 5m/s
- Walking, running, crouching with a single policy

**Marathon-envs status:** Already implemented as `ControllerMarathonMan-v0` (preview). The architecture uses `MarathonTestBedDecision.cs` and `StyleTransfer002Animator.cs` for the kinematic layer.

**Completing the DReCon implementation:**
1. Replace the motion matching stub with [JLPM22/MotionMatching](https://github.com/JLPM22/MotionMatching) (Unity 6 native, MIT)
2. Train the physics feedback policy on diverse user input trajectories
3. Result: interactive game character responding to controller input in real-time

**ONNX export:** 128-unit MLP — trivially exported. Motion matching runs in C# without a neural network.

---

## 5. SuperTrack — Motion Tracking for Physically Simulated Characters using Supervised Learning

**Fussell et al. — SIGGRAPH 2021**

**Core idea:** Train a motion tracking policy via supervised learning through a learned differentiable physics simulator (world model), rather than RL. Gradients flow backward through the world model to the policy, enabling direct optimization without reward shaping.

**What makes it special:** No RL = no reward engineering, no sample inefficiency, no hyperparameter tuning. The loss is simply pose + velocity matching. Convergence is 3–5× faster than PPO. The world model is trained on PyPhysX — the same PhysX SDK that Unity uses.

**Architecture:**
- World model W: 5-layer MLP (1024 units, ELU), predicts next rigid body state
- Policy Π: 5-layer MLP (1024 units, ELU), maps state → 56-dim PD offsets
- Window-based: Policy sees 32 future frames; world model trained on 8-frame windows

**Loss function:**
```
L = w_pose × ||pos_sim - pos_ref||²
  + w_vel  × ||vel_sim - vel_ref||²
  + w_ang  × ||ang_sim - ang_ref||²
  + w_contact × contact_penalty
```

**Training setup:**
- PyPhysX (Python wrapper around PhysX SDK — same engine as Unity)
- LaFAN MoCap database (~6.5 hours: walk, run, dance, parkour)
- Training: 20–50 GPU hours on single GTX 1070 (vs 300+ for PPO)

**Key results:**
- 80% of episodes last >1 minute
- 3–5× faster convergence than PPO
- 180μs per frame at inference
- World model generalizes across character morphologies

**Marathon-envs application:**
1. Collect simulation data from Unity PhysX by running existing marathons environments
2. Train a world model on that PhysX data
3. Replace PPO training loop with supervised gradient descent through the world model
4. Reference data: existing FBX clips + AMASS (freely available, no license issues)
5. Result: faster training, no reward weight tuning, better contact generalization

**Note on PhysX compatibility:** PyPhysX wraps the same PhysX SDK Unity uses. World model trained on PyPhysX data should generalize to Unity PhysX with minimal fine-tuning.

**Code:** Academic release (Ubisoft La Forge). Architecture fully documented in paper.

---

## 6. PHC — Perpetual Humanoid Control for Real-time Simulated Avatars

**Luo et al. — ICCV 2023**

**Core idea:** Train multiplicative primitive behaviors (idle, walk, run, jump) as separate sub-policies, gated by a learned selector. A fall-state detector triggers a dedicated recovery controller when the character falls. Runs in real-time at 32 FPS.

**What makes it special:** Fall recovery is built in — the controller never stays stuck in a failed state. The multiplicative primitive architecture naturally encodes behavioral diversity without a skill embedding. Video imitation (from MediaPipe 2D poses) removes MoCap pipeline dependency.

**Architecture:**
- 4 primitive sub-policies × 2-layer MLP (256 units)
- Gating network: selects active primitive based on state
- Fall detector: monitors head height + contact flags → triggers recovery controller
- Observation: 200-dim (joint angles, velocities, contact states, reference pose)
- Action: 28-dim normalized joint target angles
- Training: PPO + IsaacGym, 4096 parallel envs, 8× A100 GPUs

**Key results:**
- 32 FPS real-time on RTX 3090, 28.8MB model
- Mean pose error: 0.10m on CMU MoCap
- Fall recovery: 92% success rate from unexpected perturbations
- Video imitation: Reproduces motion from MediaPipe 2D pose estimates

**Marathon-envs application:**
- `BodyManager002.cs` already tracks head height and contact flags — add fall-state detection
- Train a recovery controller to return agent to standing from any fall
- This eliminates episode resets in interactive scenarios — the agent recovers instead
- Progressive curriculum: train walk/run/jump primitives → fine-tune jointly

**ONNX export:** MLP gating + 4 primitive MLPs — all trivially exported.

**Code:** [ZhengyiLuo/PHC](https://github.com/ZhengyiLuo/PHC) — IsaacGym, actively maintained 2025.

---

## 7. PULSE — Universal Humanoid Motion Representations for Physics-Based Control

**Luo et al. — ICLR 2024 (Spotlight)**

**Core idea:** Distill a large pre-trained physics imitator (PHC+) into a compact conditional VAE that captures a universal motion latent space. A conditional prior p(z|proprioception) ensures sampled motions are physically feasible given the current state. Downstream task controllers train in latent space — 10–20× faster than from scratch.

**What makes it special:** The conditional prior is the key. By conditioning the latent distribution on proprioception, the VAE guarantees all sampled motions respect current physics state. Downstream tasks take 48 hours instead of weeks, because the universal prior already encodes how to move naturally.

**Architecture:**
- Teacher: PHC+ (large imitator on CMU MoCap + video)
- Student VAE: Encoder q(z|motion) → z ∈ R^32 → Decoder p(motion|z, proprioception)
- Conditional prior: MLP p(z|proprioception) — feasibility guarantee
- Downstream: Hierarchical RL selects z; decoder executes

**Key results:**
- Downstream training: 48 hours per task (vs weeks from scratch)
- Outperforms ASE and CALM on 7 diverse tasks
- 32-dim latent captures 95% of motion diversity
- Partial zero-shot transfer to unseen terrains

**Marathon-envs application:**
1. Train a teacher policy on existing marathon-envs environments (Walker2d teacher is already trained)
2. Distill teacher into VAE — becomes the universal prior
3. All 16 marathon-envs train downstream controllers in latent space (~48 hours each)
4. New environments (new sports, terrain) require only latent-space policy training

**ONNX export:** VAE encoder + conditional prior are pure MLPs. Latent sampling uses standard Gaussian — ONNX compatible.

**Code:** [ZhengyiLuo/PULSE](https://github.com/ZhengyiLuo/PULSE) — IsaacGym / MuJoCo MJX, MIT license.

---

## 8. C·ASE — Learning Conditional Adversarial Skill Embeddings for Physics-based Characters

**SIGGRAPH 2023**

**Core idea:** Divide the MoCap dataset into semantically homogeneous subsets (walking, running, jumping, combat) and train a conditional low-level policy per subset. Adversarial discriminators enforce motion quality within each subset. Skill embeddings (8–32-dim vectors) are learned per subset and become the action space for a high-level controller.

**What makes it special:** Training per subset avoids mode collapse — different motions don't compete for the same latent code (the main weakness of ASE). The result is cleaner, more interpretable skill boundaries. 91% of the reference motion dataset is covered.

**Architecture:**
- Per-skill conditional policy: π(a|s, z) — 28-dim joint targets
- Per-subset discriminator: ensures each policy matches its data distribution
- High-level controller: selects z based on user input or task goal
- Skill embedding: 8–32-dim per subset, interpretable and interpolatable

**Training:**
- 87 motion clips (SwordShield example dataset)
- Stage 1: Train per-subset conditional policies (parallelizable across GPUs)
- Stage 2: Train high-level controller with RL on task reward
- Stage 3: Interactive inference — user input → skill selector → policy

**Key results:**
- 91% motion coverage from 87 clips
- Smooth skill transitions (<0.5s between skills)
- 10–15 distinct skills discovered without manual annotation

**Marathon-envs application:**
- The 6 style-transfer environments (Walking, Running, JazzDancing, MMAKick, PunchingBag, Backflip) map naturally to C·ASE skill subsets
- Train 6 conditional policies (parallelizable) + one high-level selector
- Result: single agent that switches fluently between all 6 styles based on game input

**ONNX export:** Conditional policy MLP + embedding lookup — feasible (7/10). Embedding table must be serialized into the graph.

---

## 9. MaskedMimic — Unified Physics-Based Character Control Through Masked Motion Inpainting

**NVIDIA — SIGGRAPH 2024**

**Core idea:** Frame character control as motion inpainting: given constraints on some body parts, generate physically plausible motion for the unconstrained parts. A binary mask specifies which joints track reference motion vs. generate freely. One model handles full-body tracking, VR control (head+hands tracked), path following, and text-guided motion.

**What makes it special:** One model replaces 6 task-specific models. The mask determines the control mode — fully constrained → DeepMimic-style tracking; partially constrained → DReCon-style hybrid; unconstrained → autonomous locomotion. No task-specific retraining.

**Architecture:**
- Stage 1: Full-body PPO tracker (baseline motion tracking controller)
- Stage 2: Conditional VAE with motion inpainting masks
  - Encoder: observed motion → 32-dim latent z
  - Decoder: full-body motion given partial constraints + binary mask
  - Mask: specifies which joints have reference constraints
- Reference data: AMASS + HumanML3D

**Key results:**
- Handles 6+ distinct control tasks with one model
- VR: hands + head tracked, body generates naturally
- Text-to-motion: motion matching text descriptions on HumanML3D
- 30+ FPS real-time on RTX 3080

**Marathon-envs application (long-term):**
- Classical environments → unconstrained mask (autonomous locomotion)
- Style transfer environments → full-constraint mask (track MoCap clip)
- Interactive control → partial mask (track user controller positions)
- VR → head + hands constrained, lower body free
- A single MaskedMimic model would subsume all 16 marathon-envs environments

**ONNX export:** VAE architecture — highly feasible (9/10). Masking is simple tensor operations.

**Code:** [NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions) — NVIDIA's current research framework, actively developed 2025.

---

## Quick Reference

| Paper | Year | Key Advance | Training Cost | ONNX | Priority |
| --- | --- | --- | --- | --- | --- |
| DeepMimic | 2018 | RSI enables complex skill learning | 30 CPU hrs/skill | ✓ | Already done |
| DReCon | 2019 | Motion matching + small feedback policy | Hours | ✓ | In progress |
| SuperTrack | 2021 | Supervised diff-physics, 3-5× faster | 20–50 GPU hrs | ✓ | Tier 1 |
| AMP | 2021 | Discriminator replaces reward engineering | 30–140 CPU hrs | ✓ | Tier 1 |
| ASE | 2022 | Reusable skill latent space | 10 days (V100) | ✓ | Tier 1 |
| PHC | 2023 | Primitives + fall recovery, real-time | 15–20 days (A100) | ✓ | Tier 2 |
| C·ASE | 2023 | Subset training, clean skill boundaries | 2–3 weeks | ✓ | Tier 2 |
| PULSE | 2024 | Distilled universal prior, fast downstream | 1 wk + 48 hr/task | ✓ | Tier 2 |
| MaskedMimic | 2024 | Single model for all control tasks | 15–20 days | ✓ | Tier 3 |
