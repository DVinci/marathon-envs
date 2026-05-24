# Motion Synthesis Papers — Deep Study

Detailed summaries of MEDIUM-relevance papers for marathon-envs. These cover kinematic motion generation, phase-based control, policy adaptation, and diffusion methods. They are most valuable as a **kinematic layer feeding targets into the physics controller** (the DReCon pattern) or as future training speedups.

All PDFs are in `D:\Projetos\Physics Character Controller\Knowledge\Character Motion\`.

---

## 1. Phase-Functioned Neural Networks for Character Control (PFNN)

**Holden et al. — SIGGRAPH 2017**

**Core idea:** Encode a phase parameter (current point in the locomotion cycle, 0→1) and use it to index into a cubic spline of network weights. Different phase values produce different network configurations — the network adapts its computation to where in the stride cycle the character is.

**What makes it special:** Phase parameterization compresses the weight space by 4× compared to a flat network. Motion is temporally coherent without RNNs. Terrain-awareness comes from sampling height at contact points. The whole system runs in 2ms per frame.

**Architecture:**
- Input: Phase p (scalar), trajectory window (desired positions/directions/velocities 12 frames ahead), terrain height samples, gait labels
- Output: Next-frame joint positions/angles
- Weights W(p) = cubic Catmull-Rom spline of 4 control point matrices
- 10 precomputed sample positions along phase space → interpolation at runtime
- Network: 1 input + 2 hidden (512 units) + 1 output, no RNN

**Training setup:**
- Kinematic only (no physics simulation)
- MoCap: walking, running, jumping, crouching at varied speeds/directions
- Terrain: automatic height-fitting with RBF mesh editing
- Training: 30 hours, Adam optimizer

**Key results:**
- Runtime: 2ms per frame (10 precomputed phase positions)
- Memory: 10 MB (vs 1000 MB for full alternatives)
- Terrain tracking: 8–26cm average error across diverse paths
- Generalization: Works on rough terrain, climbing, jumping without retraining

**Marathon-envs application:**
PFNN is most valuable as the kinematic trajectory generator in the DReCon pattern:
- PFNN generates the target pose sequence for each frame (given user input + terrain)
- The existing PPO physics controller (`MarathonAgent`) learns to track those targets with corrective offsets
- Result: terrain-aware locomotion with natural gait cycles, driven by user input

**ONNX export:** Very high feasibility. Phase function is cubic spline interpolation + linear layers — pure tensor operations, trivially exported.

**Code:** [sebastianstarke/AI4Animation](https://github.com/sebastianstarke/AI4Animation) — MIT license, Unity C# implementation included with ONNX export.

---

## 2. DeepPhase — Periodic Autoencoders for Learning Motion Phase Manifolds

**Starke et al. — SIGGRAPH 2022**

**Core idea:** Automatically learn the phase structure of motion from raw data using a periodic autoencoder (PAE). Instead of manually computing a single phase value, the PAE discovers multiple periodic channels, each capturing a different aspect of motion (e.g., left stride, right stride, arm swing).

**What makes it special:** Phase is no longer hand-engineered — it emerges from data. The periodic activation functions enforce temporal coherence. The resulting multi-dimensional phase manifold enables smooth interpolation between motion styles, speed-invariant generation, and music-driven animation.

**Architecture:**
- Input: 3D joint velocities across all joints (121 timesteps)
- Encoder: Convolutional layers → FFT-based phase extraction
- Phase parameters per channel: Amplitude A, Frequency F, Phase offset Φ, Bias B
- Reconstruction: `L = A × sin(2π(F × T - Φ)) + B` per channel
- Phase manifold: M-dimensional (M=10 for bipedal locomotion)
- Decoder: Reconstructs motion trajectories from phase parameters

**Training setup:**
- IsaacGym, 6 Hz control frequency
- 5 datasets: biped locomotion (2.2M frames), quadruped (63k), stylized (726k), dance (33k), football (147k)
- Training: 12–48 GPU hours on RTX 3080

**Key results:**
- Discovers natural motion periodicity without labels
- Foot skating reduced significantly vs alternatives
- One-shot style transfer: Apply learned walking phase to running task
- Dance synthesis: Discovers choreography patterns, generates novel variations
- Beat-synchronized animation via frequency modulation

**Marathon-envs application:**
- Replace hand-extracted phase in StyleTransfer002Animator with learned phase from PAE
- The 6 learned phase channels capture stride timing, arm swing, torso rotation independently
- Enables smooth speed variation (walk → run → sprint) via frequency interpolation
- Applicable to procedural terrain locomotion: phase adapts automatically to contact timing

**ONNX export:** Partially feasible. Core encoder/decoder MLPs export trivially. FFT phase extraction needs custom ops or offline precomputation — recommended approach: precompute phase parameters offline per MoCap frame, use learned policy online.

**Code:** [sebastianstarke/AI4Animation](https://github.com/sebastianstarke/AI4Animation) — SIGGRAPH_2022 subfolder, Unity + PyTorch, actively maintained 2025.

---

## 3. Learned Motion Matching

**Harvey et al. — SIGGRAPH 2020 (Ubisoft La Forge)**

**Core idea:** Replace the expensive MoCap database search in motion matching with three small neural networks: a Decomposer (compresses motion to latent), a Stepper (predicts next latent frame), and a Projector (finds best database match in latent space). Database footprint drops from 150MB to 10MB.

**What makes it special:** Neural approximation makes motion matching scalable. The Stepper enables autonomous generation without a database lookup — it predicts the next motion frame directly from the current latent state. Sub-millisecond response time. Generalizes across datasets after training.

**Architecture:**
- Decomposer D: 5-layer MLP (512 units), maps motion → 32-dim latent z + feature vector x
- Stepper S: 4-layer MLP (512 units), predicts (z_{t+1}, x_{t+1}) from (z_t, x_t)
- Projector P: 6-layer MLP (512 units), maps feature x → nearest latent in database
- Total: ~3M parameters, ~10MB

**Training setup:**
- PyPhysX (PhysX SDK — same as Unity)
- LaFAN database: 1.44M frames (walk, run, dance, parkour, interaction)
- Training: 20–50 GPU hours per network, Adam optimizer

**Key results:**
- Memory: 10MB (vs 150MB uncompressed database)
- Latency: 62μs per frame (interactive real-time)
- User study: preferred over basic motion matching in 13/20 trials
- Generalizes: Trained on LaFAN, transfers to new dataset at ~80% quality

**Marathon-envs application:**
The Stepper network is directly useful as a standalone motion generator for the DReCon kinematic layer:
1. Train Stepper on existing FBX animation clips (walk, run, dance, etc.)
2. Stepper generates next kinematic target frame given current frame + user trajectory input
3. Physics PPO controller tracks those kinematic targets with corrective offsets
4. Result: interactive character without a large MoCap database at runtime

**ONNX export:** All three networks are pure MLPs. Stateful inference (Stepper maintains latent z across frames) is manageable with ONNX sequential execution.

**Code:** [ubisoft-laforge/learned-motion-matching](https://github.com/ubisoft-laforge/learned-motion-matching) — Ubisoft La Forge research release.

---

## 4. CAMDM — Taming Diffusion Probabilistic Models for Character Control

**SIGGRAPH 2024**

**Core idea:** Adapt diffusion probabilistic models for real-time character control via Conditional Autoregressive Motion Diffusion. Key contributions: Separate Condition Tokenization (SCT) for better conditioning, Classifier-Free Guidance on Past Motion (CFG-PM) for temporal consistency, Heuristic Future Trajectory Extension (HFTE) for lookahead. Achieves 60+ FPS with only 8 diffusion steps.

**What makes it special:** Real-time diffusion at 60+ FPS on a consumer GPU (RTX 3060). Previous diffusion models required hundreds of steps and ran offline. The compact 20MB model covers 100 motion styles without per-style finetuning. Official Unity ONNX demo is included in the repository.

**Architecture:**
- Transformer encoder-decoder with SCT mechanism for condition injection
- Conditional on: past motion frames (20-frame lookback) + task goal
- CFG-PM: Classifier-free guidance specifically for past motion temporal consistency
- HFTE: Heuristics for predicting future trajectory beyond input window
- 8 diffusion steps → 60+ FPS on RTX 3060
- Model size: 20MB

**Training setup:**
- 100STYLE motion capture dataset (4M frames)
- Diffusion loss across 8 timesteps
- Kinematic output — no physics simulation during training

**Key results:**
- 60+ FPS real-time on RTX 3060
- 20MB model size (mobile/VR deployable)
- 100 diverse motion styles covered
- Mean pose error <0.15m on style-following tasks

**Marathon-envs application:**
CAMDM generates high-quality kinematic motion — use it as the DReCon kinematic layer:
1. Load CAMDM ONNX model in Unity Sentis (demo code already available in repo)
2. At each physics step: CAMDM generates a kinematic target pose conditioned on past motion + user input
3. Physics PPO controller tracks the CAMDM target with small corrective offsets
4. Result: diffusion-quality motion with physics robustness, running at 60+ FPS

This is the most immediately practical integration path — CAMDM already has a Unity ONNX demo.

**ONNX export:** Explicitly documented, fully supported. The official repo includes Unity Sentis integration code.

**Code:** [AIGAnimation/CAMDM](https://github.com/AIGAnimation/CAMDM) — includes trained ONNX models and Unity C# demo.

---

## 5. CALM — Conditional Adversarial Latent Models for Directable Virtual Characters

**Tessler et al., NVIDIA — SIGGRAPH 2023**

**Core idea:** Learn a semantically meaningful 64-dim latent space of motion using conditional adversarial training. Given a target motion style or direction, the model selects a latent code z that the low-level policy executes. A high-level controller conditioned on task variables composes skills without task-specific retraining.

**What makes it special:** The latent space has semantic structure — similar motions cluster together, enabling smooth interpolation between styles. Zero-shot composition is possible: specify a direction and style never seen together during training, and the model generates a plausible result.

**Architecture:**
- Encoder E: maps motion M → latent z ∈ R^64 (2-second sub-motions)
- Low-level policy: π(a|s,z), conditioned on z — 31-dim PD targets
- High-level policy: GRU (256 units), outputs z given task variables
- Training: IsaacGym, 4096 parallel envs, 5 billion PPO steps total

**Key results:**
- Latent diversity: 19.8 ± 0.4 distinct behaviors
- Controllability: 78% accuracy in generating requested motion style
- Zero-shot generalization: Unseen task/style combinations produce plausible results
- Motion interpolation in latent space: smooth style transitions

**Marathon-envs application:**
CALM sits between ASE (less controllable) and MaskedMimic (more capable). It's most useful for:
- Interactive style selection: User input maps to latent code → character adopts that style
- Text conditioning: Bind a language embedding to z for instruction-following
- Style composition: Mix two latent codes for blended styles

**ONNX export:** Medium feasibility. Encoder MLP exports cleanly; GRU recurrence needs ONNX stateful handling. Main challenge: the FSM state management for high-level policy.

**Code:** [NVlabs/CALM](https://github.com/NVlabs/CALM) — IsaacGym, actively maintained 2024.

---

## 6. AdaptNet — Policy Adaptation for Physics-Based Character Control

**Xu et al., Roblox — SIGGRAPH Asia 2023**

**Core idea:** Adapt a pre-trained locomotion policy to new tasks (morphology changes, style transfer, terrain) in 10–30 minutes by injecting a learnable modifier into the policy's latent space. Only the modifier parameters are trained; the original policy is frozen.

**What makes it special:** The latent injection approach means the pre-trained policy's physical dynamics knowledge is fully preserved. New task adaptation is cheap (minutes vs hours) because the modifier only needs to redirect existing behaviors, not learn dynamics from scratch.

**Architecture:**
- Pre-trained policy: 2-layer MLP (256 units) + GRU, frozen after pre-training
- Latent modifier: `z = E_ξ(s) + I_δ(s, c_t)` — adds task-specific component to latent
- c_t: task-specific control input (style label, morphology params, terrain type)
- Only δ parameters are updated during adaptation

**Training setup:**
- IsaacGym, LAFAN1 dataset (walk + run)
- Pre-training: 26 hours multi-GPU
- Adaptation: 10–30 minutes for simple tasks (style transfer, body scaling), up to 4 hours (terrain)

**Key results:**
- Style transfer (stomp → pace → jaunty walk): Smooth latent space interpolation
- Morphological adaptation: Works with body length/mass changes
- Terrain adaptation: Ice, rough terrain, procedural obstacles
- Latent space visualization shows semantic clustering of motion types

**Marathon-envs application:**
AdaptNet directly solves the "new environment" problem:
1. Pre-train on Walker2d or MarathonMan (already done)
2. For each new style or environment: inject a small modifier and train for 10–30 minutes
3. Eliminates the need to retrain from scratch for style variations (JazzDancing → MMAKick)
4. Style interpolation in latent space enables smooth blending between learned behaviors

**ONNX export:** High feasibility. MLP + GRU architecture, modifier is simple additive operation.

**Code:** [Roblox/AdaptNet](https://github.com/Roblox/AdaptNet) — open source, pre-trained models included.

---

## 7. PARC — Physics-based Augmentation with Reinforcement Learning for Character Controllers

**Xu et al. — SIGGRAPH 2025**

**Core idea:** Iteratively expand a MoCap dataset with physics-corrected synthetic motions. A diffusion model generates candidate terrain-adaptive motions; a PPO physics tracker executes them; motions the tracker can execute successfully are added back to the dataset. 4 iterations progressively unlock terrain behaviors not present in original data.

**What makes it special:** The physics tracker acts as a filter — only physically plausible motions survive. This feedback loop improves both the generator (better terrain motions) and the tracker (better physics execution) simultaneously. New skills emerge that were impossible from the original 14-minute dataset.

**Architecture:**
- Motion Generator G: Diffusion transformer conditioned on terrain heightmap (35×35), direction, contact labels
- Motion Tracker: 2048-unit FC policy, PPO-trained on expanding dataset
- Iteration loop: G generates → tracker evaluates → successful motions added to dataset → repeat

**Training setup:**
- IsaacGym, A6000 GPU
- Initial dataset: 14 minutes of parkour skills (vault, climb, jump, run)
- 4 iterations, ~1 month total training time

**Key results:**
- After 4 iterations: 68% success rate on 100 novel terrains
- Novel behaviors generated: Gap jumping, ledge climbing, compound parkour
- Dataset expansion: 14 min → 50+ min (spatial variations)
- Waypoint distance: 1.908 → 0.596m (final waypoint error reduction)

**Marathon-envs application:**
PARC addresses the fundamental limitation of MoCap-dependent training:
- The terrain variants (TerrainHopper, TerrainWalker, etc.) could use PARC to generate novel terrain-adaptive behaviors automatically
- Only 14 minutes of initial reference motion needed; the system generates the rest
- The iterative augmentation pattern is applicable to any marathon-envs environment

**ONNX export:** Low feasibility for the diffusion generator (requires iterative denoising). The tracker policy (FC MLP) exports trivially. Generator could be distilled into a flow model for faster inference.

**Code:** [msooloo/PARC](https://github.com/msooloo/PARC) — GitHub available.

---

## Quick Reference

| Paper | Year | Key Advance | Best Use in Marathon-Envs | ONNX | Priority |
| --- | --- | --- | --- | --- | --- |
| PFNN | 2017 | Phase-indexed weights, terrain-aware | Kinematic trajectory generator for DReCon | ✓ | Tier 2 |
| Learned Motion Matching | 2020 | Neural motion matching, 50× compression | Stepper as kinematic layer | ✓ | Tier 2 |
| DeepPhase | 2022 | Learned periodic phase manifold | Automatic phase extraction from MoCap | Partial | Tier 2 |
| CAMDM | 2024 | Real-time diffusion, 60+ FPS, Unity ONNX | Kinematic layer (best entry point) | ✓ | Tier 2 |
| CALM | 2023 | Conditional latent space, zero-shot | Style selection via latent conditioning | Partial | Tier 3 |
| AdaptNet | 2023 | Latent injection, 10-30min adaptation | Fast per-environment style transfer | ✓ | Tier 2 |
| PARC | 2025 | Iterative terrain augmentation | Auto-generate terrain behaviors | Partial | Tier 3 |
