# Reinforcement Learning Concepts

## PPO vs SAC

| | PPO | SAC |
| --- | --- | --- |
| Type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Replay buffer | No | Yes — reuses old experiences |
| Exploration | Entropy bonus | Automatic entropy tuning |
| Stability | Very stable | Stable but more hyperparams |
| Best for | Many parallel envs, game/sim | Single env, slow simulation |

**Observed results on Marathon Envs Walker2d (this machine):**

| Algorithm | Setup | Steps | Wall time | Best reward |
| --- | --- | --- | --- | --- |
| PPO | 200 envs, normalize: true | 5M | 64 min | **828** |
| SB3 SAC | 4 envs | 1M | 3h | 655 |
| ML-Agents SAC | 200 envs | 1M | 3.3h | 646 |

**PPO is the recommended algorithm for this project.** SAC's published benchmark scores (2000–5000 on Walker2d) come from MuJoCo Walker2d-v2 — a different physics engine with different reward shaping. They do not transfer to Marathon Envs.

### Why Unity ML-Agents chose PPO as default

The ML-Agents team explicitly states PPO was chosen for being *"more general purpose and stable than many other RL algorithms."* Three reasons explain why PPO outperforms SAC in game/simulation environments:

**1. On-policy stability with parallel agents**
PPO uses freshly collected data each update, so it always trains on the current policy's behavior. SAC's replay buffer accumulates transitions from past policies — when 200 parallel agents are all improving simultaneously, the buffer fills with stale, increasingly unrepresentative data. This is why `buffer_size=50000` with 200 agents is almost useless: the buffer fills in ~250 steps and the oldest experience is immediately evicted.

**2. Non-stationarity in multi-agent and self-play scenarios**
Unity explicitly recommends PPO over SAC for self-play: *"the non-stationary dynamics caused by a changing opponent can cause significant issues in the experience replay mechanism used by SAC."* In any setting where the environment's dynamics change because other agents' policies change, SAC's off-policy assumptions break down.

**3. SAC's sample efficiency advantage requires specific conditions**
SAC's edge only materializes in slow, heavy simulations (>0.1s per step) with a single environment — exactly the opposite of the parallel, fast-physics setup that marathon-envs uses. With PhysX running at 20× time scale across 200 agents, PPO generates data faster than SAC can meaningfully reuse it.

**Bottom line:** PPO is not the default because it's the best algorithm in theory — it's the best choice for the specific combination of fast physics simulation + many parallel environments that defines the Marathon Envs training setup. This repo supports both — PPO via `mlagents-learn`, SAC via `stable_baselines_sac_train.py`.

---

## Emergent Gaits

The alternating leg gait in Walker2d arises **naturally** from the reward signal (forward velocity + upright + energy efficiency). Nobody programs the gait pattern. Physics makes alternating legs the optimal solution — one leg must support while the other swings, doing both simultaneously wastes energy and loses balance.

In 3D (MarathonMan) without motion capture guidance, the agent often discovers stable but non-human gaits. This is why style transfer exists.

---

## Energy Efficiency Reward

From `BodyManager002.cs`:

```csharp
jointEffort = Mathf.Pow(Mathf.Abs(muscle.TargetNormalizedRotationX), 2);
```

- Joint actions are normalized to `[-1, 1]`
- Effort = squared magnitude of each joint action (squaring penalizes large actions disproportionately)
- All joints averaged → single effort value `[0, 1]`
- Reward = `1 - effort` → maximum reward when joints are still

Squaring means moving a joint to 50% costs 4× more than 25% — strongly discourages flailing, naturally produces smooth efficient gaits.

---

## DeepMimic / Style Transfer

Instead of rewarding forward velocity, the reward compares the agent's pose to a **motion capture clip** at every timestep using:

```
reward = weight × exp(-distance_from_reference_pose)
```

`exp(-x)` gives 1.0 when perfectly matching, drops toward 0 as distance grows.

**Reward components in this repo** (`StyleTransfer002Agent.cs`):

| Component | Weight | What it measures |
| --- | --- | --- |
| Joint rotations | 35% | Exact pose match |
| End effector position | 15% | Where hands/feet are |
| Angular momentum | 15% | Rotational energy of body |
| Center of mass velocity | 10% | Movement speed/direction |
| Joint angular velocity | 10% | How fast joints rotate |
| Center of mass position | 5% | Where body is in space |

The **phase variable** tells the agent where it is in the animation cycle so it knows which pose to target now.

Early termination: if total reward drops below 0.5 (50% match), episode resets — forces agent to stay close to mocap reference.

---

## Hierarchical RL

Policies can be stacked at arbitrary depth:

```
Level 3: Meta-controller  →  coordinates multiple agents toward shared goal
Level 2: Per-agent controller  →  selects behaviours for each agent  
Level 1: Sub-policies  →  running, jumping, walking, etc.
```

This repo implements two levels:
- **Sub-policies**: MarathonManWalking, MarathonManRunning, MarathonManJazzDancing, etc.
- **Controller**: `ControllerMarathonMan-v0` — a trained PPO agent that selects which sub-policy to activate

The controller observes higher-level state (goal direction, speed) and outputs which sub-policy to run. No hard-coded transition rules — the switching behavior is learned.

**Limitation**: adding a new sub-policy changes the controller's action space, requiring the controller to be retrained from scratch. The sub-policies themselves don't need retraining.

**Performance penalty of switching**: negligible — just two forward passes through small networks (microseconds). The real challenge is motion transition smoothness, not compute.

---

## Catastrophic Forgetting

Training a policy on a new behaviour destroys the old one — the network overwrites weights that encoded the first skill.

**Mitigations:**
- **Freeze + fine-tune**: freeze most layers, train only last 1-2 layers for new behaviour
- **Elastic Weight Consolidation (EWC)**: penalizes changing weights important for first task
- **Hierarchical approach**: keep policies separate, train a controller on top (cleanest solution, used in this repo)
- **`--initialize-from`**: use a previous run's weights as starting point for related tasks

---

## Multi-Policy Control (e.g. Walking + Throwing)

Multiple policies controlling one agent simultaneously:

1. **Additive blending**: `action = α × policy_A + β × policy_B` — simple but policies fight each other
2. **Separate action spaces**: walking controls legs, throwing controls arms — works well when body parts are independent
3. **Train jointly from scratch**: single policy with combined reward — most principled, can't reuse existing policies
4. **Residual policies**: `action = base_policy + small_residual_policy` — residual trained to add new behaviour while minimally disturbing the base

Option 2 is most practical for this repo — MarathonMan has enough joints to designate arm joints to one policy and leg joints to another.

---

## Observation Design

**Explicit velocity/acceleration** is more efficient than passing position history:
- Velocity/acceleration computed for free by physics engine
- Network uses capacity for decision-making, not finite differences
- Training is faster and more stable
- Matches biological sensory systems (muscle spindles sense velocity directly)

**When position history helps**: noisy or partially observable environments where explicit state is unavailable (real robot sensors, optical cameras).

### Fixed Observation Size Requirement

Standard MLP policies (used by PPO and SAC, including all marathon-envs environments) require a **fixed-size observation vector declared before training starts**. The input layer has a fixed number of neurons baked into the network weights — you cannot change it between episodes or mid-training without breaking the model.

In ML-Agents, the observation count is enforced at environment startup and again when loading a `.onnx` model for inference. If `CollectObservations()` produces a different count than what the model was trained on, Unity throws a mismatch error.

**Techniques for variable-situation scenarios:**

| Technique | How | When to use |
| --- | --- | --- |
| Padding + mask | Always send max-size vector; zero unused slots; include a binary mask | Variable number of nearby entities (obstacles, allies) |
| Top-K aggregation | Sort by distance, keep closest K, discard the rest | Unlimited world entities, only nearest K matter |
| Max/mean pooling | Aggregate all entities to a fixed summary vector | Order-invariant sets |
| LSTM/GRU | Per-step input is still fixed; hidden state carries memory | Sequential or partially observable scenarios |
| Attention / Transformer | Variable-length tokens, fixed embedding size | Research; not native in ML-Agents |

**In this project:** every agent class has a fixed joint count, sensor count, and contact flag count. These are defined structurally by the ragdoll prefab. If you add a body part, you must retrain — the existing `.onnx` model is incompatible because its input dimension changed.

---

## Partial Observability (Optical Sensors)

When an agent can only see what's in its field of view and must infer opponent velocity from visual input:

| Approach | How | Best for |
| --- | --- | --- |
| Frame stacking | Pass last N frames as input | Fast movements, simple scenes |
| LSTM/GRU | Hidden state summarizes all past obs | Tracking occluded objects |
| Learned optical flow | Separate net estimates velocity from frames | Clean separation of concerns |
| World models (DreamerV3) | Agent predicts future states internally | State of the art, complex |

---

## Privileged Information / Asymmetric Actor-Critic

Train with god-mode information, deploy with limited sensors:

```
Training:
  Actor  (runs at inference) → sees noisy/limited observations
  Critic (training only)     → sees full privileged state

Inference:
  Only Actor runs → no privileged information needed
```

The critic gives better gradient signal using perfect information; the actor learns to perform with only what a real agent would see. Throw the critic away at deployment.

**Noise techniques to simulate real-world limitations:**

| Technique | Simulates |
| --- | --- |
| Gaussian noise on positions | Imperfect sensors |
| Random observation dropout | Occlusion |
| Delayed observations | Reaction time / latency |
| Quantization | Low resolution sensors |
| Field of view cone masking | Limited vision range |
| Physics parameter randomization | Sim-to-real gap |

---

## Motion Matching for Behaviour Transitions

Instead of hand-authored transitions, maintain a large database of mocap clips and search for the best matching pose every frame based on current body state and desired future trajectory. Transitions emerge naturally from the database.

Applied to hierarchical RL:

```
RL Controller → desired velocity/direction
      ↓
Motion Matching → finds nearest mocap pose
      ↓
Physics agent → tracks that pose
```

This is essentially what `ControllerMarathonMan-v0` (DReCon) does. The remaining hard problem is making a ragdoll physically track kinematically valid transitions without falling.

---

## LoRA in RL Context

LoRA (Low-Rank Adaptation) works by training small adapter matrices on frozen LLM weights, exploiting the massive overparameterization of billion-parameter models.

RL policy networks (64–256 hidden units, thousands of parameters) have no redundancy to exploit — a LoRA adapter would essentially just be retraining the whole network. No practical benefit over standard fine-tuning.

Use `--initialize-from` in mlagents instead:

```bash
mlagents-learn config/... --run-id=new_run --initialize-from=old_run
```

---

## Non-Walking Locomotion (Swimming, Flying)

The core RL pipeline (PPO/SAC + active ragdoll + reward signal) is medium-agnostic. What changes per medium is the physics forces and reward design.

### Swimming

**Approach:** add buoyancy and drag forces per body part in `FixedUpdate`. The RL training loop is unchanged.

```csharp
// Per body part, in FixedUpdate:
float submergedFraction = Mathf.Clamp01((waterLevel - part.bounds.min.y) / part.bounds.size.y);
rb.AddForce(Vector3.up * buoyancy * submergedFraction, ForceMode.Force);
rb.AddForce(-rb.velocity * drag * submergedFraction, ForceMode.Force);
```

- Buoyancy counteracts gravity proportional to submersion depth
- Drag replaces air resistance with water resistance (10–100× higher coefficient)
- No new ML-Agents code needed — just additional physics forces in `BodyManager002`

**Reward redesign:** replace forward velocity reward with displacement toward a goal position in 3D. Add "not drowning" component (keep head above water surface for amphibious agents).

**Feasibility:** straightforward. The ragdoll physics handle underwater poses naturally once forces are applied.

### Flying

Feasibility depends heavily on the flight model:

| Type | Approach | Difficulty |
| --- | --- | --- |
| Jetpack / thrust | Add upward force based on action | Easy — same as swimming |
| Bird wings | Lift = f(wing angle × velocity²) | Medium — custom aerodynamic force |
| Hang glider | Drag-based glide ratio | Medium |
| True aerodynamics | CFD-accurate lift/drag surfaces | Very hard — not practical for RL |

**Practical path for game characters:** jetpack model (thrust force on torso, action maps to thrust magnitude) is trivially added to `BodyManager002`. Wing-based flight is harder because the reward landscape is sparse — the agent must discover that flapping generates lift.

**Curriculum trick for winged flight:** start with strong upward assist that decays over training, forcing the agent to learn flapping before the assist disappears.

---

## Object Interaction (Push, Carry, Pull)

### Pushing (works now)

No changes needed. A physics agent that walks into a Rigidbody will push it via PhysX contact forces. Add a "push object toward goal" reward component and the agent discovers pushing.

### Carrying and Pulling (magic grab pattern)

Finger articulation is prohibitively expensive to train (dozens of joints, contact-rich, sparse reward). The practical game-dev solution is **magic grab**:

1. On grab trigger: instantiate a `ConfigurableJoint` connecting the held object's Rigidbody to the agent's hand Rigidbody
2. Set `ConfigurableJointMotion.Locked` for all axes (or spring-damper for soft grasp)
3. On release: destroy the joint

The agent's hand is driven by its existing policy. The object follows as a constraint. No finger simulation needed. The agent learns to walk/balance with the additional load via normal RL reward.

**Reward additions for carry tasks:**
- Object stays above floor (not dropped)
- Object reaches goal position
- Agent remains upright (load changes balance)

### Hard version: Dexterous Manipulation

For actual grasping with fingers: InterMimic (CVPR 2025) trains on 46k human-object interaction clips. Requires a large MoCap + object interaction dataset. Not practical for the current setup — magic grab covers 95% of game use cases.

**Relevant papers:** InterMimic ([Sirui-Xu/InterMimic](https://github.com/Sirui-Xu/InterMimic)), Omnigrasp ([ZhengyiLuo/Omnigrasp](https://github.com/ZhengyiLuo/Omnigrasp)).

---

## Injury and Dismemberment (Adaptive Locomotion)

The agent must continue attempting locomotion with reduced or removed limbs — learning compensatory strategies.

### Joint Weakening (Injury Simulation)

`DomainRandomization.cs` already exists in `Scripts/ActiveRagdoll002/` and is the right place for this. Extend it to:

1. At episode start, randomly select 0–N joints to weaken (zero or reduce max joint torque)
2. Expose which joints are weakened as binary flags in the observation vector
3. Agent learns to compensate — if right leg is weakened, shift weight left

```csharp
// Example: halve the max force on a random joint
muscle.MaxForce *= 0.1f; // 10% of normal strength
```

The agent receives the weakness flags in its observation and learns that `flag[i] = 1` means joint `i` is unreliable. Over training it discovers compensatory gaits.

### Dismemberment

For removing limbs entirely:

1. Disable the Rigidbody and Collider for the dismembered body part
2. Set the corresponding observation values to zero
3. Add a binary `dismembered[i]` flag to the observation
4. Zero out the actions targeting the missing joint

This is exactly the same principle as **MaskedMimic's** binary mask — zero out the joints that don't exist/matter. The observation mask tells the agent which parts of its body are functional.

**Progressive curriculum:** train healthy agent first → freeze weights → introduce random weakening/dismemberment and fine-tune. The pre-trained gait knowledge transfers; the agent only needs to learn compensation.

**Relevant papers:** PHC (Perpetual Humanoid Control) trains a recovery controller that handles arbitrary body states. MaskedMimic's masking framework generalizes across joint subsets. Both use the binary mask observation pattern described above.

---

## External Perturbations (Robustness Training)

Teaching the agent to maintain locomotion under external disturbances. The core RL pipeline is unchanged — what varies is what forces are applied, what the agent observes, and the training curriculum.

### Observable vs Unobservable Perturbations

This is the central design decision for any perturbation type:

| Mode | Agent observes | Training difficulty | Generalization |
| --- | --- | --- | --- |
| Observable | Perturbation parameters directly (wind vector, gravity factor) | Easy — explicit adaptation | Limited to observed range |
| Unobservable | Only proprioception (velocities, accelerations) | Hard — must develop implicit robustness | Better OOD generalization |
| Privileged critic | Critic sees params; actor sees only proprioception | Medium | Best of both worlds |

**Privileged critic is the recommended approach for games:** the actor (deployed at runtime) never needs perturbation parameters. The critic uses them during training to give better gradient signal. Throw the critic away at deployment. This is the same pattern as Asymmetric Actor-Critic (see above).

---

### Wind and Applied Forces

**Implementation:** add a force vector to each body part's Rigidbody in `FixedUpdate`. Wind can be constant, sinusoidal (gusts), or Perlin-noise (turbulence):

```csharp
// In BodyManager002 FixedUpdate or a new WindManager component:
Vector3 wind = windDirection * windStrength;
foreach (var part in BodyParts) {
    part.rb.AddForce(wind * part.dragCoefficient, ForceMode.Force);
}
```

**DomainRandomization.cs** is the right place to randomize `windStrength` and `windDirection` per episode.

**Observable version:** add `windStrength` (scalar) or `windDirection` (normalized Vector3) to the observation. Agent learns directional leaning compensation explicitly.

**Unobservable version:** omit wind from observations. Agent develops wind-resistant posture through repeated exposure to random wind fields during training.

---

### Object Collisions and Projectiles

PhysX handles collision response automatically. A thrown Rigidbody hitting the ragdoll applies an impulse force — the agent experiences it as an unexpected velocity spike in its proprioception. **No special code is needed to receive the hit.**

What needs implementing:

- A projectile spawner that fires objects toward the agent at random intervals during training
- No observation change required (agent sees the effect, not the incoming object)
- If you want the agent to dodge: add a raycast sensor pointing outward and include distance-to-nearest-object in observations

**Training curriculum:**

1. Train base locomotion (no projectiles)
2. Introduce light, slow projectiles with long intervals
3. Progressively increase mass, speed, and frequency
4. Agent learns to absorb and recover rather than collapse

**PHC's fall recovery** (see Tier 2 roadmap, T2-C) is directly useful here — the recovery controller handles the post-impact fall state. Combine perturbation robustness training with a PHC-style recovery controller for an agent that never resets.

---

### Terrain Perturbations

**What already exists:** `TerrainGenerator.cs` and the four terrain environments (TerrainHopper, TerrainWalker, TerrainAnt, TerrainMarathonMan) already implement procedural terrain. Height samples under/around the feet are included in terrain agent observations.

#### Rugged / Uneven Ground

Extend `TerrainGenerator.cs` with a roughness parameter:

- Low roughness: gentle undulations, agent uses existing locomotion
- High roughness: large amplitude bumps, agent must actively balance mid-stride
- Sample 9–16 terrain heights in a grid around the agent's feet and include them in the observation

#### Ramps

Easiest terrain extension. Key observation addition: body inclination (already captured via `transform.up` relative to world `Vector3.up`). Agent needs to shift center of mass forward on uphill, backward on downhill. Curriculum: 0° → 5° → 15° → 30° max slope.

#### Stairs

Harder than ramps due to discrete step geometry — PhysX can catch foot colliders on step edges. Mitigations:

- Chamfer stair edge colliders (bevel the corner geometry)
- Use kinematic capsule approximations for the agent's feet colliders during training
- Add foot contact sensor to observations (which foot is grounded, contact normal)

Curriculum: 2cm steps → 10cm → 20cm. Agent at 20cm step height requires a distinct high-knee gait.

#### Ground Shaking (Earthquakes)

Apply periodic displacement to the terrain transform or impulse forces to the agent's root Rigidbody:

```csharp
// Shake the ground:
float shakeX = Mathf.Sin(Time.time * frequency) * amplitude;
groundTransform.position = new Vector3(shakeX, 0, 0);

// Or apply directly to agent root:
rootRb.AddForce(new Vector3(shakeX, 0, 0), ForceMode.VelocityChange);
```

**Observable version:** expose `shakeAmplitude` and `shakeFrequency` to the agent. Agent learns to anticipate and pre-lean.

**Unobservable version:** agent only feels the acceleration — develops a wider, lower stance as an implicit robustness strategy.

---

### Gravity Changes

`Physics.gravity` is a global per-scene Vector3. For per-agent gravity override (multiple parallel environments with different gravity):

```csharp
// Cancel global gravity, apply custom gravity per body part:
Vector3 customGravity = new Vector3(0, -9.81f * gravityMultiplier, 0);
foreach (var part in BodyParts) {
    part.rb.AddForce(customGravity - Physics.gravity, ForceMode.Acceleration);
}
```

| Gravity mode | Effect | Training challenge |
| --- | --- | --- |
| Low (0.1–0.5×) | Agent floats, overshoots joints | Damping tuning; policy tends toward slow cautious motion |
| High (1.5–3×) | Heavier; joints need more torque | MaxForce limits; agent may collapse if torques insufficient |
| Sideways | Wall-running scenarios | Reward orientation must change (upright relative to custom gravity direction) |
| Variable per episode | General gravity robustness | Agent learns to probe gravity quickly and adapt |

**Observable:** include `gravityMultiplier` (scalar) in observation. Agent learns joint scaling explicitly.

**Unobservable:** agent probes gravity by taking a small hop and observing how quickly it returns to ground. Requires implicit inference over recent proprioception — benefits from LSTM/GRU in the policy.

---

### Perturbation Observation Cheat Sheet

| Perturbation | Observable input | Effect seen in proprioception |
| --- | --- | --- |
| Wind | Wind vector (3 floats) | Linear velocity deviation |
| Projectile impact | — (no lookahead practical) | Sudden linear/angular velocity spike |
| Terrain roughness | Height map samples (9–16 floats) | Foot contact normal variation |
| Stairs | Foot contact normals + height | Sudden foot elevation changes |
| Ground shaking | Shake amplitude + frequency | Root acceleration noise |
| Gravity change | Gravity multiplier (1 float) | All accelerations scaled uniformly |

---

### DomainRandomization.cs as the Centralized Perturbation Hub

`DomainRandomization.cs` already exists and runs per-episode. Extend it with all perturbation parameters:

```csharp
windStrength      = Random.Range(0f, maxWind);
windDirection     = Random.onUnitSphere.normalized;
gravityMultiplier = Random.Range(minGravity, maxGravity);
terrainRoughness  = Random.Range(0f, maxRoughness);
// Optionally schedule projectile spawns for this episode
```

Start with all ranges near their baseline values (wind=0, gravity=1×) and expand them progressively as training stabilizes. This is identical to the domain randomization strategy used in sim-to-real robotics (OpenAI Rubik's Cube, 2019) — the wider the randomization range during training, the more robust the deployed policy.

---

## Equipment Load (Center of Mass Shifts)

Adding weapons, armor, and backpacks is a distinct perturbation type from the others: it changes the **action-to-effect mapping** rather than the environment. The same leg extension command produces different balance results with heavy leg armor than without — the agent must rescale its motor commands, not just compensate for an external force.

### Physics of Equipment

Added equipment changes three things simultaneously:

- **Center of mass** — mass at an offset from the body center shifts the composite CoM in that direction
- **Moment of inertia** — mass at the extremities (hands, feet) increases rotational inertia, making that limb sluggish to accelerate
- **Required joint torque** — heavier limbs need more torque to produce the same angular acceleration

| Equipment | Attachment | CoM shift | Primary locomotion effect |
| --- | --- | --- | --- |
| Backpack | Torso (rear offset) | Backward + upward | Must forward-lean; increased upright torque |
| Chest plate | Torso (front) | Forward | Slight backward lean; heavier breathing effort |
| Full armor suit | Distributed | Mostly uniform | All joints need more torque; slower natural frequency |
| Gun in one hand | One hand | Lateral + arm mass | Asymmetric arm swing; contralateral lean |
| Shield on one arm | One forearm | Lateral, large | Pronounced lateral tilt; asymmetric gait |
| Heavy leg greaves | Both lower legs | Distal leg mass | More knee torque; shorter natural stride; slower cadence |

### Implementation

Modify `rb.mass` and `rb.centerOfMass` directly on the relevant body part Rigidbodies. PhysX recalculates the composite CoM and inertia tensor automatically — no extra Rigidbodies or joints needed:

```csharp
// In DomainRandomization.cs or an EquipmentManager, per episode:
void ApplyEquipmentLoad(BodyPart002 part, float equipmentMass, Vector3 equipmentLocalOffset) {
    Rigidbody rb = part.rb;
    Vector3 newCoM = (rb.centerOfMass * rb.mass + equipmentLocalOffset * equipmentMass)
                     / (rb.mass + equipmentMass);
    rb.mass += equipmentMass;
    rb.centerOfMass = newCoM;
}
```

For accurate rotational dynamics with distal mass (e.g., heavy weapon at arm's end), also update `rb.inertiaTensor`. The CoM + mass change alone gets ~90% of the physical effect.

**Reset on episode end:** store original `mass` and `centerOfMass` values and restore them in `OnEpisodeBegin()`.

### Equipment Observation Encoding

Include equipment state in the observation vector so the agent can adapt explicitly:

```csharp
// Per body part: normalized added mass and CoM offset direction
sensor.AddObservation(equipmentMass / maxExpectedMass);      // [0,1]
sensor.AddObservation(equipmentLocalOffset.normalized);       // 3 floats
```

For a full equipment system with many slot types, a compact encoding is sufficient: `[totalExtraMass, torsoOffset_x, torsoOffset_z, leftArmMass, rightArmMass, leftLegMass, rightLegMass]` — 7 floats covers most practical combinations.

**Unobservable version:** omit equipment state from observations. The agent perceives only the effect (worse balance, different inertia) and must develop implicit robustness through DR exposure. Works but the agent develops a conservative average policy that handles neither extreme well.

### Training Strategies

**Domain Randomization (recommended baseline):** randomize equipment mass and attachment point each episode. Agent learns to handle the full distribution.

```csharp
float extraTorsoMass    = Random.Range(0f, maxArmorMass);
float extraRightArmMass = Random.Range(0f, maxWeaponMass);
```

**Curriculum:** train unencumbered → add light loads → add asymmetric loads (one side only) → add full heavy configuration. Asymmetric loading is harder than symmetric — it requires lateral lean correction that symmetric training never teaches.

**AdaptNet (T1-C in future-techniques.md):** the most principled approach. Their paper explicitly tested morphological adaptation including body mass changes. Pre-train the base locomotion policy unencumbered, then inject a small latent modifier conditioned on equipment state. Adapts in 10–30 minutes without retraining from scratch. Each new equipment configuration is a new "style" that AdaptNet can accommodate cheaply.

**PHC multiplicative primitives:** PHC's architecture adapts torque scaling implicitly as body configuration changes, making it naturally robust to mass changes without explicit equipment observations.

### Asymmetric Loading Is the Hard Case

A gun in the right hand only shifts CoM right, requiring leftward lean. The agent must learn:

1. Which side is heavier (from observation or proprioception)
2. How much to lean (proportional to mass × offset distance)
3. How to modulate arm swing (the loaded arm swings less freely)

Training exclusively on symmetric loads (both hands, uniform armor) produces an agent that fails badly on asymmetric loads. Always include asymmetric configurations in the DR distribution.

---

## Large Agent Crowds (Zombie Mobs, Enemy Swarms)

Fully physics-simulated crowds of RL-controlled agents are technically feasible — but require the right architecture to hit real-time frame rates. No shipped game has done this yet; it's genuine unexplored territory.

### Core Principle: One Policy, N Instances

All agents in the crowd share a single trained policy — one set of weights in memory. Each agent instance maintains its own observation and action buffers (~a few KB each). The per-agent memory cost is trivial; the constraint is physics simulation.

Train one zombie agent. Deploy the policy 100 times.

### Physics Is the Bottleneck

A full MarathonMan ragdoll is ~17 Rigidbodies + 14 ConfigurableJoints. PhysX cost scales roughly linearly with active Rigidbody count:

| Ragdoll complexity | Bodies per agent | 30 agents total | Feasibility at 60 fps |
| --- | --- | --- | --- |
| Full MarathonMan | 17 | 510 Rigidbodies | Marginal (i7-class CPU) |
| Simplified humanoid | 7 | 210 Rigidbodies | Comfortable |
| Capsule + limbs | 5 | 150 Rigidbodies | Very comfortable |

**Design implication:** build a purpose-made zombie ragdoll at 5–7 bodies. Torso, head, two arms, two legs is sufficient for convincing physical behavior. You don't need individual fingers, collarbone, or sternum.

### Physics Level of Detail (LoD)

The standard game-industry solution for large crowds: run different simulation fidelity based on distance from the camera/player.

```text
Distance from player    Simulation mode
─────────────────────────────────────────
< 10m  (5–10 agents)   Full ragdoll + RL policy at full frequency
10–30m (10–20 agents)  Simplified ragdoll + RL policy every 3 frames
30–60m (20–50 agents)  Kinematic animation driven by policy output (no physics)
> 60m  (rest of crowd) Billboard / impostor, no simulation
```

Agents transition between levels as they move relative to the player. The player can't perceive the difference at medium-to-far range.

### Batched Inference

Rather than 50 separate Sentis forward passes per frame, collect all agent observations into a single batched tensor and run one GPU dispatch. Sentis supports batched inference. Cost comparison:

- **Unbatched:** N kernel launches, N CPU↔GPU transfers → expensive for N > 5
- **Batched:** 1 kernel launch, 1 transfer, GPU parallelizes across agents → scales to 100+ agents with marginal cost increase

Implementation: a central `ZombieSwarmBrain` MonoBehaviour collects observations from all active agents each `FixedUpdate`, runs a single batched `Model.Execute()`, then distributes actions back. Each zombie reads its slice of the output tensor.

### Staggered Decision Updates

Not every agent needs a policy decision every physics step. Stagger updates by agent index:

```csharp
// Agent i makes a decision only on frames where (frame % decisionInterval == i % decisionInterval)
bool makeDecision = (Time.frameCount % decisionInterval) == (agentIndex % decisionInterval);
```

At 60 fps with `decisionInterval = 3`: each agent decides at 20 Hz, which is still faster than human perception for locomotion. This reduces inference load by 3× for free.

### Navigation Layer (Hierarchical Architecture)

The RL policy handles local physics (balance, recovery, collision response). A separate navigation layer handles global pathfinding toward the player. This is the same hierarchical pattern described in the Hierarchical RL section above.

**Practical setup:**

1. Unity NavMesh computes a path to the player for each zombie (cheap, runs on CPU, standard Unity feature)
2. NavMesh agent gives a `desiredDirection` vector (2 floats: x, z)
3. The RL policy receives `desiredDirection` as part of its observation alongside standard proprioception
4. The policy learns to locomote physically toward the goal direction, recovering from collisions and terrain

The RL policy never needs to know global map layout. It only needs to know "go this way and stay upright."

### Training for Crowd Behavior

Train a single-agent zombie in isolation first:

- Task: move toward a target position, recover from perturbations (falls, pushes)
- Observation: proprioception + goal direction + distance to goal
- Reward: goal proximity + upright + alive time

For crowd-specific behavior (agent-to-agent collision response, separation), one of two approaches:

**Option A — Emergence via physics:** just spawn multiple agents. PhysX ragdoll-to-ragdoll contact forces handle separation naturally. No explicit crowd coordination needed. Agents push each other out of the way as a side effect of normal locomotion.

**Option B — Multi-agent observation:** add nearby agent positions/velocities to each agent's observation (nearest 3 agents, relative position and velocity). Agent learns to route around them. Requires multi-agent training setup.

Option A is the practical starting point. Emergence via physics contact is computationally cheap, requires zero additional training, and looks physically plausible.

### Feasibility Estimate (RTX 2070 Super, i7-class CPU)

| Mode | Active RL agents | Physics agents | Approx. cost |
| --- | --- | --- | --- |
| Conservative | 10 full ragdoll | 10 | Comfortable 60 fps |
| Practical crowd | 20 full ragdoll | 40 simplified | Achievable with LoD |
| Ambitious | 30 simplified + LoD | 100 total | Possible with batching + staggering |

The CPU is the binding constraint (PhysX single-threaded). GPU inference cost is negligible with batching. On higher-end CPU hardware (12+ core), the practical crowd tier becomes very comfortable.

### What Makes This Novel

No shipped game has a crowd of RL-policy-driven fully physics-simulated humanoids. Existing physics crowd games (Gang Beasts, TABS) use hand-crafted active ragdolls, not trained RL policies. The closest research is multi-agent locomotion (e.g., DeepMimic multi-character) but these run offline, not at game frame rates with a live player.

The combination of: shared policy + batched Sentis inference + physics LoD + NavMesh goal direction makes this achievable today with marathon-envs as the training backbone.

---

## Mob Orchestration and Pack Hunting

Coordinated group behavior — encirclement, flanking, wave attacks — layered on top of individual locomotion policies. The architecture separates concerns cleanly: the individual policy handles physics, the conductor handles strategy.

### Architecture: Conductor + Individual Policies

The practical game-dev architecture is two layers operating at different timescales:

```text
[Mob Conductor]  runs every 1–2 seconds
      │  assigns subgoal positions to each zombie
      ▼
[NavMesh Pathfinder]  per zombie, continuous
      │  converts subgoal → next-step direction vector
      ▼
[RL Locomotion Policy]  per zombie, every frame
      │  takes (proprioception + goal direction) → physical actions
      ▼
[PhysX Ragdoll]  physics simulation
```

The RL locomotion policy is already trained (from the crowd section above). The conductor and NavMesh are added on top without retraining the individual policy — the policy already accepts a goal direction as input.

### Conductor: Rule-Based vs RL

**Start rule-based.** A behavior tree or state machine is fast to iterate, predictable, and debuggable. Upgrade to RL only if emergent complexity is required.

**Rule-based conductor logic (sketch):**

```csharp
void UpdateMobStrategy() {
    Vector3 playerPos    = player.transform.position;
    Vector3 playerVel    = player.GetComponent<Rigidbody>().velocity;
    int     activeAgents = zombies.Count(z => z.IsActive);

    if (activeAgents < 3) {
        // Too few to encircle — all rush
        AssignSubgoals_Rush(playerPos);
    } else if (PlayerIsEscaping(playerVel)) {
        // Player moving fast — intercept predicted position
        AssignSubgoals_Intercept(playerPos, playerVel, lookaheadSeconds: 1.5f);
    } else if (AllOnSameSide(playerPos)) {
        // Mob clumped — redistribute around player
        AssignSubgoals_Encircle(playerPos, activeAgents);
    } else {
        // Default: maintain encirclement, tighten slowly
        AssignSubgoals_Tighten(playerPos, closingSpeed: 0.2f);
    }
}
```

**Encirclement geometry** (pure math, no RL):

```csharp
void AssignSubgoals_Encircle(Vector3 center, int count) {
    for (int i = 0; i < count; i++) {
        float angle   = (360f / count) * i * Mathf.Deg2Rad;
        float radius  = encircleRadius;
        Vector3 subgoal = center + new Vector3(Mathf.Cos(angle) * radius, 0, Mathf.Sin(angle) * radius);
        zombies[i].SetSubgoal(subgoal);
    }
}
```

Each zombie receives a subgoal position. NavMesh converts that to a direction. The RL policy walks there physically.

### Pack Hunting Behaviors

#### Encirclement

Each agent occupies a distinct angular arc around the player. Agents that share an arc with an ally move laterally to fill uncovered arcs.

**Emergent version (reward shaping):** give a team reward proportional to angular coverage of the player. Agents self-organize — rushing from the same direction as an ally yields zero marginal reward, so they spread out.

**Angular coverage reward:**

```csharp
// Compute how many distinct 45° sectors around the player are occupied
float CoverageReward(Vector3 playerPos, List<Zombie> zombies) {
    bool[] sectors = new bool[8];
    foreach (var z in zombies) {
        Vector3 dir   = (z.position - playerPos).normalized;
        int sectorIdx = Mathf.FloorToInt((Mathf.Atan2(dir.z, dir.x) + Mathf.PI) / (Mathf.PI / 4));
        sectors[sectorIdx % 8] = true;
    }
    return sectors.Count(s => s) / 8f; // [0,1]
}
```

#### Interception (Cutting Off Escape Routes)

Instead of giving each zombie the player's current position as a subgoal, give a predicted future position:

```csharp
Vector3 intercept = playerPos + playerVel * lookaheadSeconds;
```

Agents that are faster than the player converge ahead of them. Agents too far away fall back to encirclement. The conductor chooses per-agent whether to intercept or encircle based on each zombie's distance and speed.

#### Wave Attacks

Divide zombies into cohorts. Only cohort 0 attacks; cohorts 1 and 2 hold back. When cohort 0 is neutralized (or a timer elapses), cohort 1 rushes. This keeps the player under continuous pressure while preventing all zombies from being dispatched simultaneously.

The conductor rotates cohort assignments each wave — survivors from cohort 0 join cohort 2; cohort 1 becomes cohort 0.

#### Flanking

Designate zombie roles at spawn or episode start: `Rusher` (direct approach), `Flanker_L` (approach from player's left), `Flanker_R` (approach from player's right). Each role gets a different subgoal position:

```csharp
zombies[0].subgoal = playerPos;                              // Rusher: direct
zombies[1].subgoal = playerPos + playerRight * flankOffset;  // Flanker R
zombies[2].subgoal = playerPos - playerRight * flankOffset;  // Flanker L
```

The individual locomotion policy is identical for all roles — only the subgoal differs. No per-role retraining.

### Multi-Agent RL (For Emergent Coordination)

When you want coordination to emerge from training rather than be programmed explicitly:

**MAPPO (Multi-Agent PPO)** is the strongest practical baseline:

- **Training:** centralized critic receives all agents' observations concatenated → better credit assignment ("my reward came from ally's action, not mine")
- **Inference:** each agent's policy uses only local observations → decentralized, O(N) compute at runtime
- **Compatible with PPO:** marathon-envs already uses PPO; MAPPO is a direct extension

**Team reward structure for pack hunting:**

```python
# Individual reward components
r_individual = proximity_to_player * 0.3   # each agent rewarded for being close
r_team       = attack_landed * 0.7         # shared when any agent lands a hit

# Total team reward
r_total = r_individual + r_team
```

High weight on team reward forces agents to learn that coordinated attacks (where distraction by one enables attack by another) yield more reward than isolated rushes.

**ML-Agents limitation:** native ML-Agents has no built-in CTDE (centralized training, decentralized execution) support. Options:

| Path | Effort | What you get |
| --- | --- | --- |
| Shared reward + local obs (emergent) | Low — works in ML-Agents today | Weak coordination, emergent separation |
| PettingZoo + MAPPO + ONNX export | Medium — Python only, then export | Strong coordination, more training complexity |
| Custom centralized critic in ML-Agents | High — requires training code changes | Best result, most work |

**Start with Option 1.** Shared reward + ally positions in observations produces surprisingly good emergent encirclement without any CTDE complexity.

### Observation Design for Coordination

Add to each individual zombie's existing observation:

```csharp
// Nearest N allies: relative position (3 floats) + velocity (3 floats) each
for (int i = 0; i < nearestAllies; i++) {
    sensor.AddObservation(transform.InverseTransformPoint(allies[i].position)); // local space
    sensor.AddObservation(allies[i].velocity);
}

// Optional: conductor subgoal (in local space)
sensor.AddObservation(transform.InverseTransformPoint(assignedSubgoal));

// Optional: role embedding (if using explicit roles)
sensor.AddObservation(roleOneHot); // [1,0,0] = rusher, [0,1,0] = flankerL, etc.
```

Nearest 3 allies adds 18 floats to the observation. This is small relative to the proprioception vector and trains quickly.

### The Left 4 Dead Director as Reference

Left 4 Dead's AI Director is a rule-based orchestrator that controls zombie spawn timing, placement, and type to maintain a target tension level. It doesn't control individual zombie locomotion (kinematic) but the concept directly applies here:

- Director monitors player stress (health, ammo, movement speed)
- When stress drops below threshold: spawn more zombies, trigger specials
- When stress exceeds threshold: give player a break

This macro-level director pairs naturally with the RL locomotion layer — the director manages game feel, the RL agents handle physical execution.

### Self-Play for Adversarial Sharpening

Once the zombie policy and conductor are working, train against an adaptive player agent via self-play:

1. Train zombie policy vs scripted player → basic pursuit
2. Train player policy vs frozen zombie policy → player learns to escape
3. Train zombie policy vs new player policy → zombies adapt
4. Repeat → arms race that sharpens both sides

Self-play is how OpenAI's hide-and-seek emergent coordination (2019) produced tool use and ramp exploitation — the adversarial pressure forced increasingly sophisticated strategies. The same mechanism applied to a zombie mob would produce pack hunting strategies the designer never programmed.

### Summary: Practical Layered Architecture

| Layer | Implementation | Timescale | Trained or programmed |
| --- | --- | --- | --- |
| Individual locomotion | RL policy (marathon-envs) | Every frame | Trained |
| Local pathfinding | Unity NavMesh | Every frame | Programmed |
| Subgoal assignment | Conductor (rule-based BT) | Every 1–2s | Programmed |
| Macro strategy | Director (tension-based) | Every 5–10s | Programmed |
| Adversarial sharpening | Self-play (optional) | Training only | Trained |

Start with all programmed layers above the RL policy. Replace with RL from the bottom up as you want more emergent behavior. The individual locomotion policy is the critical piece — once that works, everything above it is additive.

---

## Competitive Agent Training and Self-Play

Two or more agents competing — race to a goal, physical combat, sumo, keep-away. The training mechanism is **self-play**: agents train against copies or past versions of themselves, creating an arms race that drives both to improve. AlphaGo, OpenAI Five, and AlphaStar all use this mechanism.

### ML-Agents Built-In Self-Play

ML-Agents has native self-play support — no custom training code needed. Add a `self_play` block to the behavior config in `marathon_envs_config.yaml`:

```yaml
behaviors:
  CompetitiveAgent:
    trainer_type: ppo
    hyperparameters:
      # ... standard PPO settings
    self_play:
      save_steps: 20000          # snapshot frequency (steps)
      team_change: 100000        # steps before swapping team sides
      swap_steps: 2000           # how often to swap current opponent snapshot
      play_against_latest_model_ratio: 0.5  # 50% vs latest, 50% vs historical pool
      window: 10                 # how many historical snapshots to keep
```

ML-Agents automatically:

- Saves policy snapshots every `save_steps`
- Maintains an opponent pool of the last `window` snapshots
- Tracks an ELO rating per agent (visible in TensorBoard)
- Manages team assignment so each agent competes fairly

### Self-Play Variants

| Variant | How | Risk | When to use |
| --- | --- | --- | --- |
| Naive (vs current self) | Both agents always use latest policy | Cycling — A beats B, B adapts, A regresses | Never in practice |
| Historical pool | Train vs mix of current + past frozen policies | Slow progress if pool is too large | Default (ML-Agents default) |
| League training | Multiple agent populations with different strategies | Very complex, long training | AlphaStar-scale problems |

The historical pool prevents the **rock-paper-scissors cycle** (A → B → C → A) because past snapshots can't adapt back. ML-Agents' `window` parameter controls pool size — larger window = more diversity, slower convergence.

### Competitive Scenario Types

#### Race to Goal

Two agents start at the same position; first to reach the goal wins.

```text
Reward = -distance_to_goal * 0.01   (dense — keeps learning signal alive)
       + 1.0                         (on reaching goal first)
       - 0.5                         (on reaching goal second)
```

**Physical interference:** since agents are ragdolls, they can body-check each other. The race becomes a strategy game — sprint directly (fast but vulnerable to block) vs position yourself to block the opponent (slower but forces detour). This emergent strategic choice doesn't exist in kinematic racing.

#### Sumo (Push Out of Ring)

Both agents try to push the opponent outside a circular arena. ML-Agents ships a Sumo example environment — the pattern transfers directly to marathon-envs ragdolls.

```text
Reward = +0.01 per step in ring     (survival incentive)
       + 1.0 when opponent falls out
       - 1.0 when self falls out
```

The ragdoll physics make sumo especially compelling: agents discover that low center-of-mass stances are harder to push, leading to emergent crouching behavior. No reward term for crouching — it emerges from the physics.

#### Physical Combat

Agents deal damage on contact, proportional to impact force. PhysX already computes `collision.impulse.magnitude` — use it directly:

```csharp
void OnCollisionEnter(Collision col) {
    if (col.gameObject.CompareTag("Opponent")) {
        float impact = col.impulse.magnitude;
        opponentHealth -= impact * damageScale;
    }
}
```

"Defeat" condition: opponent health reaches zero, or opponent falls and triggers the same early-termination check as the existing marathon-envs locomotion environments (head below threshold).

```text
Reward = damage_dealt_this_step * 0.1    (dense — rewards aggression)
       + 0.005 per step alive            (survival incentive)
       + 1.0 on opponent defeated
       - 1.0 on self defeated
```

#### Keep-Away / Grappling

One agent holds an object (via ConfigurableJoint magic grab). The other tries to separate agent from object (break the joint by pulling with sufficient force, or knock the holder down).

- Holder reward: time holding object + object stays above floor
- Attacker reward: time object is uncontrolled + force applied to holder

#### Tag / Predator-Prey

Asymmetric: one agent (predator) tags the other (prey) by touching. Roles may swap on tag. Classic emergent behavior benchmark — OpenAI's hide-and-seek used this structure and produced tool use.

### Reward Shaping Principles for Competition

**Sparse alone is too hard:** if the only reward is win/lose, early training is pure noise — neither agent has learned enough to produce a meaningful match. Always add dense intermediate rewards.

**Asymmetric reward for asymmetric games:** if roles are different (attacker vs defender), each role should have a different reward function. A single shared reward produces agents that are mediocre at both roles.

**Survival incentive:** a small `+reward_per_step_alive` prevents agents from discovering that immediate self-destruction avoids negative reward. Without it, agents sometimes learn to fall over instantly to "avoid losing."

**Reward clipping:** clip the combined reward to `[-1, 1]` if using mixed dense + sparse signals to prevent the sparse win/lose bonus from drowning out dense shaping rewards.

### Physical Competition Scenarios for Marathon-Envs

These scenarios are uniquely compelling with full ragdoll simulation — they have no kinematic equivalent:

| Scenario | Setup | What emerges |
| --- | --- | --- |
| Sumo ring | Circular platform, push opponent off | Low stance, center-of-mass control |
| Race with blocking | Linear track, first to goal wins | Sprint vs strategic blocking |
| Grappling / takedown | Fall the opponent (head-to-ground) | Leverage, balance disruption |
| King of the hill | Elevated platform, hold position longer | Defensive stance, displacement attacks |
| Keep-away | One holds object, other separates | Tight body shielding vs pulling attacks |

### Teams (2v2, NvN)

ML-Agents self-play supports team-based competition natively. Each agent has a `Team ID`; agents on the same team share reward. Cooperative behavior within teams emerges from the shared team reward; competitive behavior between teams from the opposing objective.

```yaml
self_play:
  # ... same parameters
  # ML-Agents automatically groups agents by Team ID
  # Team 0 trains against Team 1's frozen snapshots
```

2v2 produces more interesting emergent behavior than 1v1 — agents learn to set picks (position self to block opponent of teammate), coordinated attacks, and role specialization (one rushes, one positions defensively) without any explicit programming.

### Training Curriculum for Competition

Competitive training from scratch is hard: if both agents start terrible, no meaningful learning signal exists because matches are random.

**Recommended curriculum:**

1. **Pre-train locomotion** — run standard marathon-envs PPO until the agent can walk/run stably
2. **Introduce scripted opponent** — a heuristic that moves toward the goal or punches directly. Easy to beat, but provides a signal
3. **Switch to self-play** — use the pre-trained policy as the initial snapshot; both agents start at reasonable skill
4. **Expand historical pool** — over time the pool fills with increasingly capable snapshots; the arms race begins

Skipping step 1 is the most common mistake — agents spend millions of steps just learning to stand up before any competitive behavior is possible.

### ELO Tracking

ML-Agents computes ELO in TensorBoard under `self_play/ELO`. Use it to:

- Verify skill is monotonically increasing (ELO rising over time)
- Detect plateaus (ELO flat for long periods → policy stuck in local optimum)
- Compare across training runs (higher ELO at convergence = better policy)

ELO assumes a transitive skill ordering. If you observe **cycling** (ELO oscillates without trend), the historical pool `window` is too small — increase it so the agent trains against more diverse past versions.

### Connection to Existing Marathon-Envs Infrastructure

The individual locomotion policy trained in any marathon-envs environment (Walker2d, MarathonMan) is the direct starting point for competitive scenarios:

- Pre-trained locomotion policy → initialize competitive agent weights via `--initialize-from`
- Same `BodyManager002` / `Muscle002` / `MarathonAgent` stack — add competitive reward on top
- Same ONNX export path → deploy both competing agents in Unity via Sentis at inference time

---

## Neuroevolution: Genetic Algorithms and Neural Networks

Neural network weights are a vector of floats — that vector is a genome. Evolutionary algorithms can optimize it without gradient descent, using population selection and mutation instead. This is **neuroevolution**, and it has produced strong locomotion results on the exact MuJoCo environments that marathon-envs replicates.

### Weights as Genome

A network with N parameters is represented as a vector `w ∈ ℝᴺ`. Evolutionary operations:

- **Mutation:** `w_child = w_parent + σ × N(0, I)` — add Gaussian noise scaled by mutation rate σ
- **Selection:** run each individual in the environment; keep top-K by episode reward
- **Crossover:** for two parents A and B, each weight independently drawn from A or B (uniform crossover) — but this is problematic for neural networks (see below)

**The crossover problem:** weight 42 in network A and weight 42 in network B likely encode unrelated features (different training history → different representations). Averaging or swapping them destroys both. NEAT solves this; OpenAI ES avoids it entirely.

### Key Algorithms

#### OpenAI Evolution Strategies (ES, 2017)

The most directly applicable approach. Not classical GA — no population or crossover. Instead: perturb current weights across many workers, measure fitness change, update in the direction that improved fitness:

```text
For each generation:
  Sample perturbations ε₁...εₙ ~ N(0, I)
  Fitness Fᵢ = evaluate(w + σεᵢ)  [run each worker independently]
  w ← w + α/(nσ) × Σ Fᵢεᵢ        [weighted average of good perturbations]
```

This is equivalent to gradient descent on expected fitness, but computed without backpropagation — pure black-box optimization.

**Results on marathon-envs environments (from the paper):** Walker2d, Hopper, Ant, HalfCheetah — same benchmarks. Comparable final performance to PPO. With 720 CPU workers: 1 hour wall-clock time for Walker2d. PPO on a single machine takes days.

**Key advantage:** embarrassingly parallel. Each worker is independent — no gradient synchronization, no replay buffer coordination. CPU cluster (not GPU) is optimal, since each worker runs a short episode. This directly addresses the PhysX single-threaded bottleneck: more CPU cores = proportionally faster training.

#### NEAT (NeuroEvolution of Augmenting Topologies, 2002)

Evolves both weights and network topology. Structural mutations add neurons and connections:

- **Weight mutation:** perturb existing weights
- **Add connection:** add a new synapse between existing neurons
- **Add node:** split an existing connection into two with a new neuron between

**Crossover fix via innovation numbers:** every gene (weight or structural element) gets a global innovation number when it first appears. During crossover, genes with matching innovation numbers align correctly, even if network structures differ. Genes without a match are inherited from the fitter parent.

**Species protection:** topologically similar networks are grouped into species. Competition happens within species first, preventing a well-performing innovation from being immediately eliminated before it matures.

**Relevance to marathon-envs:** NEAT could evolve the ragdoll control network architecture, discovering how many hidden layers and what connectivity pattern works best for each environment — without manually tuning layer sizes.

#### HyperNEAT

Instead of evolving weights directly, evolves a small generator network (CPPN — Compositional Pattern Producing Network) that *produces* the weights as a function of neuron geometry:

```text
weight(i→j) = CPPN(position_i, position_j)
```

The CPPN captures spatial regularities — e.g., "connections between nearby neurons are stronger," "left-right symmetry in motor cortex." Scales to networks with millions of weights because you're evolving a compact ~50-parameter CPPN, not all weights directly. Naturally discovers symmetric motor patterns for symmetric bodies (humanoid limbs).

#### Population-Based Training (PBT)

The practical hybrid of gradient RL and evolutionary selection. Run N PPO agents in parallel with different hyperparameters:

```text
Every K training steps:
  1. Rank agents by current episode reward
  2. Bottom 20%: copy weights from a top-20% agent (GA selection)
  3. Perturb hyperparameters of copied agent (GA mutation):
     lr *= random.choice([0.8, 1.2])
     batch_size = random.choice([512, 1024, 2048])
  4. Continue training
```

Result: hyperparameters are automatically tuned during training, not before it. Used by DeepMind for AlphaStar. Available in Ray RLlib as `PopBasedTraining` scheduler.

### Hierarchical GA → Neural Network System

The user's exact framing — a two-level hierarchy where the GA drives the outer loop and gradient descent (or neuroevolution) drives the inner loop:

```text
OUTER LOOP — Genetic Algorithm (slow, coarse):
  Population of "meta-genomes" encoding one of:
    • Reward component weights (tune the 7 DeepMimic components)
    • Network architecture (layer sizes, connectivity)
    • Hyperparameters (learning rate, entropy coefficient)
    • Morphology (ragdoll joint limits, body proportions)

  Fitness function:
    For each individual → run inner loop for N steps → return final episode reward

INNER LOOP — Gradient Descent / PPO (fast, fine-grained):
  Standard marathon-envs PPO training under the meta-genome's parameters
  Reports cumulative reward back to outer GA
```

This is how **Neural Architecture Search (NAS)** works (Google Brain, 2017) and how **Eureka** (OpenAI, 2023) uses LLMs instead of a GA to propose reward functions that PPO then trains under.

For marathon-envs, the most practical target for the GA outer loop is **reward weight search**: the 7 component weights in `StyleTransfer002Agent.cs` are currently hand-tuned. A GA can find combinations that produce more natural gaits faster than manual tuning — especially for new motion styles where intuition about weight balance is poor.

### MAP-Elites: Quality-Diversity

Rather than finding one best policy, MAP-Elites finds the best policy for *each cell in a behavior space* — producing a **library of diverse strategies**:

```text
Behavior space: 2D grid of (forward_velocity × energy_efficiency)
Each cell: stores the highest-fitness policy with that behavior profile

Training:
  1. Randomly initialize a policy
  2. Evaluate: measure fitness + behavior descriptor (velocity, efficiency)
  3. Place in corresponding cell if better than current occupant
  4. Mutate the policy; repeat
```

After training, you have hundreds of policies — fast-but-wasteful, slow-but-efficient, high-stepping, shuffling, etc. — all stored simultaneously.

**Game application:** select from the library at runtime based on game context. Sprint policy when pursuing a player; stealth policy (slow, low energy signature) for ambush; injured-gait policy when health is low. Same training run produces all variants.

**Python library:** [pyribs](https://pyribs.org/) — the standard MAP-Elites implementation, compatible with any environment that returns a fitness score and behavior descriptor vector.

### Comparison for Marathon-Envs

| Algorithm | Sample efficiency | Parallelism | Sparse reward | Architecture search | Best for |
| --- | --- | --- | --- | --- | --- |
| PPO (current) | High | Limited | Struggles | No | Dense reward locomotion |
| OpenAI ES | Low | Perfect (CPU cluster) | Handles well | No | Alternative to PPO, massive CPU scale |
| NEAT | Very low | Good | Handles well | Yes | Discovering architectures |
| PBT | High | Good | Normal | Partial (hyperparams) | Auto-tuning during training |
| GA → PPO (hierarchical) | Medium | Per-individual | Normal | Yes (any parameter) | Reward weight search, morphology |
| MAP-Elites | Low | Good | Handles well | No | Building behavior libraries |

### Practical Entry Points

**Immediate (no infrastructure change):**
GA outer loop over reward weights → run `mlagents-learn` inner loop → parse TensorBoard logs for fitness. Each GA individual is one `mlagents-learn` run. Parallelizable across machines.

**Medium effort:**
OpenAI ES as a drop-in replacement for `mlagents-learn`. The ES trainer replaces PPO; the Unity environment is unchanged. [evosax](https://github.com/RobertTLange/evosax) provides ES implementations (including OpenAI ES and CMA-ES) compatible with JAX.

**Research direction:**
NEAT or HyperNEAT for joint morphology + control co-evolution. Evolve the ragdoll joint limits and body proportions alongside the control network — the agent discovers its own body plan.

---

## Parameterized Gait Styles

Masculine/feminine, young/old, drunk/sober, tired/rested — and any blend between them. Marathon-envs already has style transfer infrastructure (`StyleTransfer002Agent.cs`, 6 style environments). This section covers how to extend it to a parameterized, continuous style space.

### What Each Style Actually Changes

Every gait style maps to specific kinematic differences. Understanding these helps design both MoCap labels and reward shaping:

| Style | Stance width | Stride length | Cadence | Hip sway | Posture | Arm swing |
| --- | --- | --- | --- | --- | --- | --- |
| Masculine | Wide | Long | Medium | Low | Upright | Wide, shoulder-driven |
| Feminine | Narrow | Medium | Higher | High (lateral) | Upright | Close to body |
| Young | Medium | Long | High | Medium | Upright, bouncy | Natural |
| Old | Wide (balance) | Short | Slow | Low | Slightly hunched | Reduced |
| Energetic | Medium | Long | High | Medium | Upright | Exaggerated |
| Tired | Wide | Short | Slow | Low | Hunched, head down | Minimal |
| Drunk | Wide (unstable) | Irregular | Variable | High (random) | Swaying | Compensatory |
| Injured | Asymmetric | Short on hurt side | Slow | Compensatory lean | Guarded | Asymmetric |
| Confident | Wide | Long | Medium | Low | Very upright | Broad |
| Fearful | Narrow | Short | Quick | Tucked | Hunched | Close |

### Three Levels of Implementation

#### Level 1 — One Policy per Style (Current Approach Extended)

Train a separate network for each style using the existing `StyleTransfer002Agent` pipeline. Simple, already works, produces high-quality individual styles.

**Limitation:** no blending. You can't get "slightly drunk" or a smooth transition from sober to drunk over time. N styles = N separate training runs.

**When to use:** small fixed set of styles (4–6) where transitions aren't needed.

#### Level 2 — Conditional Policy (Continuous Style Vector)

Single network receives a style embedding `z` alongside the standard observation:

```csharp
// Observation construction in CollectObservations():
sensor.AddObservation(standardProprioception);  // joints, velocities, etc.
sensor.AddObservation(styleVector);              // e.g. float[4]
```

The style vector encodes style axes:

```text
styleVector = [instability, energy, posture_hunch, stride_width]
              [0.0 → 1.0,   0.0→1.0, 0.0→1.0,     0.0→1.0    ]

sober+rested = [0.0, 1.0, 0.0, 0.5]
drunk        = [0.8, 0.6, 0.3, 0.9]
old+tired    = [0.1, 0.2, 0.7, 0.7]
```

**Blending at runtime:**

```csharp
// Smooth style transition: sober → drunk over 10 seconds
float drunkProgress = Mathf.Clamp01(timeDrunk / 10f);
styleVector = Vector4.Lerp(soberStyle, drunkStyle, drunkProgress);
```

The policy produces a gait that smoothly interpolates between the two reference styles. No additional training needed for in-between states — the network generalizes across the style space if trained with varied style vectors.

**Training:** randomize `styleVector` each episode, provide matching MoCap reference to the DeepMimic reward. The network learns that style input predicts which reference to match.

#### Level 3 — CAMDM (Get It for Free)

CAMDM was trained on the **100STYLE dataset** — 100 distinct locomotion styles including age variations, emotional gaits, impaired gaits (drunk, tired), and character styles (masculine, feminine, military, sneaky, etc.). The pre-trained 20MB ONNX model covers all of these.

Integrating CAMDM as the kinematic layer (roadmap T2-D) gives you all 100 styles immediately:

```text
Game state (style parameter + player input)
      ↓
CAMDM (20MB ONNX, Unity Sentis, 60+ FPS)
      ↓ target pose per frame, conditioned on style
DReCon physics feedback policy (marathon-envs trained)
      ↓ physical corrective actions
PhysX ragdoll
```

Style selection is a conditioning input to CAMDM — change the style index and the kinematic target changes, which the physics layer tracks. This is the shortest path to the full style palette.

### MoCap Data for Style Training

| Dataset | Styles | License | Access |
| --- | --- | --- | --- |
| 100STYLE | 100 locomotion styles (age, emotion, impairment) | Research | [100style.edwsmit.com](https://100style.edwsmit.com) |
| CMU MoCap | Large general library, some stylistic variation | Free | mocap.cs.cmu.edu |
| AMASS | 40+ hours, diverse actors and styles | Academic | amass.is.tue.mpg.de |
| Edinburgh Loco | Specific walking style annotations | Research | homepages.inf.ed.ac.uk |
| SFU MoCap | Varied motions and actors | Free | mocap.cs.sfu.ca |

**100STYLE is the priority** — it's specifically annotated for the styles the user described and is what CAMDM was trained on.

### AdaptNet for Fast Style Addition

From the roadmap (T1-C): pre-train a base walking policy, then inject a small latent modifier conditioned on the style label. Each new style takes 10–30 minutes to adapt, not a full retraining run.

The Roblox paper explicitly demonstrates smooth interpolation in the latent space between trained styles — moving the style vector continuously produces valid in-between gaits not seen during training.

```text
Pre-trained policy (frozen) + Latent modifier (small MLP, trained per style)
      │
style_vector → modifier computes Δz
      │
base_policy_latent + Δz → modified action
```

### Physics Perturbations as Style Shortcuts

For **impairment styles**, you can bypass MoCap entirely and produce emergent style-like behavior through physics perturbations during training:

| Target style | Perturbation | Training mechanism |
| --- | --- | --- |
| Drunk | Random balance forces on torso | Agent trained with constant perturbations develops swaying recovery gait |
| Tired | Reduce all max joint torques to 30–50% | Agent with weakened joints develops energy-conserving shuffle |
| Injured leg | Weaken one leg's joints (DomainRandomization.cs) | Asymmetric gait emerges from compensating for weak side |
| Heavy armor | Add mass to torso/limbs | Slower, more deliberate movements emerge from physics |

These emergent gaits aren't identical to MoCap-recorded human impairment, but they're physically plausible and often more natural-looking than hand-animated equivalents, because the physics simulation imposes real biomechanical constraints.

### Continuous Style Evolution at Runtime

The most compelling game use case: style parameters that evolve continuously based on game state:

```csharp
void UpdateStyle() {
    // Tiredness increases with distance traveled
    styleVector.energy = 1f - Mathf.Clamp01(distanceTraveled / maxEndurance);

    // Drunkness fades over time after drinking
    styleVector.instability = Mathf.Lerp(styleVector.instability, 0f, Time.deltaTime * soberRate);

    // Injury accumulates from damage
    styleVector.asymmetry = Mathf.Clamp01(rightLegDamage / maxHealth);

    // Age is fixed per character
    styleVector.posture = characterAge / 100f;
}
```

The policy receives the updated `styleVector` each frame. Gait transitions smoothly without any state machine or animation blending logic — the network handles it continuously.

### Reward Shaping for Style Dimensions (Without MoCap)

Some style axes can be rewarded directly without reference clips:

```csharp
// Hip sway reward (masculine=low, feminine=high)
float hipSway = Mathf.Abs(hipBone.localPosition.x - previousHipPos.x);
float swayReward = targetHipSway - Mathf.Abs(hipSway - targetHipSway);

// Energy level reward (energetic=high effort allowed, tired=penalize high effort more)
float effortPenalty = GetEffortNormalized() * (1f + tirednessFactor * 2f);

// Stride width reward
float strideWidth = Mathf.Abs(leftFoot.position.x - rightFoot.position.x);
float widthReward = -Mathf.Abs(strideWidth - targetStrideWidth);
```

This approach requires no MoCap. The reward encodes the biomechanical signature of the style directly. Useful for styles that are hard to find in MoCap databases (very specific character archetypes).

### Connection to Existing Marathon-Envs

`StyleTransfer002Animator.cs` already drives a kinematic reference pose that `StyleTransfer002Agent.cs` tracks with physics. Extending to parameterized styles:

1. Replace the single FBX clip with a CAMDM model conditioned on a style vector (T2-D)
2. Or: add `styleVector` as extra observation to the existing conditional policy
3. Or: use AdaptNet latent injection on any trained walking policy (T1-C)

The physics layer (`MarathonAgent`, `BodyManager002`, `Muscle002`) is unchanged regardless of which style approach is used.
