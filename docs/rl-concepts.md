# Reinforcement Learning Concepts

## PPO vs SAC

| | PPO | SAC |
| --- | --- | --- |
| Type | On-policy | Off-policy |
| Sample efficiency | Lower | Higher |
| Replay buffer | No | Yes — reuses old experiences |
| Exploration | Entropy bonus | Automatic entropy tuning |
| Stability | Very stable | Stable but more hyperparams |
| Best for | Many parallel envs | Fewer envs, longer training |

SAC typically reaches the same locomotion quality as PPO in 3× fewer steps. This repo supports both — PPO via `mlagents-learn`, SAC via `stable_baselines_sac_train.py`.

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
