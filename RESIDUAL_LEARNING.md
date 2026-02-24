# Residual Learning — Deep Dive with Math and Examples

This document explains **why** residual RL works, the **zero initialization** trick, **action scaling** math, and how the residual formulation makes RL training dramatically easier for robotics tasks.

Back to: [RESIDUAL_RL_TRAINING.md](RESIDUAL_RL_TRAINING.md)

---

## Table of Contents

1. [The Problem: Why Is RL Hard for Robotics?](#1-the-problem-why-is-rl-hard-for-robotics)
2. [The Idea: Don't Train From Scratch](#2-the-idea-dont-train-from-scratch)
3. [Mathematical Formulation](#3-mathematical-formulation)
4. [Zero Initialization: Starting From Nothing](#4-zero-initialization-starting-from-nothing)
5. [Action Scaling: Constraining the Correction](#5-action-scaling-constraining-the-correction)
6. [Numerical Example: End-to-End Walkthrough](#6-numerical-example-end-to-end-walkthrough)
7. [Why the Residual Learning Rate Is So Low](#7-why-the-residual-learning-rate-is-so-low)
8. [Comparison: RL From Scratch vs Residual RL](#8-comparison-rl-from-scratch-vs-residual-rl)

---

## 1. The Problem: Why Is RL Hard for Robotics?

Standard RL for robotic manipulation faces several compounding difficulties:

### Sparse Rewards

Most manipulation tasks have binary success/failure rewards:

$$r_t = \begin{cases} 1 & \text{if task completed at step } t \\ 0 & \text{otherwise} \end{cases}$$

In a 300-step episode, the agent gets **zero reward** for 299 steps and then either 1 or 0 at the end. To learn, the agent must:
1. Discover a successful trajectory by chance (extremely unlikely with random exploration)
2. Propagate the reward backward through 300+ Q-value estimates

### High-Dimensional Continuous Actions

A dual-arm robot has 14 action dimensions (7 per arm: 6 DoF + gripper). Random exploration in $[-1, 1]^{14}$ has probability approximately 0 of producing any useful behavior.

### Long Horizons

With 300-step episodes and $\gamma = 0.995$, the effective horizon is:

$$H_{\text{eff}} = \frac{1}{1 - \gamma} = \frac{1}{0.005} = 200 \text{ steps}$$

The agent must plan 200 steps ahead coherently.

### The Practical Result

Training RL from scratch on a task like `TwoArmBoxCleanup` (two arms, 14 action dims, 300-step horizon, sparse reward) would require **millions** of environment steps and might never converge at all.

---

## 2. The Idea: Don't Train From Scratch

We already have a BC policy that gets ~60-70% success rate. Instead of starting from zero, **build on top of it**:

```
Without residual:                  With residual:
                                   
RL agent must learn:               RL agent must learn:
  "Insert peg into hole"             "Adjust by 2mm to the left"
  
Difficulty: ★★★★★                  Difficulty: ★★☆☆☆
```

The base policy handles the gross motor plan. The residual handles fine adjustments — corrections that are small in magnitude but critical for success.

---

## 3. Mathematical Formulation

### Standard RL

The policy directly outputs the action:

$$a_t = \pi_\theta(s_t) + \epsilon_t$$

where $\pi_\theta$ is the learned policy and $\epsilon_t$ is exploration noise.

### Residual RL

The action is composed of two parts:

$$a_t = \underbrace{a_t^{\text{base}}}_{\text{frozen BC}} + \underbrace{a_t^{\text{res}}}_{\text{learned residual}}$$

where:
- $a_t^{\text{base}} = \pi_{\text{BC}}(s_t)$ — output of the frozen (non-trainable) BC policy
- $a_t^{\text{res}} = f_\theta(s_t, a_t^{\text{base}})$ — output of the trainable residual network

The residual network takes both the state $s_t$ AND the base action $a_t^{\text{base}}$ as input. This is important — the residual can see what the base policy suggests and decide how to correct it.

### Policy Gradient

The actor loss is:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{B}} \left[ Q_\phi(s, \text{clip}(a^{\text{base}} + a^{\text{res}}, -1, 1)) \right]$$

The gradient flows through:

$$\nabla_\theta \mathcal{L} = -\nabla_a Q \cdot \nabla_\theta a^{\text{res}}$$

Notice: the gradient only flows through $a^{\text{res}}$ (the residual network). The base policy is frozen — no gradients flow through it.

### Clamping to [-1, 1]

The combined action is clamped to the valid normalized action range:

$$a_t = \text{clip}(a_t^{\text{base}} + a_t^{\text{res}}, -1, 1)$$

If $a^{\text{base}} = 0.9$ and $a^{\text{res}} = 0.3$, the raw sum is 1.2, which clips to 1.0. This means the gradient of the clip function is zero in this region — the residual can't push beyond the action limits.

---

## 4. Zero Initialization: Starting From Nothing

### The Trick

The last layer of the residual actor has its weights and biases initialized to **exactly zero**:

```python
# In actor.py, _initialize_weights():
if cfg.actor_last_layer_init_scale is not None:  # = 0.0
    # Find the last Linear layer before Tanh
    utils.initialize_layer_weights(
        final_layer,
        distribution="normal",    # N(0, scale²)
        scale=0.0,                # scale = 0.0 → all zeros
    )
```

### What This Means

At the start of training, the MLP's last layer computes:

$$\text{output} = W_{\text{last}} \cdot h + b_{\text{last}} = \mathbf{0} \cdot h + \mathbf{0} = \mathbf{0}$$

Then through Tanh and action_scale:

$$a^{\text{res}} = \text{Tanh}(\mathbf{0}) \times 0.2 = 0 \times 0.2 = \mathbf{0}$$

So the combined action is:

$$a_t = a_t^{\text{base}} + \mathbf{0} = a_t^{\text{base}}$$

**The agent starts by doing exactly what the BC policy would do.** This is crucial:
- Performance starts at the BC baseline (~60-70% success), not zero
- The agent doesn't need to "rediscover" basic motor skills
- Gradient updates gradually push the residual away from zero only where corrections help

### Important: This Is Weight Initialization, NOT a Runtime Override

A common misconception is that the residual is "overridden to zero" for some number of initial steps and then "switched on". **That is not what happens.** The zero output is a consequence of the weight initialization — `nn.init.normal_(weight, mean=0, std=0)` sets all weights and biases to exactly 0. Since the MLP ends with `Linear → Tanh`, and `tanh(0) = 0`, the output is zero at initialization.

From the very first training step (step 0), the residual actor is **free to learn nonzero outputs**. There is no override, no schedule, no switching. The first gradient update that flows through the actor creates the first nonzero weights, and these are targeted — they push the residual in the direction that improves Q-values. The small `action_scale` (e.g., 0.2) keeps the outputs bounded, so the residual moves away from zero **gradually and only where corrections help**.

### Why Not Small Random Instead of Zero?

Small random initialization (e.g., `scale=1e-3`) would add tiny random perturbations to every action at every timestep. While individual perturbations are small, their cumulative effect over a 300-step episode can be significant — the trajectory diverges from the BC policy's trajectory.

Zero initialization guarantees the **exact same trajectory** as the BC policy at step 0. The first gradient update creates the first non-zero weights, and these weights are targeted — they push the residual in the direction that improves Q-value.

---

## 5. Action Scaling: Constraining the Correction

### The Problem

Without constraints, the residual could learn to output large actions that completely override the base policy. This defeats the purpose of residual learning and makes training unstable.

### The Solution: Tanh × action_scale

The actor architecture enforces a hard bound on the residual:

```
MLP output → Tanh → × action_scale
                          ↓
              Tanh ∈ [-1, 1]
              × 0.2
              → output ∈ [-0.2, 0.2]
```

With `action_scale = 0.2`, the residual is constrained to:

$$a^{\text{res}} \in [-0.2, 0.2]^d$$

in the **normalized** action space (where the full range is [-1, 1]).

### How the Output Range Works (No Division Involved)

The actor's output range is **not** [-1, 1] divided by something. The flow is:

1. The MLP produces raw logits (any real number)
2. `Tanh` squashes them to **[-1, +1]**
3. **Multiply** by `action_scale`: `scaled_mu = tanh(logits) × 0.2` → **[-0.2, +0.2]**

```python
# In actor.py forward():
mu = self.policy(policy_input)       # Tanh output → [-1, 1]
scaled_mu = mu * self.cfg.action_scale  # × 0.2 → [-0.2, 0.2]
```

This is a direct multiplication, not a division. The Tanh already bounds the output to [-1, 1], and the `action_scale` simply shrinks that range.

During **evaluation** (`eval_mode=True`), the action is exactly `scaled_mu` — strictly bounded to [-0.2, +0.2].

During **training**, `TruncatedNormal(scaled_mu, std=0.025)` samples around the mean with exploration noise, so actual sampled actions can slightly exceed ±0.2 depending on `std`, but the distribution is centered within that range.

### What ±0.2 Means in Practice

Consider a single action dimension — say, the x-position of a robot's end effector:

| Value | Meaning |
|---|---|
| Raw action range | [-0.05, 0.05] m/step (from dataset min/max) |
| Normalized range | [-1, 1] |
| Base action | e.g., 0.3 (normalized) |
| Residual range | [-0.2, 0.2] (normalized) |
| Combined range | [0.1, 0.5] (normalized) |
| In raw units | residual = ±0.2 × 0.05 m = ±0.01 m = ±1 cm |

The residual can adjust the end-effector position by at most ±1 cm per step. This is enough for fine positioning corrections but prevents wild, destructive movements.

### action_scale and ActionScaler Interaction

The `action_scale` parameter appears in **two** different places with different effects:

1. **Actor's `action_scale` (0.2)**: Multiplies the Tanh output → constrains residual magnitude
2. **ActionScaler's `action_scale` (0.2)**: Expands the normalization range by `(1 + 0.2) = 1.2×` → leaves room for the residual to push beyond the demonstrated action range

The second one is important: if the normalization range exactly matches the dataset's min/max, then any action at the boundary (e.g., normalized = 1.0) plus a positive residual would be clipped. Expanding the range by 20% ensures there's "headroom" for corrections.

**They are separate mechanisms connected by the same config value.** The command-line arg `agent.actor.action_scale=0.2` sets the actor's output bound. This same value is also passed to `ActionScaler.from_dataset_stats(action_scale=cfg.agent.actor.action_scale)` to expand the normalization range. The dataset statistics (min/max) define the *center and shape* of the normalization — `action_scale` does NOT overwrite those stats. It just tells the scaler "allow 20% extra headroom beyond what the dataset contained" so combined actions (base ± residual) can still be unmapped to valid raw actions.

---

## 6. Numerical Example: End-to-End Walkthrough

Let's trace through one complete step with concrete numbers.

### Setup

- Task: TwoArmBoxCleanup
- Action dimension: 14 (7 per arm)
- action_scale: 0.2
- We focus on one dimension: `joint_3_velocity` of robot arm 1

### Dataset Statistics for `joint_3_velocity`

```
action_min[3] = -0.083  (raw units, from dataset)
action_max[3] = 0.071   (raw units)
```

### ActionScaler Initialization

```
mid = (-0.083 + 0.071) / 2 = -0.006
half_range = (0.071 - (-0.083)) / 2 = 0.077
expanded_half_range = 0.077 × (1 + 0.2) = 0.0924

final_min = -0.006 - 0.0924 = -0.0984
final_max = -0.006 + 0.0924 = 0.0864
```

### At Timestep t

**1. Base policy outputs raw action:**

$$a_{\text{base}}^{\text{raw}} = 0.025 \text{ (in raw units)}$$

**2. Scale to [-1, 1]:**

$$a_{\text{base}}^{\text{norm}} = 2 \times \frac{0.025 - (-0.0984)}{0.0864 - (-0.0984)} - 1 = 2 \times \frac{0.1234}{0.1848} - 1 = 2 \times 0.668 - 1 = 0.335$$

**3. Residual actor outputs correction:**

The actor MLP produces (before Tanh): $z = 0.15$

$$a_{\text{res}}^{\text{norm}} = \tanh(0.15) \times 0.2 = 0.149 \times 0.2 = 0.0298$$

**4. Combine:**

$$a_{\text{combined}}^{\text{norm}} = \text{clip}(0.335 + 0.0298, -1, 1) = 0.365$$

**5. Unscale to raw action:**

$$a_{\text{combined}}^{\text{raw}} = -0.0984 + \frac{0.365 + 1}{2} \times 0.1848 = -0.0984 + 0.6825 \times 0.1848 = -0.0984 + 0.1262 = 0.0278$$

**6. Effect of the residual:**

$$\Delta a^{\text{raw}} = 0.0278 - 0.025 = 0.0028 \text{ (raw units)}$$

The residual changed the joint velocity by 0.0028 units — a small but potentially critical adjustment.

---

## 7. Why the Residual Learning Rate Is So Low

The actor learning rate is `1e-6` — **100× lower** than the critic's `1e-4`. Here's why:

### Risk of Catastrophic Forgetting

The base policy represents thousands of gradient steps of BC training. If the residual changes too quickly, it can:
1. Push the combined action far from the demonstrated distribution
2. Create trajectories the critic has never seen → garbage Q-estimates
3. The actor follows garbage gradients → policy collapses → success rate drops to 0%

### Conservative Updates

With `lr = 1e-6` and gradient clipping at norm 1.0:

$$\Delta \theta_{\text{actor}} = -\text{lr} \times \nabla_\theta \mathcal{L} \approx 10^{-6} \times \text{direction}$$

This means each update changes the actor weights by approximately one millionth of the gradient magnitude. Over 500,000 steps with 1 actor update per step = 125,000 actor updates (UTD=4, actor every 4th), the cumulative change is small enough to stay near the BC policy's behavior while gradually improving.

### Learning Rate Hierarchy

```
critic_lr  = 1e-4    (critic must be ahead of actor)
encoder_lr = 1e-4    (same as critic — encoder trained by critic loss)
actor_lr   = 1e-6    (very conservative — actor changes slowly)
```

This 100× ratio ensures the critic's Q-estimates stabilize before the actor tries to exploit them. If the actor moved faster than the critic, it would chase phantom Q-values.

---

## 8. Comparison: RL From Scratch vs Residual RL

### Exploration Efficiency

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Initial success rate | ~0% | ~60-70% (BC baseline) |
| Exploration strategy | Random in $[-1,1]^{14}$ | BC policy ± 0.2 |
| Steps to first success | ~100,000+ | ~300 (BC does it) |
| Useful experience in buffer | <1% early on | ~60%+ from the start |

### Learning Stability

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Q-value scale | Unknown, drifts | Anchored near 0–1 (sparse reward) |
| Gradient signal | Very noisy | Stable (50% offline demos) |
| Policy collapse risk | High | Low (constrained by action_scale) |
| Recovery from bad updates | Very difficult | Automatic (residual can return to ~0) |

### Computational Cost

| Aspect | RL From Scratch | Residual RL |
|---|---|---|
| Typical timesteps needed | 2M–10M | 200K–500K |
| Wall-clock time | Days | Hours |
| GPU memory (extra) | — | Base policy forward pass (small overhead) |

### When Residual RL Can Fail

1. **Base policy is terrible** (<10% success) — not enough good trajectories to build on
2. **Task requires fundamentally different strategy** — residual can only adjust, not replan
3. **action_scale too small** — the residual literally can't make the needed correction
4. **action_scale too large** — the residual overpowers the base, reverting to scratch RL

The 0.2 action_scale is a sweet spot: large enough for meaningful corrections, small enough to keep the policy well-behaved.

Back to: [RESIDUAL_RL_TRAINING.md](RESIDUAL_RL_TRAINING.md)
