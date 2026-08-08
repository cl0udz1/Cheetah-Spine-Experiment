# Does an active spine help a quadruped run? A controlled MuJoCo study under CPG control

A cheetah-inspired quadruped with a 3-DOF active spine, measured against a rigid trunk and a passive compliant trunk under identical hand-written CPG controllers. **The spine buys roughly 1–2% forward speed and 8% turn rate, and pays for it with about 80% worse path tracking. Most of the turning benefit is a constant ~16° yaw offset — articulated steering rather than undulation. The direction of the flexion asymmetry matters more than its presence: running it backwards costs 11%.**

The qualifier in the title is load-bearing: every result here is conditional on the controller being a fixed CPG rather than a learned policy, and on a trot rather than a bound. See [What this does and does not show](#what-this-does-and-does-not-show).

![Side-by-side animation of two identical quadrupeds executing the same 0.8 rad/s turn command, the left one with an active 3-DOF spine and the right one with a rigid trunk. Both circle at visibly similar rates; the spine model's trunk holds a slight constant bend rather than visibly undulating.](docs/media/hero_turn.gif)

*Active spine (left) and rigid trunk (right) running the same turn command from the same initial state. The spine model turns 8.1% faster. Watch the trunk: it holds a steady bend rather than oscillating — that constant offset supplies about two thirds of the advantage.*

## TL;DR

- **Straight-line running: a small but real gain.** At 1.0 m/s the active spine reaches 0.8724 ± 0.0007 m/s against rigid's 0.8639 ± 0.0003 (**+0.99%**); at 2.0 m/s, 1.8315 ± 0.0066 against 1.7992 ± 0.0088 (**+1.79%**). Both sit outside the seed spread.
- **The gain is active, not compliance.** The passive spine returns +0.25% and +0.21% — indistinguishable from rigid. Whatever the spine is doing, it does with its motors.
- **Asymmetry direction dominates.** Holding amplitude and phase fixed and varying only the flexion:extension ratio: 2.0 flexion-dominant (biological) gives +0.99%/+1.79%, 1.0 symmetric gives −1.59%/+1.65%, and 0.5 extension-dominant gives **−11.44%/−7.20%**. Running the asymmetry backwards costs far more than not having one.
- **Turning: +8.12% yaw rate** (0.7788 ± 0.0010 vs 0.7203 ± 0.0016 rad/s), and again it requires actuation — the passive spine is 6.75% *worse* than rigid.
- **Most of the turning benefit is steering, not undulation.** A spine holding only a static yaw bias reproduces 64% of the advantage. Undulation *alone*, with the bias removed, is **12.6% worse than rigid**; it contributes only in combination with the bias.
- **The cost is path tracking.** Straight-line cross-track error is 85% worse at 1.0 m/s and 77% worse at 2.0 (0.893 ± 0.014 vs 0.483 ± 0.003 m). Energy is a wash — cost of transport is within 0.02% at 1.0 m/s and 2.26% *cheaper* at 2.0.
- The biological framing survives only in a weak form. The flexion-dominant asymmetry genuinely is the best of the three directions tested, but the payoff is a low-single-digit speed gain in a trot — and a trot is not the gait the biological argument concerns.

> **Correction, August 2026.** An earlier version of this README reported *no* straight-line effect and described the realised flexion:extension ratio as "heavily attenuated". Both were artefacts of an inverted waveform: a `flexion_sign` parameter defaulting to `-1.0` ran the asymmetry backwards, so every undulation result was measured on an extension-dominant spine. All of them have been re-measured. See [Errors found and corrected](#errors-found-and-corrected), item 10.

## Background

The MIT Biomimetic Robotics Lab's early [Cheetah](https://biomimetics.mit.edu/) designs incorporated a compliant, articulated spine element, motivated directly by the sagittal flexion of a galloping cheetah. By the time of **Cheetah 3** (Bledt, Powell, Katz, Di Carlo, Wensing and Kim, *MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot*, IROS 2018), the trunk was rigid. The spine had been engineered away: it added mass, actuators, control complexity and failure modes, and the performance case for it did not hold up.

**S-Cheetah** ([arXiv:2605.27909](https://arxiv.org/abs/2605.27909), ShanghaiTech, 2026) revived the idea, adding a 3-DOF active spine (yaw → pitch → roll) and reporting locomotion gains. Those gains were reported **in simulation only, with no hardware**.

That is the disagreement this repo examines: two credible groups reached opposite conclusions about whether a quadruped should have a spine.

**What this repo is not.** It is not a replication of S-Cheetah's result. Their gains come from a *learned* controller; this study uses hand-written open-loop and closed-loop CPG controllers. A learned policy can co-adapt its gait to a spine in ways a fixed CPG cannot, and that is precisely where a spine would be most likely to pay off. What this repo establishes is a carefully controlled CPG baseline, and the finding that under that baseline the biological mechanism does not appear.

## Method

### Robot model

`spine_quadruped.xml` — a cheetah-proportioned quadruped, 6.769171 kg, MuJoCo `implicitfast` integrator at a 2 ms timestep.

- 12 leg joints (abduction, hip pitch, knee, per leg), torque limit ±33.5 N·m
- 3 spine joints in **yaw → pitch → roll** order, matching the S-Cheetah ordering, torque limit ±50 N·m
- 2-DOF weighted tail (yaw, pitch), torque limit ±8 N·m
- Thigh and calf 0.22 m each; hip-to-foot reach 0.363 m in the standing pose

### The three variants

All three are **mass-matched at 6.769171 kg**, verified at build time. That is what makes the comparison a test of trunk topology rather than of mass distribution.

| variant | construction | spine joints | spine actuators | nq | nv | nu | njnt |
|---|---|---|---|---|---|---|---|
| `spine` | unmodified: 3 spine joints, motor-driven | **3** | **3** | 24 | 23 | 17 | 18 |
| `rigid` | spine `<joint>` and `<motor>` elements **deleted**; MuJoCo welds the child body to its parent at compile time | **0** | **0** | 21 | 20 | 14 | 15 |
| `passive` | spine joints kept with `stiffness=400`, `damping=12`; **all spine motors deleted** | **3** | **0** | 24 | 23 | 14 | 18 |

`rigid` and `passive` both have `nu=14`, but for opposite reasons: `rigid` has no spine joints to drive, while `passive` keeps all three joints and simply has no motors on them. The distinguishing numbers are `nv` — 20 for `rigid` against 23 for `passive` and `spine` — and the spine-actuator column. `build_model` asserts both.

The rigid variant deletes joints rather than stiffening them. Stiffening a hinge to fake a weld degrades the mass-matrix conditioning and the solver diverges while still returning finite-looking numbers — a failure mode this study measured directly (see the calibration table below, `kp=1200` and `kp=4000` rows).

Construction is verified against the *compiled* model, never against the XML edit. `build_model` raises `ModelBuildError` if the surviving joint set differs from the request, if a removed motor is still present, if any actuator can still reach a spine joint in the passive variant, or if a kept passive joint has no spring. A regex that silently matches nothing is exactly how a fake "rigid" result gets made.

The `passive` variant exists to answer one question: **is any benefit compliance or control?** A PD controller holding a joint at zero with gains (kp, kd) is exactly a spring-damper of stiffness kp and damping kd, so the passive defaults match `kp_spine`/`kd_spine` and a passive spine is the same mechanical system as a held actuated one.

### Controllers

**Open-loop CPG.** Trot with an explicit stance/swing split and duty factor: the planted foot sweeps through stance while the knee flexes to clear ground through swing. Stride amplitude follows from the commanded speed geometrically — a foot planted through a stance sweep of 2A carries the body 2·A·0.363 m per cycle. Amplitudes ramp in over 0.6 s via smoothstep; a step discontinuity at gait onset saturates every actuator and throws the robot before it takes a stride.

Leg gait parameters (`freq=2.6`, `duty=0.75`, `knee_amp=0.95`, `kp_leg=150`, `kd_leg=6`) were selected by grid search **on the rigid model only**, scoring the worst case across two commanded speeds rather than the best. The rigid variant has no spine, so no spine setting could influence the choice. This biases the baseline in the rigid variant's favour, making any spine advantage measured against it conservative.

**Closed-loop layer.** Two feedback loops, both necessary for the metrics to mean anything:

- *Speed*: a PI loop on measured forward speed modulating gait frequency, with conditional-integration anti-windup. Without it, commanding 1.0 m/s produced 1.84 m/s and the command was a label rather than a target.
- *Path*: heading feedback plus a **cross-track position term**. Heading feedback alone cannot fix lateral error — driving yaw to the reference heading just makes the robot run *parallel* to the path at whatever offset it already had.

Spine steering follows the *commanded* yaw rate, not the feedback-corrected one; routing the correction through the spine as well as the legs double-counts it (see Errors, below).

Gait phase is integrated (`phase += freq·dt`) rather than computed as `freq·t`, because under a speed loop `freq(t)·t` is not the phase of a frequency-modulated oscillator.

**Asymmetric flexion.** A real cheetah's spine flexes roughly twice as far as it extends, so a symmetric sinusoid is the wrong prior. `asymmetric_wave(phase, ratio)` produces a unit oscillation whose flexion excursion is `ratio` times its extension; at `ratio=1` it degenerates to `sin()`, which is the control condition for testing whether the asymmetry matters at all.

### The calibration this study rests on

A spine **held at neutral must be dynamically equivalent to a deleted spine.** If it is not, every spine-vs-rigid number measures how well the spine is held rather than trunk topology. Eight seeds, 8 s, straight command, from `tools/calibration_table.py`:

| spine hold | net speed @1.0 | falls | deflection | net speed @2.0 | falls | deflection |
|---|---|---|---|---|---|---|
| **rigid (joints deleted)** | 0.864 ± 0.000 | 0/8 | — | 1.799 ± 0.009 | 0/8 | — |
| actuated, kp=120 kd=4 | 0.860 ± 0.000 | 0/8 | 2.36° | 1.799 ± 0.015 | 0/8 | 4.49° |
| **actuated, kp=400 kd=12** | **0.866 ± 0.000** | **0/8** | **0.78°** | **1.793 ± 0.017** | **0/8** | **1.71°** |
| actuated, kp=1200 kd=40 | 0.841 ± 0.001 | 0/8 | 1.04° | **0.995 ± 0.440** | **6/8** | 1.07° |
| actuated, kp=4000 kd=120 | 0.808 ± 0.004 | 0/8 | 2.14° | **1.429 ± 0.418** | **3/8** | 2.26° |
| passive spring k=400 c=12 | 0.866 ± 0.000 | 0/8 | 0.79° | 1.803 ± 0.010 | 0/8 | 1.81° |
| passive spring k=100 c=6 | 0.857 ± 0.000 | 0/8 | 2.49° | 1.781 ± 0.047 | 1/8 | 4.48° |

`kp_spine=400` is the operating point: a held spine matches the deleted spine at both speeds, with sub-2° residual deflection. Two things fall out of this table. First, the equivalence must be checked at *every* speed the study reports — `kp=120` looks fine at 1.0 m/s and is back-driven 4.49° at 2.0 m/s. Second, **stiffening breaks down exactly as expected**: at `kp≥1200` the robot falls on 3–6 of 8 seeds with the speed variance exploding to ±0.44. That is the numerical penalty for faking a weld with stiffness, and it is why the rigid variant deletes joints.

### Metrics

- **peak speed** — maximum of a 0.1 s moving average of body-frame forward speed. The raw per-step maximum is contact-impulse noise, not a speed anyone can use.
- **net progress** — net displacement projected on the *starting heading*, divided by elapsed time. **Signed**: negative when the robot goes the wrong way.
- **turn rate** — mean signed yaw rate, from unwrapped yaw.
- **cost of transport** — dimensionless, mechanical energy ∫Σ|τ·ω|dt divided by (m·g·distance). NaN rather than infinite when the robot did not move.
- **cross-track error** — RMS distance from the actual path to the *reference curve*, speed-independent. Reported separately from `path_error_rms_m`, which is measured against the time-parameterised reference and so conflates going off course with going slowly.
- **speed tracking error** — steady-state achieved speed (last 60% of the run) minus commanded. This is the number that decides whether a config label like `straight_1.0` is honest.

### Divergence policy

Any rollout that trips the stability monitor — non-finite `qpos`/`qvel`/`qacc`, velocity or position bounds, or MuJoCo's `mjWARN_BADQACC`/`BADQVEL`/`BADQPOS` counters — has **every** metric forced to NaN and prints an unmissable banner. There is no partial-credit path where the first 60% of a blown-up rollout gets averaged into a plausible number. NaN is visible in a CSV; a plausible float is not.

## Results

All results: 8 seeds with randomised initial joint angles, drop height, heading and trunk velocity; 8 s per rollout; identical leg gait across variants. **Zero falls and zero divergences in every row below.**

### Straight-line running: a small active gain, paid for in path tracking

| command | variant | net progress (m/s) | vs rigid | peak speed (m/s) | cost of transport | cross-track (m) |
|---|---|---|---|---|---|---|
| 1.0 m/s | spine | **0.8724 ± 0.0007** | **+0.99%** | 1.1163 ± 0.0025 | 1.7677 ± 0.0010 | **0.893 ± 0.014** |
| 1.0 m/s | rigid | 0.8639 ± 0.0003 | — | 1.0670 ± 0.0017 | 1.7680 ± 0.0009 | 0.483 ± 0.003 |
| 1.0 m/s | passive | 0.8661 ± 0.0002 | +0.25% | 1.0650 ± 0.0027 | 1.7523 ± 0.0010 | 0.519 ± 0.003 |
| 2.0 m/s | spine | **1.8315 ± 0.0066** | **+1.79%** | 2.2873 ± 0.0112 | 1.3980 ± 0.0047 | **0.849 ± 0.120** |
| 2.0 m/s | rigid | 1.7992 ± 0.0088 | — | 2.3282 ± 0.0055 | 1.4302 ± 0.0077 | 0.480 ± 0.034 |
| 2.0 m/s | passive | 1.8030 ± 0.0101 | +0.21% | 2.3007 ± 0.0125 | 1.4236 ± 0.0097 | 0.527 ± 0.045 |

The active spine is faster at both commands, by margins that sit outside the seed spread. The passive spine is not — +0.25% and +0.21% are inside it. So the gain comes from driving the spine, not from having a compliant one.

It is a small gain and it is not free. Cross-track error is **85% worse at 1.0 m/s and 77% worse at 2.0 m/s**: the undulating trunk pushes the body laterally every cycle and the path controller only partly rejects it. Energy is roughly neutral — cost of transport is within 0.02% at 1.0 m/s and 2.26% cheaper at 2.0.

Note also that peak speed and net progress disagree in sign at 2.0 m/s: the spine has the *lower* peak (2.287 vs 2.328) but the *higher* net progress. It is sustaining speed better rather than reaching a higher one.

The asymmetry direction is what actually matters. Holding amplitude at 0.05 rad and phase at 0.25 and varying only the flexion:extension ratio, 8 seeds:

| flexion:extension | net @1.0 m/s | vs rigid | net @2.0 m/s | vs rigid | realised ratio |
|---|---|---|---|---|---|
| **2.0** flexion-dominant (biological) | 0.8724 ± 0.0007 | **+0.99%** | 1.8315 ± 0.0066 | **+1.79%** | 2.215 |
| 1.0 symmetric sinusoid | 0.8502 ± 0.0006 | −1.59% | 1.8290 ± 0.0073 | +1.65% | 1.205 |
| 0.5 extension-dominant | 0.7650 ± 0.0009 | **−11.44%** | 1.6697 ± 0.0306 | **−7.20%** | 0.600 |

Monotone and far outside noise. The biological direction is the best of the three, and getting it backwards is roughly ten times more costly than the benefit of getting it right. The realised ratios confirm the commanded asymmetry reaches the joint accurately in all three cases.

![Top-down trajectory plot for the 1.0 m/s straight command showing the commanded straight path as a dashed grey line and three near-identical robot tracks overlaying it closely.](docs/media/trajectory_straight_1.0.png)

*Straight-line path tracking at 1.0 m/s, seed 0. This is the cost of the spine made visible: the spine track (blue) ends about 1.0 m off the reference against rigid's 0.72 m and passive's 0.36 m, which is the 85% cross-track penalty in the table. The commanded path (dashed) is not horizontal because the reference is seeded from the robot's randomised initial heading, so the controller is tracking a slightly rotated line. Note also that none of the three converges back onto it — the path loop reduces the residual from the 2.3 m of the open-loop version (see Errors item 8) but does not eliminate it.*

![Contact sequence diagram with one row per foot, dark bars marking ground contact, showing a regular alternating diagonal trot pattern.](docs/media/gait_straight_1.0.png)

*Contact sequence at 1.0 m/s. Look for the diagonal pairing — front-left with rear-right, front-right with rear-left — and its regularity. This is what a working trot looks like; the asymmetry between front and rear stance durations is a property of the gait, not of the spine.*

### Turning: the one real effect, and it needs actuation

| variant | turn rate (rad/s) | vs rigid | cross-track (m) | cost of transport |
|---|---|---|---|---|
| spine | **0.7788 ± 0.0010** | **+8.12%** | 0.252 ± 0.005 | **2.1290 ± 0.0024** |
| rigid | 0.7203 ± 0.0016 | — | 0.247 ± 0.003 | 2.2010 ± 0.0090 |
| passive | 0.6717 ± 0.0013 | −6.75% | **0.232 ± 0.006** | 2.3005 ± 0.0099 |

Commanded yaw rate was 0.8 rad/s; all three undershoot. The active spine undershoots least, and does so 3.3% more cheaply than rigid.

The passive spine being *worse* than rigid is the load-bearing observation. Compliance alone does not help turning — it hurts. Whatever the spine is doing, it is doing it with its motors.

Unlike straight-line running, turning costs the spine nothing in tracking: cross-track is 0.252 against rigid's 0.247, a 2.2% difference that is barely outside the seed spread. The lateral disturbance the undulation injects is small next to the commanded yaw.

![Top-down trajectory plot for the 0.8 rad/s turn command. A dashed grey commanded circle is shown with three robot tracks, all of which trace circles larger than commanded, with the spine track tighter than rigid but still outside the reference.](docs/media/trajectory_turn_0.8.png)

*Turning at 0.8 rad/s. All three variants trace a real circle — compare against the pre-fix behaviour described in Errors, where neither variant turned at all. The spine track sits closer to the commanded circle than rigid here, but its RMS cross-track over the full run is higher; the advantage in rate does not translate into an advantage in tracking.*

[Side-by-side MP4: spine vs rigid on the turn command](docs/media/sidebyside_turn_0.8.mp4) (2.5 MB — GitHub shows relative-path MP4s as a download link, not an inline player; the GIF at the top of this page is the autoplaying version.)

### Mechanism: articulated steering, not undulation

The spine's yaw joint holds a near-constant −16° offset during a turn rather than oscillating. Decomposing the spine yaw command into its static and oscillatory parts and testing each alone, at 8 seeds:

| configuration | turn rate (rad/s) | vs rigid | spine yaw bias | spine yaw peak-to-peak |
|---|---|---|---|---|
| rigid | 0.720 | — | — | — |
| spine: full (bias + undulation) | **0.779** | **+8.1%** | −16.12° | 3.33° |
| **spine: static bias only** | **0.758** | **+5.2%** | −16.13° | 0.86° |
| **spine: undulation only** | **0.630** | **−12.6%** | −0.12° | 3.04° |
| spine: held neutral | 0.674 | −6.4% | −0.15° | 0.93° |
| passive spring | 0.672 | −6.7% | −0.15° | 0.96° |

**The static bias alone reproduces 64% of the advantage. The undulation alone is 12.6% worse than rigid** — worse even than holding the spine still.

The two do not add. Undulation on its own is strongly negative, yet full (bias + undulation) at 0.779 beats bias-alone at 0.758. The remaining 36% therefore comes from an interaction: the oscillation is useful only once the trunk is already bent into the turn, and harmful otherwise. That is a real coupling, not a second independent mechanism.

The dominant term remains articulated steering — bending the body into the turn and holding it there, the way a bus or tractor-trailer steers. That is a legitimate use for a spine DOF, and it is not the mechanism the biological argument proposes.

![Time series of the three spine joint angles during a turn, showing spine yaw holding a roughly constant negative offset near -16 degrees with small ripple, while pitch and roll oscillate at low amplitude around zero.](docs/media/spine_angles_turn_0.8.png)

*Spine joint angles during the turn. Two things to look at. The blue trace — spine yaw — settles to a sustained −16° offset within 0.6 s and oscillates about that, rather than about zero: that offset is the articulated-steering term. The red trace — spine pitch — is visibly asymmetric, reaching further below zero than above, which is the corrected flexion-dominant waveform. Before the fix described in Errors item 10, that trace was asymmetric the other way.*

The realised sagittal flexion:extension ratio is 2.06 ± 0.01 at the 1.0 m/s command and 2.37 ± 0.04 at 2.0 m/s, against a commanded 2.0 — so the asymmetry reaches the joint essentially intact, with no meaningful attenuation. Sign convention was verified against the model rather than assumed: setting `spine_pitch` to −0.4 rad puts the tail base at z = 0.297 m and +0.4 rad puts it at z = 0.516 m, against 0.408 m at neutral, so negative `spine_pitch` lowers the hind end and is flexion.

An earlier version of this section reported 0.43–0.66 and called it attenuation. A ratio below 1.0 cannot be attenuation — it means the joint extends further than it flexes. That was the symptom of the inverted waveform described in [Errors](#errors-found-and-corrected), item 10.

### Speed tracking

| command | variant | commanded (m/s) | achieved (m/s) | error (m/s) |
|---|---|---|---|---|
| straight | spine | 1.0 | 0.9532 ± 0.0002 | −0.047 |
| straight | rigid | 1.0 | 0.9305 ± 0.0003 | −0.070 |
| straight | passive | 1.0 | 0.9341 ± 0.0001 | −0.066 |
| straight | spine | 2.0 | 1.9980 ± 0.0046 | −0.002 |
| straight | rigid | 2.0 | 1.9719 ± 0.0105 | −0.028 |
| straight | passive | 2.0 | 1.9701 ± 0.0139 | −0.030 |
| turn | spine | 1.0 | 0.8710 ± 0.0007 | −0.129 |
| turn | rigid | 1.0 | 0.8512 ± 0.0049 | −0.149 |
| turn | passive | 1.0 | 0.8135 ± 0.0039 | −0.187 |

Commanded speeds are tracked to within 7% at 1.0 m/s and 1.5% at 2.0 m/s, so the configuration labels mean what they say. Under a turn command all variants lose 13–19% of commanded speed, which is expected — turning costs forward progress. The residual shortfall is listed in [Limitations](#what-this-does-and-does-not-show).

![Forward speed against time at the 2.0 m/s command, showing the three variants converging to and holding a speed close to the commanded value marked by a dashed reference line.](docs/media/speed_straight_2.0.png)

*Forward speed at the 2.0 m/s command. Look at where the traces settle relative to the dashed commanded line, and at how closely the three variants track together — the spine's 1.79% advantage is real but is not visible at this scale.*

## What this does and does not show

**Shows.** Under a tuned CPG controller with closed-loop speed and path tracking, on this robot: an active 3-DOF spine gives a small straight-line gain over a rigid trunk (+0.99% at 1.0 m/s, +1.79% at 2.0 m/s) that a passive compliant spine does not reproduce (+0.25%, +0.21%), so the benefit requires actuation rather than compliance; the gain is bought with 77–85% worse cross-track error at roughly neutral energy cost; the spine's 8.12% turning advantage is real, also requires actuation, and is 64% attributable to a static yaw bias with the remainder coming from an interaction that only appears once the trunk is already bent; and the direction of the flexion asymmetry matters an order of magnitude more than its presence, with the anti-biological direction costing 7–11%.

**Does not show.** That an active spine is useless in general. Specifically:

- **One morphology.** A single robot at 6.77 kg with fixed segment lengths and mass distribution. Spine benefit plausibly depends on trunk length, mass fraction and leg-to-spine inertia ratio, none of which were varied.
- **One turn rate.** Turning was tested only at 0.8 rad/s. The turning result may not extrapolate to tighter or gentler turns, and the spine's overshoot suggests its behaviour is rate-dependent.
- **CPG, not learned control.** This is the central limitation. S-Cheetah's claim is about learned policies. A fixed CPG cannot co-adapt gait timing, footfall placement and spine phase together, and that coupling is where a spine would most plausibly pay off. This study does not refute their result; it establishes that the effect does not appear under hand-designed control.
- **Trot only.** The gait is a trot. A bounding or galloping gait loads the sagittal spine very differently, and that is the gait the biological argument is actually about. `bound` is implemented but showed continuous yaw instability (−7.72°/s) and was not used for the comparison.
- **Simulation only, no hardware.** Same limitation S-Cheetah has. Contact modelling, actuator dynamics and structural compliance are all idealised. No claim here transfers to a physical robot without validation.
- **Tracking trade-off unresolved.** The spine's straight-line speed gain of 1–2% comes with 77–85% worse cross-track error, because the undulating trunk injects a lateral disturbance every cycle. Whether that is a net benefit depends entirely on whether the application values speed or path accuracy, and this study does not settle which matters. It is also possible that a path controller tuned specifically for an undulating trunk would recover most of the tracking loss; the one used here was tuned on the rigid variant.
- **Commanded speed is not reached exactly.** The PI speed loop leaves a systematic steady-state shortfall: −6.6 to −7.0% at a 1.0 m/s command, −1.4 to −2.0% at 2.0 m/s, and −14.9 to −18.7% under a turn command. The error is consistent across variants, so it does not bias the comparison, but "1.0 m/s" means 0.93 m/s and absolute speeds should be read with that in mind. The residual is integral-limited rather than authority-limited; a longer episode or a higher `ki_speed` would reduce it at the cost of stability margin.

## Errors found and corrected

Every result above was produced by code that had, at some point, a defect producing confident and wrong numbers. This section exists because the credibility of the rest depends on it.

**1. Inverted hip sign — the robot ran backwards at 1.7 m/s.** `hip_pitch` rotates about +y, and R_y maps the downward leg vector (0,0,−L) to x = −L·sin(θ), so *increasing* hip pitch drives the foot toward −x. The stance sweep ran the wrong way. The robot trotted smoothly in reverse and **survived an entire 108-point grid search**, ranking first, because the search scored an unsigned displacement norm — which scores fast-reverse as fast. Caught by noticing that the top-ranked configuration had `peak_speed ≈ −0.007` m/s alongside a 1.688 m/s "net displacement". Fixed in `05d7779`; the metric that hid it fixed in `99aa744`.

**2. Unsigned speed metrics.** The direct enabler of error 1. `net_displacement_speed_mps` used `‖p_end − p_start‖`, discarding direction. Replaced with `net_progress_speed_mps`, the displacement projected on the starting heading, which goes negative when the robot goes the wrong way. `99aa744`.

**3. Deterministic seeds averaging identical rows.** The CPG has no noise source, so `--seeds 0,1,...,7` produced eight bit-identical rollouts and the reported standard deviations were exactly zero by construction. Averaging eight copies of one number is not evidence. Fixed by randomising initial joint angles, drop height, heading and trunk velocity per seed (`42c52a2`). The immediate consequence: the spine variant turned out to fall on 4 of 8 seeds under settings that had looked clean on the single deterministic run.

**4. Command never reached the controller.** The commanded velocity reached the metrics' reference path but not the controller, so `straight_1.0`, `straight_2.0` and a 0.8 rad/s turn produced three *identical* result rows. The sweep was silently comparing nothing. Caught by noticing the identical rows. `876650b`.

**5. Inverted turn gain — nearly reported "the spine is worse at turning".** `turn_spine_gain` had the wrong sign, so the spine yaw bias fought the differential-stride turn instead of assisting it, cutting achieved yaw rate to 0.466 rad/s against a commanded 0.8. This was about to be written up as a spine deficit. Testing the sign explicitly gave 1.110 rad/s at −0.35 and 1.313 at −0.70. `b781c70`.

**6. Withdrawn claim: the +8.4% straight-line advantage at 2 m/s.** An earlier report gave the spine 1.856 m/s against rigid's 1.712 m/s at the 2.0 m/s command. That figure rested partly on the rigid variant falling on one of eight seeds. Recomputed on the six seeds where nothing fell, it dropped to **+2.3% and inside the noise**; after the controller defects below were fixed, it became **−0.30%**. The claim is withdrawn. It should never have been reported without the fall-exclusion check.

**7. Heading loop routing through the spine gain — 8 of 8 falls, and four wrong explanations.** After the closed-loop layer was added, the actuated spine fell on every seed at 2.0 m/s while a passive spring of identical stiffness never fell. Those should be the same mechanical system. Four hypotheses were tested and rejected: torque saturation (peak spine torque 20.1 N·m against a 50 N·m limit, 0% of samples saturated, and raising the limit to 400 N·m changed nothing); explicit-versus-implicit integration; explicit PD damping; and gain tuning (it fell at *every* gain from 60 to 400). The tell was that spine deflection stayed at 5.2° regardless of gain — meaning the deflection was **commanded, not compliant**. The cause was in the new controller: the heading loop's correction was routed through `turn_spine_gain` as well as through the legs, so on a *straight* command the "held" spine sat at a continuous 5° steering deflection and over-steered at speed. Fixed by having spine steering follow only the commanded yaw rate (`67c6c81`). With the fix, a truly held spine matches rigid and passive at both speeds.

**8. Lateral drift that was being reported as steering quality.** The rigid variant drifted 2.3 m laterally over 10 m on a straight command, making cross-track error a measure of that defect rather than of steering. Root-caused to a one-time yaw impulse at gait onset: `yaw(t)` is flat until 0.6 s, swings to −13.6° by 1.8 s, then holds, with a late-time rate of only −0.66°/s. Confirmed by mirroring the left/right phase assignment, which flipped the drift exactly (−2.27 m → +2.19 m), and by the `walk` gait — where feet land one at a time — showing +0.37°. A secondary contributor was the tail, which has actuators the controller simply never commanded; holding it cut the late-time drift rate from −0.66 to −0.08°/s. Fixed by adding cross-track position feedback and holding the tail (`d37d185`). Straight-line cross-track fell from 1.46–1.77 m to 0.44–0.52 m.

**9. A sanity check that cried wolf.** The joint-stiffness stability bound was evaluated against the *global minimum* armature, pairing a stiff spine spring with a leg's 0.01 armature and producing a bound 20× too small, and it ignored the integrator. It fired on every healthy passive-variant run. A warning that fires on correct models is worse than no warning, because it teaches you to ignore the ones that matter. `71f607d`.

**10. The flexion asymmetry ran backwards for the entire first phase of the study.** The single most consequential error here, and it was published before it was caught. `asymmetric_wave(ratio=2)` produces a wave peaking at −1.0 and +0.5, so its large excursion is negative. A separate `flexion_sign` multiplier, defaulting to `-1.0`, then negated it — making the *large* excursion positive. Since flexion is negative `spine_pitch` (verified against the model: −0.4 rad puts the tail base at z = 0.297 m, +0.4 rad at z = 0.516 m, against 0.408 m neutral), the shipped default commanded an **extension-dominant** spine at a realised ratio of 0.50 while the parameter name claimed flexion-dominance at 2.0.

The symptom was in the README for a full revision: a realised ratio of 0.43–0.66 against a commanded 2.0, which I wrote up as attenuation. It cannot be. Attenuation lands between 1.0 and the commanded value; below 1.0 means the asymmetry is inverted. The number was diagnostic and I misread it as noise.

The 96-point parameter search did explore both signs and selected `-1.0` — it was not broken, it correctly found that the anti-biological waveform performed better at the phase it was searching, and the default then shipped that under a misleading name. Fixed by deleting `flexion_sign` entirely and letting `flexion_ratio` span both directions (2.0 flexion-dominant, 1.0 symmetric, 0.5 extension-dominant); the flag was redundant anyway, since negating the wave is identical to inverting the ratio and shifting phase by 0.5, and phase was already swept.

Consequences, all re-measured: the straight-line result changed from "no effect" (+0.02% / −0.30%) to a real gain (+0.99% / +1.79%); the turning advantage from +6.07% to +8.12%; the static-bias share of the turning mechanism from 104% to 64%; and cost of transport from 2% worse than rigid to neutral-or-better. The controlled ratio comparison that now anchors the straight-line section — showing the anti-biological direction costs 11.4% — only exists because of this bug.

## Reproducing

Requires Python 3.11+. Dependencies: `mujoco`, `stable-baselines3`, `gymnasium`, `numpy`, `matplotlib`, plus `imageio` and `imageio-ffmpeg` for MP4 output.

```bash
python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/bin/pip install mujoco gymnasium stable-baselines3 numpy matplotlib imageio imageio-ffmpeg
```

Verify the environment, model integrity and numerical stability before trusting anything:

```bash
python harness.py check
```

This prints the GL backend, all three variants' DOF counts, confirms mass-matching and joint removal, and runs a 5 s shakedown at zero and full-amplitude random torque on each variant. It exits non-zero if joint removal failed or anything diverged.

The main comparison — measured at 3.1 minutes headless on a laptop CPU, 5.6 minutes with `--render`:

```bash
python harness.py run --name closed_loop --variants spine,rigid,passive --seeds 0,1,2,3,4,5,6,7 --duration 8.0 --render
```

The spine-hold calibration table (5.0 minutes):

```bash
python tools/calibration_table.py
```

Spine undulation amplitude sweep including the zero-amplitude control (3.1 minutes):

```bash
python harness.py spine-sweep
```

The corrected free-fall reorientation test, which drives each variant with the DOF it actually has:

```bash
python harness.py freefall
```

Rendering is off by default so experiments stay fast. `MUJOCO_GL` is set automatically before `mujoco` is imported — `glfw` on macOS and Windows, `egl` then `osmesa` on Linux — and an explicit environment setting always wins. If a GL context cannot be created the numeric experiments still run to completion and say so loudly; they are never silently skipped.

### Provenance

Every run writes `results/<name>/results.json` alongside `metrics.csv`, recording the git commit and whether the tree was dirty, Python/MuJoCo/NumPy versions, the resolved GL backend, the full config, each variant's compiled DOF counts and mass, whether mass-matching held, and a **per-rollout stability verdict** — divergence reasons, first bad step, peak `|qvel|` and `|qacc|`, and MuJoCo warning counts. A result can therefore be distrusted later on evidence rather than on memory.

## Repository layout

```
cheetah/
  __init__.py     sets MUJOCO_GL before mujoco is imported anywhere
  glbackend.py    GL backend selection and a probe that actually renders a frame
  model.py        the three variants; verified against the compiled model
  control.py      CPG, asymmetric flexion waveform, closed-loop speed and path loops
  rollout.py      the single place physics is stepped and divergence is detected
  metrics.py      locomotion metrics; NaN-on-divergence policy
  stability.py    per-step divergence detection and model sanity checks
  render.py       offscreen rendering, MP4, side-by-side compositing
  plots.py        gait diagrams, spine angle traces, trajectories, speed traces
  experiment.py   config -> CSV/JSON harness
harness.py        CLI: check, run, spine-sweep, freefall
tools/            calibration table and README media generation
spine_quadruped.xml
docs/media/       the curated figure set this README embeds (~4.2 MB)
```

`media/` and `results/` are gitignored; a full render run is about 17 MB and is cheap to regenerate. `docs/media/` is the size-budgeted subset the README needs, rebuilt with `python tools/make_readme_media.py`.

## Citation

If this study is useful, cite it as a negative result on CPG-controlled spine locomotion:

```bibtex
@software{alharbi2026spine,
  author = {Alharbi, Muhannad},
  title  = {Does an active spine help a quadruped run? A controlled MuJoCo study},
  note   = {Independent CPG-based study of active, passive and rigid quadruped
            trunks. Finds a 1-2% straight-line speed gain at 77-85% worse path
            tracking, and attributes most of the turning benefit to articulated
            steering rather than to spine undulation.},
  year   = {2026},
  url    = {https://github.com/cl0udz1/Cheetah-Spine-Experiment}
}
```

The works this study responds to are cited in [Background](#background).

## Licence

[MIT](LICENSE). Copyright (c) 2026 Muhannad Alharbi.

`spine_quadruped.xml` and `run.py` are the original starter files, preserved unmodified in commit `cbe923a` for reference; `run.py` is superseded by `harness.py` and its free-fall test is corrected in `harness.py freefall` (the original drove the spine on the spine variant and drove nothing on the rigid variant, so its 0° result measured the absence of a command rather than the absence of a capability).
