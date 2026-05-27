# Pattern 6 — Plan-Execute (SkillGroup)

A `SkillGroup` chains two skills: a DRL skill plans the temperature setpoint, then an MPC `SkillController` executes precise coolant control to hit it within hard constraints.

**SDK primitives:** `Agent`, `Skill`, `SkillGroup`, `SkillController`, `Teacher`, `Sensor`, `Scenario`, `Trainer`

**Result: ~95% conversion — highest across all six designs.**

---

## Architecture

```
Agent
└── SkillGroup
        ├── Skill("control") ── CSTRTeacher    (DRL: plans temperature setpoint)
        └── Skill("mpc")     ── MPCController  (MPC: executes coolant ΔTc)
```

The `SkillGroup` passes the DRL skill's output as an additional input to the MPC skill at each step. DRL sets the target; MPC tracks it within constraints.

## The code

```python
control_skill = Skill("control", CSTRTeacher)   # DRL planning skill
for scenario_dict in reaction_scenarios:
    control_skill.add_scenario(Scenario(scenario_dict))

mpc_skill = Skill("mpc", MPCController)          # MPC execution skill (no training)

agent = Agent()
agent.add_sensors(sensors)
agent.add_skill(control_skill)

skill_group = SkillGroup(control_skill, mpc_skill)
agent.add_skill_group(skill_group)

trainer.train(agent, train_cycles=10)   # trains DRL skill only
agent.export(PATH_CHECKPOINTS)
```

## Why this outperforms everything else

DRL and MPC have complementary failure modes:

| | DRL | MPC |
|---|---|---|
| Learns from data | ✓ | ✗ |
| Handles nonlinear dynamics | ✓ | limited |
| Enforces hard constraints | ✗ (soft, via reward) | ✓ (structural) |
| Interpretable decisions | ✗ | ✓ |

The plan-execute pattern assigns each technology the job it's suited for. DRL is never asked to satisfy hard constraints. MPC is never asked to learn from data. The result outperforms either alone by a significant margin.

## The MPC execution layer

`MPCController` (see `controller.py`) uses do_mpc + CasADi:

- **State variables:** Ca (concentration), T (temperature)
- **Control variable:** Tc (jacket coolant temperature)
- **Horizon:** 20 steps
- **Hard constraints:** T ≤ 400 K, 273 K ≤ Tc ≤ 322 K, |ΔTc| ≤ 10 K/step
- **Objective:** minimize `(Ca - Cref)²`

The temperature safety bound is structural — the MPC optimizer cannot return a solution that violates it.

## Result

| Metric | Value |
|---|---|
| Product conversion | **~95%** |
| Temperature constraint violations | none |
| vs. MPC alone | +13 pp |
| vs. DRL alone | +5 pp |
| vs. Strategy Pattern | +2 pp |

## Pre-trained checkpoints

The DRL control skill checkpoint is included. Skip training and run inference directly:

```bash
docker run --rm -it -p 1337:1337 composabl/sim-cstr

cd plan_execute_pattern
python agent_inference.py
```
