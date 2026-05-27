# Pattern 1 — Single-Skill Deep Reinforcement Learning

One `Skill` backed by one `Teacher`. Train a single DRL policy to handle the full reaction trajectory end-to-end.

**SDK primitives:** `Agent`, `Skill`, `Teacher`, `Sensor`, `Scenario`, `Trainer`

---

## Architecture

```
Agent
└── Skill("reaction") ── CSTRTeacher
        └── Scenario(Cref_signal="complete")   # full 90-step trajectory
```

## The code

```python
reaction_skill = Skill("reaction", CSTRTeacher)
reaction_skill.add_scenario(Scenario({"Cref_signal": "complete"}))

agent = Agent()
agent.add_sensors(sensors)
agent.add_skill(reaction_skill)

trainer.train(agent, train_cycles=10)
agent.export(PATH_CHECKPOINTS)
```

## How the Teacher works

`CSTRTeacher` defines three things:

**Reward** — inversely proportional to concentration tracking error:
```python
error = (obs['Cref'] - obs['Ca']) ** 2
reward = 1 / math.sqrt(error + 1e-10)
```
Near-perfect tracking → large reward. Large deviation → reward approaches zero. This gives the policy a strong gradient signal close to the setpoint.

**Sensor filter** — the policy sees `['T', 'Tc', 'Ca', 'Cref', 'Tref']` (all five process signals).

**Termination** — never terminates early. The full 90-step episode always runs.

## When to use this pattern

- Good first attempt before decomposing into multiple skills.
- The task is continuous and a single policy can plausibly generalize across all regimes.
- You want a quick training baseline to compare against.

## Where it falls short

The policy has to simultaneously learn SS1 regulation, nonlinear transition management, and SS2 regulation. It compromises across regimes rather than specializing in any. This shows up as oscillation during the transition phase and a ceiling around 90% conversion.

## Result

| Metric | Value |
|---|---|
| Product conversion | ~90% |
| Training | one skill, one scenario |
| Compared to MPC baseline | +8 pp |

The 5-point gap to plan-execute (95%) motivates the patterns that follow.
