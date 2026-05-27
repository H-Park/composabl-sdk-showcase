# Pattern 3 — Strategy Pattern with Learned Selector

Decompose the problem into three specialized DRL skills, one per operating regime. A fourth `SkillSelector` skill learns when to hand off between them.

**SDK primitives:** `Agent`, `Skill`, `SkillSelector`, `Teacher`, `Sensor`, `Scenario`, `Trainer`

---

## Architecture

```
Agent
└── SkillSelector("selector") ── CSTRTeacher      # learned: outputs 0, 1, or 2
        ├── Skill("ss1")        ── SS1Teacher      # specializes on Ca = 8.57 K
        ├── Skill("transition") ── TransitionTeacher # specializes on the ramp
        └── Skill("ss2")        ── SS2Teacher      # specializes on Ca = 2.0 K
```

## The code

```python
with SkillSelector("selector", CSTRTeacher) as selector_skill:
    selector_skill.add_scenario(Scenario({"Cref_signal": "complete"}))

    with Skill("ss1", SS1Teacher) as ss1_skill:
        ss1_skill.add_scenario(Scenario({"Cref_signal": "ss1"}))
        selector_skill.add_skill(ss1_skill)

    with Skill("transition", TransitionTeacher) as transition_skill:
        transition_skill.add_scenario(Scenario({"Cref_signal": "transition"}))
        selector_skill.add_skill(transition_skill)

    with Skill("ss2", SS2Teacher) as ss2_skill:
        ss2_skill.add_scenario(Scenario({"Cref_signal": "ss2"}))
        selector_skill.add_skill(ss2_skill)

agent.add_selector_skill(
    selector_skill,
    [ss1_skill, transition_skill, ss2_skill],
    fixed_order=False,
    fixed_order_repeat=False
)
trainer.train(agent, train_cycles=100)
```

The `with Skill(...) as skill:` context manager registers each child skill into the selector's scope. The `SkillSelector` is itself a DRL policy that outputs a discrete routing action at each step.

## Scenario decomposition

Each skill trains on a narrow slice of the problem:

| Skill | `Cref_signal` | Sim behavior |
|---|---|---|
| `ss1` | `"ss1"` | Holds steady state 1 (Ca = 8.57, T = 311 K) |
| `transition` | `"transition"` | Ramps from SS1 to SS2 over steps 22–74 |
| `ss2` | `"ss2"` | Holds steady state 2 (Ca = 2.0, T = 373 K) |
| `selector` | `"complete"` | Full 90-step trajectory — selector sees the real sequence |

Skills specialize in one regime. The selector sees the full trajectory and learns when each skill should be active.

## Why this beats single-skill DRL

A single DRL policy tries to be three different controllers from one checkpoint. The strategy pattern gives each regime its own policy. The 3-point improvement comes entirely from that specialization — the DRL algorithms are identical.

## Result

| Metric | Value |
|---|---|
| Product conversion | ~93% |
| vs. single DRL | +3 pp |
| vs. MPC baseline | +11 pp |
| Skills trained | 4 (3 child + selector) |
