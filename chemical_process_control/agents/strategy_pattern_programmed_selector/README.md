# Pattern 4 — Strategy Pattern with Programmed Selector

Same three DRL child skills as Pattern 3, but the selector is a deterministic `SkillController` based on episode step count. No selector training needed.

**SDK primitives:** `Agent`, `Skill`, `SkillSelector`, `SkillController`, `Teacher`, `Sensor`, `Scenario`, `Trainer`

---

## Architecture

```
Agent
└── SkillSelector("selector") ── ProgrammedSelector (SkillController, no training)
        ├── Skill("ss1")        ── SS1Teacher
        ├── Skill("transition") ── TransitionTeacher
        └── Skill("ss2")        ── SS2Teacher
```

## The programmed selector

```python
class ProgrammedSelector(SkillController):
    async def compute_action(self, obs, action):
        self.counter += 1
        if self.counter <= 22:   return 0  # → ss1
        elif self.counter >= 76: return 2  # → ss2
        else:                    return 1  # → transition
```

The CSTR scenario has a known schedule: SS1 for steps 0–22, transition for 22–74, SS2 from 76 onward. Encoding this directly is faster and more interpretable than training it.

## The only change from Pattern 3

```python
# Pattern 3:
with SkillSelector("selector", CSTRTeacher) as selector_skill:

# Pattern 4:
with SkillSelector("selector", ProgrammedSelector) as selector_skill:
```

One line. Everything else — the child skills, scenarios, agent wiring — is identical.

## Learned vs. programmed selector: when to use each

| | Learned | Programmed |
|---|---|---|
| Routing logic is known and fixed | — | ✓ |
| Routing depends on observed state | ✓ | — |
| Want to save training compute | — | ✓ |
| Need full routing interpretability | — | ✓ |
| Schedule varies between runs | ✓ | — |

## Result

| Metric | Value |
|---|---|
| Product conversion | ~93% |
| vs. Pattern 3 (learned selector) | identical performance |
| Selector training | none |
| Routing interpretability | full |

Same outcome, less compute, fully auditable routing.
