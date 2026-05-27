# Pattern 2 — Single-Skill MPC Benchmark

Replace DRL entirely with a `SkillController` that runs a do_mpc + CasADi optimization at every timestep. No training required.

**SDK primitives:** `Agent`, `Skill`, `SkillController`, `Sensor`, `Scenario`

---

## Architecture

```
Agent
└── Skill("reaction") ── MPCController (SkillController, no training)
        └── Scenario(Cref_signal="complete")
```

## The code

```python
reaction_skill = Skill("reaction", MPCController)  # MPCController extends SkillController
reaction_skill.add_scenario(Scenario({"Cref_signal": "complete"}))

agent = Agent()
agent.add_sensors(sensors)
agent.add_skill(reaction_skill)
# trainer.train() runs one pass to register the skill — MPC itself needs no training
```

## How the controller works

`MPCController.compute_action()` is called at each step. It builds and solves a constrained optimization problem using do_mpc + CasADi:

```python
class MPCController(SkillController):
    async def compute_action(self, obs, action):
        # CSTR ODE model
        model.set_rhs("Ca", F/V*(Cafin - Ca) - k0*exp(-E/(R*T))*Ca)
        model.set_rhs("T",  F/V*(Tf - T) - ΔH/phoCp*k0*exp(-E/(R*T))*Ca - UA/(phoCp*V)*(T - Tc))

        # Objective: minimize concentration tracking error
        mpc.set_objective(mterm=(Ca - Cref)**2, lterm=(Ca - Cref)**2)

        # Hard constraints — these cannot be violated by construction
        mpc.bounds["upper", "_x", "T"]  = 400   # thermal runaway limit
        mpc.bounds["lower", "_u", "Tc"] = 273   # physical coolant lower bound
        mpc.bounds["upper", "_u", "Tc"] = 322

        u0 = mpc.make_step(x0)
        return [float(u0[0][0]) - obs["Tc"]]  # return ΔTc
```

Horizon: 20 steps. Input penalty: 1.5 (limits actuator aggression).

## Why MPC as a baseline

MPC is the industry-standard advanced control method for constrained process systems. It's the right benchmark because it represents what a process engineer would deploy without machine learning — fully interpretable, provably constraint-satisfying, no training data required.

The SDK treats it as a `SkillController`, so the agent composition API is identical to a DRL-backed skill. You can swap MPC for DRL (or vice versa) by changing one line.

## Where it falls short

A linear model of a nonlinear system carries model-mismatch error. The CSTR reaction rate is exponential in temperature (Arrhenius kinetics) — a linear approximation degrades during the transition regime. This caps performance at ~82%.

## Result

| Metric | Value |
|---|---|
| Product conversion | ~82% |
| Temperature constraint violations | none (structural) |
| Training required | none |
| Interpretability | full |

Every other pattern in this repo beats this number.
