# Pattern 5 — Strategy Pattern with ML Perceptor

Strategy Pattern (Pattern 3) plus a scikit-learn classifier running on every observation step. It injects a `thermal_runaway_predict` signal into the observation space before any skill sees it.

**SDK primitives:** `Agent`, `Skill`, `SkillSelector`, `Perceptor`, `PerceptorImpl`, `Teacher`, `Sensor`, `Scenario`, `Trainer`

---

## Architecture

```
Agent
├── Perceptor("thermal_runaway_predict") ── ThermalRunawayPredict (PerceptorImpl)
│       └── sklearn classifier  ← ml_models/ml_predict_temperature.pkl
│
└── SkillSelector("selector") ── CSTRTeacher
        ├── Skill("ss1")        ── SS1Teacher
        ├── Skill("transition") ── TransitionTeacher
        └── Skill("ss2")        ── SS2Teacher
```

The perceptor runs before any skill on every step. Its output is appended to the observation dict that the selector and all child skills receive.

## What a Perceptor does

Raw sensors report what the process is doing right now. A `Perceptor` transforms, enriches, or interprets that raw signal before it reaches skills. The skills get a richer observation and can condition their behavior on derived features — without needing to know how those features were computed.

`PerceptorImpl.compute()` accepts any Python callable: a sklearn model, a neural network, a physics model, an LLM call. The interface is the same regardless.

## The perceptor implementation

```python
class ThermalRunawayPredict(PerceptorImpl):
    def __init__(self):
        self.ml_model = pickle.load(open("ml_models/ml_predict_temperature.pkl", "rb"))

    async def compute(self, obs_spec, obs):
        if float(obs["T"]) >= 340:
            X = [[obs["Ca"], obs["T"], obs["Tc"], self.ΔTc]]
            # 0.3 threshold: prefer false positives over missed runaway events
            if self.ml_model.predict_proba(X)[0][1] >= 0.3:
                return {"thermal_runaway_predict": 1}
        return {"thermal_runaway_predict": 0}

ml_predict = Perceptor("thermal_runaway_predict", ThermalRunawayPredict, "Predict thermal runaway")
```

The 0.3 probability threshold deliberately biases toward safety — a missed runaway is worse than an unnecessary precautionary action.

## Wiring it to the agent

```python
agent = Agent()
agent.add_sensors(sensors)
agent.add_perceptors([ml_predict])   # one line adds the full ML inference pipeline
# ... same SkillSelector + Skill setup as Pattern 3 ...
```

## The ML model

- **Type:** scikit-learn 1.0.2 classifier
- **Inputs:** `[Ca, T, Tc, ΔTc]`
- **Output:** binary — 0 (normal) or 1 (thermal runaway imminent)
- **Update path:** swap the `.pkl` file; no DRL retraining needed

## Result

| Metric | Value |
|---|---|
| Product conversion | ~93% |
| Thermal runaway incidents | significantly reduced vs. Pattern 3 |
| Perceptor update path | replace pkl, no skill retraining |

The conversion rate matches Pattern 3 — the perceptor's contribution is safety, not yield. The perception model and the control skills are independently maintainable.
