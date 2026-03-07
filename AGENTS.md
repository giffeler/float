# AGENTS.md

## Project purpose

This repository investigates a physical hypothesis about floating bodies on water.

Hypothesis under investigation:

> When two elongated floating bodies on water fall below a certain lateral distance, they tend to move toward each other because fewer wave events exert pressure on their inner sides than on their outer sides. This creates a net inward force and may eventually lead to side contact.

This hypothesis must **not** be assumed true.
All work in this repository must preserve the ability to **confirm, weaken, or falsify** it.

The project is exploratory and research-oriented, not demonstrative propaganda for a preferred conclusion.

---

## Primary goals

1. Build a scientifically useful simulation framework for the hypothesis.
2. Start with a **reduced, interpretable model** before adding more realism.
3. Make all assumptions explicit.
4. Measure observables that can distinguish between:
   - fewer inner-side wave hits
   - lower cumulative inner-side impulse
   - purely geometric shielding
   - model artifacts
   - alternative explanations
5. Keep the code and notebook structure suitable for later extension.

---

## Modeling strategy

### Phase 1: Minimal falsifiable model
Start with a **2D top-view event-based Monte Carlo simulation**.

Use:
- two elongated floating bodies
- simple geometry (rectangles or capsules)
- stochastic wave events distributed in space
- a simplified attenuation law
- a directional contribution model for side pressure / impulse
- explicit logging of inner-side and outer-side hits and impulses

Do **not** start with:
- full CFD
- Navier-Stokes solvers
- high-complexity fluid packages
- visually impressive but conceptually opaque simulations

### Phase 2: Improved wave representation
Only after the minimal model works and is interpretable, a more advanced wave-field model may be added.

### Phase 3: Extensions
Possible later additions:
- non-parallel body orientations
- body rotation
- reflection / partial reflection
- interference effects
- current / drift terms
- capillary effects for small objects
- empirical calibration if data becomes available

---

## Scientific standards

All implementations must follow these rules:

- Do not bake the desired conclusion into the mechanics.
- Prefer transparent and falsifiable logic over realism theater.
- State assumptions explicitly in code comments and notebooks.
- Make parameter choices visible and easy to vary.
- Record enough intermediate metrics to inspect why an effect appears.
- When results are ambiguous, say so.
- When a result depends strongly on a modeling choice, document that dependency.
- Distinguish clearly between:
  - observation
  - model assumption
  - inference
  - speculation

---

## Repository workflow for coding agents

Before making substantial changes:

1. Inspect the repository structure.
2. Summarize the intended change briefly.
3. Identify affected files.
4. Prefer small, coherent commits.

When implementing:
- Keep code modular and typed where practical.
- Prefer simple helper modules over oversized notebooks.
- Keep the notebook readable and explanatory.
- Use deterministic seeds where possible.
- Add lightweight tests for nontrivial geometry or force-calculation logic.

After implementing:
- Summarize what changed.
- Summarize assumptions introduced or modified.
- Note conceptual risks or limitations.
- Suggest the next most sensible step.

---

## Notebook expectations

The main notebook should usually contain:

1. Problem statement
2. Reformulated hypothesis
3. Why this modeling level was chosen
4. Mathematical or algorithmic assumptions
5. Simulation design
6. Observables / metrics
7. Experiments
8. Visualizations
9. Interpretation
10. Alternative explanations / confounders
11. Limitations
12. Next steps

The notebook must be understandable without reading the entire codebase.

---

## Code quality expectations

Use Python.
Prefer:
- `dataclasses`
- type hints
- small pure functions where possible
- NumPy for vectorized work
- Matplotlib for clear plots

Avoid:
- unnecessary frameworks
- hidden state
- magic constants without explanation
- premature optimization
- decorative abstractions with no analytical value

Performance matters only after correctness and interpretability.

---

## Validation expectations

Where appropriate, add checks for:
- symmetry sanity checks
- no-force expectation in balanced configurations
- monotonicity expectations for simple attenuation rules
- reproducibility under fixed random seeds

A simulation that produces attraction everywhere and always is suspicious.
A useful simulation must be able to produce null results or contradictory results.

---

## Communication style in artifacts

Use clear technical English.
Be precise, restrained, and explicit about uncertainty.
Do not overclaim physical truth from a toy model.
Do not describe results as “proof” unless they actually justify that level of certainty.

---

## Decision rule for next steps

If the minimal model does not show a robust effect, do not blindly add complexity just to recover the desired outcome.
Instead:
- inspect assumptions
- test alternative source distributions
- test whether the hypothesis formulation itself needs revision

If the minimal model does show an effect, the next step is not celebration but robustness analysis.

---

## Immediate priority

The current priority is to create the first well-structured Jupyter notebook implementing the minimal event-based 2D Monte Carlo model.
