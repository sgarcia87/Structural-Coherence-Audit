# Structural-Coherence-Audit
Reference implementation accompanying the paper on structural coherence loss, formalizing perspective-induced divergence via Jensen-Shannon distance over PPR dynamics.


Minimal, reproducible experiment for detecting **structural coherence loss** in semantic systems under **perspective shift**.

This repository accompanies the paper:

> **Detecting Structural Coherence Loss in Semantic Systems Under Perspective Shift**  
> Sergi Garcia Mecinas (2026)

The goal of this code is not to build a reasoning model or a production framework, but to **demonstrate a specific failure mode**:  
semantic systems can remain locally plausible while **silently losing internal structural coherence** when the perspective from which reasoning is initiated changes.

---

## Motivation

Current evaluation paradigms in AI and semantic systems tend to focus on:
- output correctness,
- representation similarity,
- robustness to input perturbations,
- post-hoc explainability.

However, these approaches often fail to detect **pre-failure conditions**, where a system is still operational but has become structurally fragile.

This repository demonstrates that:
- **structural coherence is a global property**, not reducible to local correctness;
- **coherence loss can manifest as degenerate convergence toward semantic hubs**, not as random fragmentation;
- such loss can be detected **before** visible malfunction occurs.

---

## Core Idea

We treat a semantic system as a graph and define **perspective** as a change in the restart distribution of a structural random walk.

Structural coherence is measured as the **invariance of perspective-induced dynamics**, quantified via divergence between stationary distributions.

A loss of coherence occurs when:
- perspective-induced dynamics diverge,
- equilibrium centers become unstable or collapse toward a single hub,
- the system’s conceptual geometry becomes viewpoint-dependent.

---

## What This Repository Is

✔ A **minimal experimental demonstration**  
✔ A **pre-action structural auditing instrument**  
✔ A **reproducible reference implementation** for the paper  
✔ A concrete illustration of a **leading indicator of fragility**

---

## What This Repository Is NOT

✘ Not a reasoning model  
✘ Not an AI architecture  
✘ Not a training pipeline  
✘ Not a production-ready system  
✘ Not a full implementation of IA_m

This repository isolates **one phenomenon** using **one observational mechanism**.

---

## Method Overview

For a fixed graph \(G\):

1. Define a set of **perspectives** as pole pairs \((A, B)\).
2. For each perspective:
   - compute two Personalized PageRank distributions from \(A\) and \(B\);
   - combine them into a perspective-induced structural field.
3. Measure:
   - divergence between perspective-induced distributions (Jensen–Shannon divergence);
   - equilibrium nodes and their dominance across perspectives;
   - stability and balance properties.

Structural coherence is defined as:

\[
\mathcal{C}(G) = 1 - \mathbb{E}_{\phi_i,\phi_j}[\mathrm{JSD}(\pi_{\phi_i} \,\|\, \pi_{\phi_j})]
\]

---

## Experiment

The demo compares two graphs:

- **G₀ (coherent)**: a conceptually consistent graph.
- **G₁ (contaminated)**: the same graph with a small number of structurally misaligned “bridge” edges.

Key properties:
- same node set,
- nearly identical edge set,
- no factual contradictions introduced.

Despite minimal change, G₁ exhibits:
- lower structural coherence,
- higher equilibrium dominance (hub collapse),
- increased fragility under perspective shift.

---

## Running the Demo

### Requirements

- Python ≥ 3.9
- `networkx`

Install dependencies:

```bash
pip install networkx
````

Run the experiment:

```bash
python3 perspectiva_v2.py
```

The script prints:

* structural coherence (C(G)),
* equilibrium dominance,
* stability rates,
* per-perspective equilibrium diagnostics.

---

## Interpreting the Output

Typical observations:

* (C(G_1) < C(G_0)): divergence under perspective shift.
* Dominance increases in (G_1): multiple perspectives collapse to the same hub.
* Outputs remain plausible: the degradation is **silent**.

This is the phenomenon described in the paper.

---

## Relation to IA_m

This work was developed within a broader investigation into **semantic observability**, referred to as **IA_m**.

IA_m is a meta-framework for auditing structural coherence in semantic and reasoning systems.
It does not train models or generate outputs.

**Semantic diffraction** is one observational instrument within IA_m; this repository isolates it to demonstrate structural coherence loss independently of the full framework.

---

## License

Apache License 2.0

This license allows academic and industrial use while preserving attribution and reproducibility.

---

## Citation

If you use or discuss this work, please cite:

```bibtex
@misc{garciamecinas2026structuralcoherence,
  title={Detecting Structural Coherence Loss in Semantic Systems Under Perspective Shift},
  author={Garcia Mecinas, Sergi},
  year={2026},
  note={Preprint}
}
```

---

## Contact

Sergi Garcia Mecinas
📧 [sos.prevencio@gmail.com](mailto:sos.prevencio@gmail.com)

Feedback, critique, and discussion are welcome—especially regarding structural failure modes, AI safety, and semantic observability.

