# Structural Coherence Loss — Minimal Example (`example.py`)

This repository provides a **minimal, fully reproducible reference implementation** accompanying the paper:

> **Detecting Structural Coherence Loss in Semantic Systems Under Perspective Shift**  
> Sergi Garcia Mecinas (2026)

The purpose of this code is **not** to build a reasoning system or a semantic model, but to demonstrate a **specific structural failure mode**:

> A semantic system may remain locally plausible while **silently losing internal structural coherence** when the *perspective* from which reasoning is initiated changes.

---

## What This Example Demonstrates

This example shows that:

- **structural coherence is a global property** of a semantic graph;
- coherence loss does not necessarily appear as random fragmentation;
- instead, it often manifests as **degenerate convergence toward semantic hubs**;
- such degradation can be detected **before observable malfunction** occurs.

The script `example.py` is intentionally minimal to ensure:
- clarity,
- reproducibility,
- and unambiguous interpretation of results.

---

## Core Idea

A semantic system is represented as a **directed graph** \( G \).

A **perspective** is defined as a pair of conceptual poles \( (A, B) \).  
Each perspective induces a change in the restart distribution of a structural random walk.

For each perspective:

1. Two Personalized PageRank (PPR) distributions are computed:
   \[
   \pi_A,\ \pi_B
   \]
2. They are combined into a perspective-induced structural field:
   \[
   \pi_\phi = \tfrac{1}{2}(\pi_A + \pi_B)
   \]

Structural coherence is measured as the **invariance of these fields across perspectives**, quantified using Jensen–Shannon divergence.

---

## Structural Coherence Metric

Structural coherence is defined as:

\[
\mathcal{C}(G) =
1 - \mathbb{E}_{\phi_i,\phi_j}
\left[
\mathrm{JSD}(\pi_{\phi_i} \,\|\, \pi_{\phi_j})
\right]
\]

- High \( \mathcal{C}(G) \): dynamics are stable across perspectives.
- Low \( \mathcal{C}(G) \): dynamics depend strongly on viewpoint.

---

## What This Repository Is

✔ A **minimal experimental demonstration**  
✔ A **paper companion example**  
✔ A **structural auditing instrument**  
✔ A **leading indicator of semantic fragility**

---

## What This Repository Is NOT

✘ Not a reasoning model  
✘ Not an AI architecture  
✘ Not a training pipeline  
✘ Not a production system  
✘ Not a full implementation of IA_m  

This example isolates **one phenomenon** using **one observational mechanism**.

---

## The Experiment

The default experiment compares two graphs:

### G₀ — Coherent Baseline
A conceptually consistent graph with aligned structural relations.

### G₁ — Contaminated Variant
The same graph with a **small number of structurally misaligned edges** (“bridge” edges).

Properties:

- identical node sets,
- nearly identical edge sets,
- no factual contradictions introduced.

Despite minimal modification, G₁ exhibits:

- lower structural coherence,
- increased equilibrium dominance (hub collapse),
- higher fragility under perspective shift.

---

## Running the Example

### Requirements

- Python ≥ 3.9
- `networkx`
- `matplotlib` (only if figures are generated)

Install dependencies:

```bash
pip install networkx matplotlib
```

---

### Reproduce the Paper Result (Default)

```bash
python3 example.py
```

This single command:

- builds G₀ and G₁ internally,
- evaluates the same set of perspectives on both graphs,
- prints coherence, dominance, fragmentation, and stability metrics,
- reports deltas (G₁ − G₀).

This is the **reference experiment described in the paper**.

---

### Generate Figure 1 (Optional)

```bash
python3 example.py --figure1
```

This generates two images:

- `figure1_structural_coherence_G0.png`
- `figure1_structural_coherence_G1.png`

Each figure contains three panels:

- **(A)** Base graph structure  
- **(B)** Contaminating edges (highlighted in red)  
- **(C)** Perspective-induced equilibrium convergence  

---

## Output Metrics

The script reports:

- **Structural coherence** \( \mathcal{C}(G) \)
- Mean Jensen–Shannon divergence
- **Dominance** (degree of equilibrium hub collapse)
- **Fragmentation** (diversity of equilibrium nodes)
- **Stability rate**
- **Balance** of interference per perspective

These metrics are **structural indicators**, not accuracy or performance scores.

---

## Optional: Running on a Custom Graph

Although designed for the paper experiment, the script can also be applied to a custom graph:

```bash
python3 example.py --json your_graph.json --auto_perspectives 6
```

Expected JSON format (node-link):

```json
{
  "nodes": [{"id": "A"}, {"id": "B"}],
  "links": [{"source": "A", "target": "B", "weight": 1.0}]
}
```

This mode is provided for exploratory use and is **not required** to reproduce the paper.

---

## Relation to IA_m

This example emerged from a broader investigation into **semantic observability**, referred to as **IA_m**.

IA_m is a meta-framework for auditing structural coherence in semantic and reasoning systems.
It does not train models or generate outputs.

The present example isolates **semantic diffraction under perspective shift** to demonstrate coherence loss independently of the full framework.

---

## License

Apache License 2.0

---

## Citation

```bibtex
@article{garciamecinas2026structuralcoherence,
  title  = {Detecting Structural Coherence Loss in Semantic Systems Under Perspective Shift},
  author = {Garcia Mecinas, Sergi},
  year   = {2026},
  note   = {Preprint}
}
```

---

## Contact

**Sergi Garcia Mecinas**  
📧 sos.prevencio@gmail.com

Feedback and critique are welcome, particularly regarding  
**structural failure modes**, **AI evaluation**, and **semantic observability**.
