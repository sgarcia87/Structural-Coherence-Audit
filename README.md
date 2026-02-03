# Structural-Coherence-Audit

Reference implementation accompanying the paper on **structural coherence loss**, formalizing **perspective-induced divergence** via Jensen–Shannon distance over **Personalized PageRank (PPR)** dynamics.

A **minimal, reproducible experiment** for detecting **structural coherence loss** in semantic systems under **perspective shift**.

This repository accompanies the paper:

> **Detecting Structural Coherence Loss in Semantic Systems Under Perspective Shift**  
> Sergi Garcia Mecinas (2026)

---

## Abstract

Semantic and reasoning systems can remain locally plausible while silently losing their internal structural integrity.  
This repository presents a minimal experimental framework that exposes such **pre‑failure structural degradation** by observing how global graph dynamics change under shifts in perspective.

Rather than evaluating correctness or similarity, the method audits **structural coherence**: the invariance of meaning‑preserving dynamics when reasoning is initiated from different conceptual poles.

---

## Motivation

Current evaluation paradigms in AI and semantic systems tend to focus on:

- output correctness,
- representation similarity,
- robustness to input perturbations,
- post‑hoc explainability.

However, these approaches often fail to detect **pre‑failure conditions**, where a system is still operational but has become **structurally fragile**.

This repository demonstrates that:

- **structural coherence is a global property**, not reducible to local correctness;
- coherence loss can manifest as **degenerate convergence toward semantic hubs**, not as random fragmentation;
- such loss can be detected **before visible malfunction occurs**.

---

## Core Idea

We treat a semantic system as a directed graph \(G\) and define **perspective** as a change in the restart distribution of a structural random walk.

Structural coherence is measured as the **invariance of perspective‑induced dynamics**, quantified via divergence between stationary distributions.

A loss of coherence occurs when:

- perspective‑induced dynamics diverge,
- equilibrium centers become unstable or collapse toward a single hub,
- the system’s conceptual geometry becomes viewpoint‑dependent.

---

## Formal Definition

For a perspective \(\phi = (A, B)\):

1. Compute two Personalized PageRank distributions:
   \[
   \pi_A, \; \pi_B
   \]

2. Combine them into a perspective‑induced structural field:
   \[
   \pi_\phi = \tfrac{1}{2}(\pi_A + \pi_B)
   \]

Structural coherence is defined as:

\[
\mathcal{C}(G) = 1 -
\mathbb{E}_{\phi_i,\phi_j}
\left[
\mathrm{JSD}(\pi_{\phi_i} \,\|\, \pi_{\phi_j})
\right]
\]

Low values of \(\mathcal{C}(G)\) indicate high sensitivity to perspective.

---

## What This Repository Is

✔ A **minimal experimental demonstration**  
✔ A **pre‑action structural auditing instrument**  
✔ A **reproducible reference implementation** for the paper  
✔ A concrete illustration of a **leading indicator of fragility**

---

## What This Repository Is NOT

✘ Not a reasoning model  
✘ Not an AI architecture  
✘ Not a training pipeline  
✘ Not a production‑ready system  
✘ Not a full implementation of **IA_m**

This repository isolates **one phenomenon** using **one observational mechanism**.

---

## Experiment Design

The reference experiment compares two graphs:

- **G₀ (coherent)**: a conceptually consistent structure.
- **G₁ (contaminated)**: the same graph with a small number of structurally misaligned “bridge” edges.

Key properties:

- identical node sets,
- nearly identical edge sets,
- no factual contradictions introduced.

Despite minimal change, **G₁** exhibits:

- lower structural coherence,
- higher equilibrium dominance (hub collapse),
- increased fragility under perspective shift.

---

## Implementation Overview

Main script:

```
perspectiva2_integrated_v3.py
```

Capabilities include:

- virtual reference graphs (G₀ / G₁),
- loading real graphs from JSON (node‑link format),
- automatic or manual perspective selection,
- structural contamination injection (controlled drift),
- Figure‑based visualization of coherence loss.

---

## Running the Demo

### Requirements

- Python ≥ 3.9
- `networkx`
- `matplotlib` (optional, only for figures)

Install dependencies:

```
pip install networkx matplotlib
```

---

### Reference Experiment (Paper Reproduction)

```
python3 perspectiva2_integrated_v3.py --compare_virtual
```

---

### Running on a Real Graph

```
python3 perspectiva2_integrated_v3.py \
  --json red_fractal_demo.json \
  --auto_perspectives 6
```

---

### Manual, Semantically Grounded Perspectives (Recommended)

```
python3 perspectiva2_integrated_v3.py \
  --persp "calor,frío" \
  --persp "presión,volumen" \
  --persp "espacio,tiempo"
```

---

## Automatic Perspective Selection Policies

```
--auto_policy free | balanced | strict
```

- **free**  
  No filtering; exploratory, degree‑based selection.

- **balanced** *(default)*  
  Requires a minimum number of **commonly reachable nodes** between poles.

- **strict**  
  Same as balanced, plus:
  - excludes equilibrium / synthesis / emergent nodes as poles.

Additional controls:

```
--exclude_poles "tibio,presión,temperatura"
--min_common_reachable 3
```

---

## Structural Contamination (Drift Injection)

```
python3 perspectiva2_integrated_v3.py \
  --inject_contamination 2 \
  --figure1
```

Injected edges are:

- temporary (not saved),
- explicitly marked as contaminating,
- highlighted in visualization outputs.

This enables **before/after coherence comparison** without modifying the source graph.

---

## Output Metrics

The script reports:

- **Structural coherence** \(\mathcal{C}(G)\)
- Mean Jensen–Shannon divergence
- Equilibrium dominance (hub collapse indicator)
- Fragmentation
- Stability rate
- Balance metrics per perspective

These metrics function as **early‑warning structural indicators**, not accuracy scores.

---

## Relation to IA_m

This work was developed within a broader investigation into **semantic observability**, referred to as **IA_m**.

IA_m is a meta‑framework for auditing structural coherence in semantic and reasoning systems.
It does not train models or generate outputs.

**Semantic diffraction** is one observational instrument within IA_m; this repository isolates it to demonstrate structural coherence loss independently of the full framework.

---

## License

Apache License 2.0

This license allows academic and industrial use while preserving attribution and reproducibility.

---

## Citation

```
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

Feedback, critique, and discussion are welcome—especially regarding  
**structural failure modes**, **AI safety**, and **semantic observability**.
