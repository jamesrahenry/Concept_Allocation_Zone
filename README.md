# Concept Assembly Zone

**Tracking How Concepts Form Across Transformer Depth**

*James Henry — March 2026 (revised April 2026)*

---

When does a transformer "know" something?

Not at the input — the token embedding is just a lookup. Not at the output — by then the decision is made. Somewhere in the middle, the model allocates geometric directions in its residual stream to serve concepts. The **Concept Assembly Zone (CAZ)** is where that happens.

A CAZ is not a concept. It is a layer-level event where the model's geometry expresses influence to serve one or more concepts. Multiple concepts can share a CAZ (48% do), and a single concept typically participates in multiple CAZes across depth.

---

## The Framework

Three layer-wise metrics characterize concept formation:

| Metric | What it measures |
|---|---|
| **S(l) — Separation** | How distinguishable are the two concept classes at this layer? |
| **C(l) — Coherence** | Is the concept encoded as one clean direction, or smeared across many? |
| **v(l) — Velocity** | Is the concept actively forming right now, or has it already formed? |

A scored detection method identifies CAZes across the full spectrum:

| Type | Score | Description |
|---|---|---|
| **Black hole** | > 0.5 | Dominant, concentrated assembly event |
| **Moderate** | 0.05 - 0.5 | Clear assembly with moderate strength |
| **Gentle** | < 0.05 | Subtle but causally active — invisible to standard detection |
| **Embedding** | any (layer 0-1) | Driven by token features, not transformer computation |

---

## Seven Predictions

| # | Prediction | Status |
|---|---|---|
| P1 | Optimal ablation depth relative to CAZ | Revised — depends on encoding strategy |
| P2 | Architecture-stable CAZ ordering | Confirmed across 7 families |
| P3 | CAZ width correlates with abstraction | Initial support |
| P4 | Post-CAZ degradation correlates with unembedding | Not yet tested |
| P5 | Cross-architecture alignment is depth-matched | Strongly confirmed |
| P6 | Shallow peaks are lexical, deep are compositional | Not supported by initial test |
| P7 | Multi-modality is architectural, not scale-dependent | Supported with nuance |

Full empirical results across 30 models and 7 architectural families are reported in the companion validation paper.

---

## Contents

| File | Description |
|---|---|
| [`CAZ_Framework.md`](CAZ_Framework.md) | Full paper — theory, metrics, predictions, related work, proof of concept |
| [`viz_caz_framework_figures.py`](viz_caz_framework_figures.py) | Script to generate all paper figures |
| [`figures/`](figures/) | Generated figures (PNG) |

---

## Reference Implementation

The CAZ extraction pipeline is implemented in [rosetta_tools](https://github.com/jamesrahenry/Rosetta_Tools) (v1.0.0), an open-source Python library providing separation, coherence, velocity, boundary detection, scored detection, Procrustes alignment, directional ablation, and feature tracking.

## Related Repositories

- [**caz_scaling**](https://github.com/jamesrahenry/caz_scaling) — 30-model empirical validation across 7 architectural families
- [**Rosetta Manifold**](https://github.com/jamesrahenry/Rosetta_Manifold) — original 8-model empirical pipeline
- [**Rosetta Program**](https://github.com/jamesrahenry/Rosetta_Program) — parent program coordinating all research

---

*jamesrahenry@henrynet.ca*
