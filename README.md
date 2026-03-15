# Concept Assembly Zone

**A dynamical systems framework for cross-layer semantic manifold tracking in transformers**

*James Henry — March 2026*

---

When does a transformer "know" something?

Not at the input — the token embedding is just a lookup. Not at the output — by then the decision is made. Somewhere in the middle, a concept transitions from vague syntactic probability into a geometrically stable, surgically ablatable direction in the residual stream. This paper formalizes where that happens and how to measure it.

The **Concept Assembly Zone (CAZ)** is the contiguous sequence of middle-to-late transformer layers across which a semantic concept assembles: separation grows, coherence peaks, and the direction stabilizes. It is a property of the concept and the model scale, not just the architecture.

---

## The Framework

Three layer-wise metrics characterize the CAZ:

**S(l) — Separation**
Fisher-normalized centroid distance between concept classes at layer l. Measures how discriminable the two classes are in the residual stream.

**C(l) — Coherence**
Explained variance ratio of the primary PCA component across the concept direction. Measures how geometrically concentrated the concept representation is.

**v(l) — Velocity**
Rate of change of separation (dS/dl, smoothed). Used for boundary detection: the CAZ begins where velocity crosses a sustained threshold, peaks where separation is maximum, and ends where velocity turns consistently negative.

Three zones follow:

```
Pre-CAZ     →    Assembly Zone    →    Post-CAZ
                      ↑
                   CAZ Peak
              (maximum separation,
               optimal ablation point)
```

---

## Four Testable Predictions

The framework makes predictions that distinguish it from a descriptive account:

**1. Mid-Stream Ablation Hypothesis**
Ablation at the CAZ peak suppresses concept expression while minimally disrupting general capability. Ablation before the CAZ is ineffective (concept not yet assembled); ablation after the CAZ is destructive (too entangled with downstream computation).

**2. Architecture-Stable CAZ Positioning**
The relative CAZ position (as a fraction of total depth) is concept-specific but consistent across architectures of similar scale. Credibility peaks near 92% depth; negation near 81% — in both GPT-2 (12L) and GPT-2-XL (48L).

**3. CAZ Width and Concept Abstraction**
More abstract concepts (epistemic) have wider CAZ windows than structural concepts (syntactic). Width is a proxy for how distributed the assembly process is.

**4. Post-CAZ Degradation as Logit Interference**
Late-layer re-entanglement of concept directions correlates with structure in the unembedding matrix — the concept direction begins to interact with output vocabulary in ways that reduce surgical ablation quality.

---

## Empirical Validation

These predictions are tested in the companion repository:

**[Rosetta Manifold](https://github.com/jamesrahenry/Rosetta_Manifold)** — full empirical pipeline across 3 concepts, GPT-2 (124M) and GPT-2-XL (1.5B). See [RESULTS.md](https://github.com/jamesrahenry/Rosetta_Manifold/blob/main/RESULTS.md) for complete results and methodology notes.

Results at GPT-2-XL scale (48 layers, 100 contrastive pairs per concept):

| Concept | Type | Peak layer | Relative depth | Peak S |
|---|---|---|---|---|
| Negation | Syntactic | L30 / 48 | 63% | 0.257 |
| Sentiment | Affective | L31 / 48 | 65% | 0.326 |
| Credibility | Epistemic | L46 / 48 | 96% | 0.736 |

**What is confirmed:**
- The concept-type ordering (syntactic < affective < epistemic) emerges clearly at 48-layer scale
- Credibility has dramatically stronger separation signal than negation or sentiment
- The epistemic concept (credibility) has a distinct pre-CAZ region — assembly doesn't begin until layer 21

**What is not yet confirmed:**
- **Prediction 2 (Architecture-Stable Positioning):** Relative depths differ substantially between GPT-2 (all concepts peak at 83%) and GPT-2-XL (63–96%). The 12-layer GPT-2 is too shallow to differentiate concepts. Testing Prediction 2 properly requires multiple architectures at the *same* parameter scale — the frontier-scale work.
- **Prediction 1 (Mid-Stream Ablation):** Confirmed at GPT-2 (100% separation reduction). Not confirmed at GPT-2-XL — the concept direction at 1.5B parameters is too distributed for single-layer projection ablation.
- **Predictions 3 and 4** require frontier-scale models (70B+) for meaningful test.

The concept-ordering finding is the most robust result. The architecture-stability claim requires more work.

---

## Contents

| File | Description |
|---|---|
| [`CAZ_Framework.md`](CAZ_Framework.md) | Full paper — theory, methods, predictions, related work |
| [`CAZ_Framework.tex`](CAZ_Framework.tex) | LaTeX source |
| [`CAZ_Framework_Paper.pdf`](CAZ_Framework_Paper.pdf) | Compiled PDF |

---

## Relationship to Existing Work

The CAZ framework builds on and extends:

- **Arditi et al. (2024)** — *Refusal in Language Models Is Mediated by a Single Direction*: established that single-layer Difference-of-Means extraction suffices for behavioral ablation. CAZ asks whether there is a privileged layer (or window) for this extraction.
- **Zou et al. (2023)** — *Representation Engineering*: introduced Linear Artificial Tomography (LAT) for concept extraction. CAZ provides a principled account of which layers to apply it at.
- **Engels et al. (2024)** — *Not All Language Model Features Are Linear*: identified multi-dimensional concept representations. CAZ is agnostic to linearity but provides the temporal (depth) axis Engels et al. do not address.
- **Wollschläger et al. (2025)** — *Geometry of Concept Embeddings*: characterizes geometric structure of concept directions. CAZ adds the layer-wise dynamics.

The core contribution is framing concept representation not as a static property of a model but as a **process unfolding across depth** — with measurable onset, peak, and degradation.

---

## Status

This is a theoretical framework paper with partial empirical validation. Proxy-scale results for 3 concepts on GPT-2 and GPT-2-XL are complete. The concept-type ordering prediction is confirmed; the architecture-stability prediction (Prediction 2) and ablation hypothesis at larger scale (Prediction 1) require same-scale cross-architecture experiments at frontier scale.

Frontier-scale validation (Llama 3 70B, Qwen 2.5 72B, Mistral Large) is pending compute access. The paper is a preliminary draft and has not been submitted for peer review.

---

## Related

- [**Rosetta Manifold**](https://github.com/jamesrahenry/Rosetta_Manifold) — the empirical pipeline
- [**Pop Goes the Easel**](https://github.com/jamesrahenry/pop_goes_the_easel) — a companion interpretability study using CAZ reference curves

---

*jamesrahenry@henrynet.ca*
