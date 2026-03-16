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

**[Rosetta Manifold](https://github.com/jamesrahenry/Rosetta_Manifold)** — full empirical pipeline across 8 concepts × 8 model architectures (GPT-2 family, GPT-Neo, Pythia, OPT). See [RESULTS.md](https://github.com/jamesrahenry/Rosetta_Manifold/blob/main/RESULTS.md) for complete results and methodology notes.

Results at GPT-2-XL scale (48 layers, 100 contrastive pairs per concept):

| Concept | Type | Peak layer | Relative depth | Peak S |
|---|---|---|---|---|
| temporal_order | relational | L36 / 48 | 75% | 0.449 |
| causation | relational | L37 / 48 | 77% | 0.488 |
| negation | syntactic | L39 / 48 | 81% | 0.314 |
| certainty | epistemic | L44 / 48 | 92% | 0.500 |
| moral_valence | affective | L44 / 48 | 92% | 0.294 |
| sentiment | affective | L44 / 48 | 92% | 0.396 |
| credibility | epistemic | L46 / 48 | 96% | 0.736 |
| plurality | syntactic | L47 / 48 | 98% | 0.322 |

**What is confirmed:**
- A broad late-assembly pattern: relational and syntactic concepts generally precede affective and epistemic concepts at 48-layer scale
- Credibility has substantially stronger separation signal (S=0.736) than all other concepts
- Affective and epistemic concepts cluster tightly (L44–L46, 92–96%)
- Prediction 1 (Mid-Stream Ablation) confirmed at GPT-2 scale (100% separation reduction)

**What is not confirmed / anomalous:**
- **The specific syntactic < relational ordering is reversed** — relational concepts (causation, temporal_order) assemble earlier than negation, contradicting the predicted ordering within those two types
- **Plurality is anomalously deep** (L47, 98%) — the deepest concept measured, deeper than credibility; not explained by the current framework
- **Prediction 2 (Architecture-Stable Positioning):** The broad ordering holds across scales but absolute depths vary. Proper test requires same-scale cross-architecture comparison (e.g., GPT-Neo-1.3B vs OPT-1.3B vs GPT-2-XL)
- **Prediction 1 at GPT-2-XL:** Not confirmed — single-layer projection ablation insufficient at 1.5B scale
- **Predictions 3 and 4** require frontier-scale models (70B+)

The clearest result is the separation of the relational/syntactic cluster (75–81%) from the affective/epistemic cluster (92–96%). The within-type ordering is more complex than the framework predicted.

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
