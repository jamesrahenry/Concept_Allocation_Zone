# The Concept Assembly Zone

**A Dynamical Systems Framework for Cross-Layer Semantic Manifold Tracking in Transformers**

**James Henry**
*Independent Researcher*
jamesrahenry@henrynet.ca

March 2026

---

## Abstract

Mechanistic interpretability methods commonly extract concept representations by identifying the single optimal layer of a Transformer's residual stream where class separation peaks. This "best layer" heuristic is computationally efficient and empirically grounded, but it captures a snapshot of a process rather than the process itself. We introduce the **Concept Assembly Zone** (CAZ): the contiguous sequence of middle-to-late layers across which a semantic concept transitions from vague syntactic probability to a rigid, geometrically extractable direction. We formalize the CAZ through three layer-wise metrics—Separation, Concept Coherence, and Concept Velocity—and derive a principled method for identifying CAZ boundaries without manual layer sweeps. The framework generates four testable predictions, including the Mid-Stream Ablation Hypothesis: that orthogonal projection ablation applied *within* the CAZ produces superior behavioral suppression with lower collateral capability damage than post-CAZ ablation, offering a candidate explanation for the elevated KL divergences observed in abliteration work. We situate the CAZ within the emerging geometric interpretability program as a complement to existing feature-extraction and manifold-level methods.

---

## 1. Introduction

The dominant paradigm in mechanistic interpretability extracts concept representations by identifying the single "best layer"—the residual stream depth at which a linear probe or difference-of-means (DoM) vector achieves maximum class separation [Zou et al., 2023; Arditi et al., 2024]. This heuristic is computationally convenient and empirically grounded. It is also, by design, a snapshot: it identifies the peak of a process rather than characterizing the process itself.

Transformers are iterative dynamical systems. Each layer applies a sequence of attention and MLP operations that *write* new information into the residual stream, modifying and extending what prior layers contributed [Elhage et al., 2021]. A concept observed at Layer 15 was shaped by Layers 10 through 14 before it; the best-layer heuristic tells us where the concept is most legible, not how it arrived there.

This paper introduces the **Concept Assembly Zone** (CAZ) framework, which extends the interpretability toolkit from anatomy—*where is the concept most visible?*—to dynamical flow—*how does the concept form?* The CAZ is defined as the specific sequence of layers where a semantic concept transitions from vague syntactic entanglement to a rigid, mathematically extractable geometric direction.

The framework has immediate practical implications. If concepts are assembled across a window of layers rather than peaking at a point, then:

1. CAZ-windowed extraction methods may capture information present in the assembly dynamics that single-layer methods do not;
2. Ablation interventions applied during concept assembly may produce qualitatively different effects than interventions applied after assembly is complete;
3. The "dark matter" of unexplained model behavior [Engels et al., 2024] may partially correspond to in-progress concept construction that hasn't yet resolved into the linear directions that sparse autoencoders (SAEs) are designed to capture.

We make no empirical claims in this paper. The CAZ is a theoretical framework generating specific, falsifiable predictions. We describe the minimal experiments required to validate or refute it, and we are explicit about the assumptions the framework inherits from the broader interpretability literature.

---

## 2. Background

### 2.1 The Residual Stream and Concept Representation

The residual stream formulation [Elhage et al., 2021] treats each layer's output as an additive contribution to a shared communication channel. Attention heads and MLPs read from and write to this stream; the final residual vector is projected onto the unembedding matrix to produce logits. This architecture makes layer-by-layer tracking of concept geometry natural: we can ask, at each layer *l*, how well the current residual stream separates two contrastive classes.

### 2.2 Difference-of-Means and Linear Artificial Tomography

DoM extracts a concept direction V_concept ∈ ℝ^d as the normalized difference between class-conditional mean activations at the chosen layer [Zou et al., 2023]. Linear Artificial Tomography (LAT) uses a similar contrastive approach. Both methods produce a single vector at a single depth—a precise and useful representation of where the concept is most geometrically legible. The CAZ framework asks what additional information about concept formation might be recoverable from the layers surrounding that peak.

### 2.3 Abliteration and Intervention Depth

Arditi et al. [2024] demonstrated that refusal behavior across 13 open-source models is mediated by a single direction removable via weight orthogonalization ("abliteration"). Independent replications have observed KL divergences between abliterated and unmodified models ranging from 3.16 to 5.71—suggesting that while behavioral suppression is effective, the intervention also affects general model capabilities. The CAZ framework offers a candidate explanation: if abliteration applies orthogonal projection at the best layer, which typically lies at or after the CAZ peak, the intervention may be acting on a representation that downstream computation has already incorporated. Intervening earlier in the assembly process may allow later layers to adapt, potentially preserving general capabilities while maintaining behavioral suppression.

### 2.4 The Emerging Geometric Program

Gurnee et al. [2025] demonstrated that character counts are represented on low-dimensional curved helical manifolds in the residual stream, with attention heads performing geometric transformations on these structures. Engels et al. [2025] found circular multi-dimensional representations for temporal concepts (days, months, years) that are not decomposable into independent one-dimensional SAE features. Wollschläger et al. [2025] showed that refusal occupies multi-dimensional polyhedral concept cones with multiple independent directions. These findings establish a growing body of evidence for rich geometric structure in activation space, and have explicitly called for unsupervised methods to detect and characterize it. The CAZ framework is designed to complement this geometric program by providing a layer-indexed account of when such structures crystallize.

---

## 3. The Concept Assembly Zone

### 3.1 Concept Lifecycle

By tracking the residual stream across model depth, three phases of concept formation are empirically distinguishable:

**Phase 1: Pre-CAZ (Context and Syntax)**

In early layers, the residual stream primarily resolves local context, grammar, and surface token relationships. Projecting contrastive datasets into this space produces heavily entangled activations; the separation metric is near zero. The model has not yet committed to a semantic trajectory.

**Phase 2: The Assembly Zone**

In middle layers, attention heads and MLPs iteratively write semantic features into the residual stream. Contrastive activations diverge into distinct geometrically coherent manifolds. A stable concept direction crystallizes, and the variance explained by the primary component spikes. This is the CAZ.

**Phase 3: Post-CAZ (Logit Projection)**

In late layers, the model transitions from abstract representation to concrete next-token prediction. The residual stream is projected toward the unembedding matrix. The clean semantic manifold often degrades or re-entangles as abstract concept geometry gives way to vocabulary-specific structure.

### 3.2 Layer-Wise Metrics

Let h_l^(i) ∈ ℝ^d be the residual stream activation at layer *l* for sample *i*, and let 𝒜, ℬ be contrastive classes with conditional means h̄_𝒜^(l), h̄_ℬ^(l) and within-class covariance matrices Σ_𝒜^(l), Σ_ℬ^(l).

**Separation Metric**

We define the separation at layer *l* using a Fisher-normalized criterion:

```
S(l) = ||h̄_𝒜^(l) - h̄_ℬ^(l)||₂ / √[(1/2)(tr(Σ_𝒜^(l)) + tr(Σ_ℬ^(l)))]
```

Raw centroid distance is misleading when cluster dispersion varies across layers. Early layers tend toward diffuse, high-variance representations; normalization by within-class spread corrects for this. Mahalanobis distance would account for full covariance structure but is numerically unstable without regularization in high-dimensional activation spaces. Fisher normalization provides the appropriate tradeoff between geometric fidelity and computational feasibility for initial experiments.

**Concept Coherence**

Separation alone is insufficient: two classes could exhibit identical centroid separation while one forms a tight cluster and the other a diffuse cloud. We track Concept Coherence as the explained variance ratio of the first principal component of the between-class direction at each layer:

```
C(l) = λ₁^(l) / Σᵢ λᵢ^(l)
```

where λᵢ^(l) are the eigenvalues of the pooled activation covariance at layer *l*, projected onto the contrastive subspace. A concept is *well-formed* when both S(l) and C(l) are high: the classes are far apart *and* the separating direction is geometrically clean.

**Concept Velocity**

To identify CAZ boundaries, we compute the rate of geometric divergence between layers. Because raw layer-to-layer differences are noisy, we apply a smoothed estimate:

```
v_concept(l) = (1/(2k+1)) Σⱼ₌ₗ₋ₖ^(l+k) [S(j) - S(j-1)]
```

where *k* is the smoothing half-window. A practical heuristic is k = ⌊L/24⌋, where L is total model depth, yielding k=1 for 12–24 layer models, k=2 for 48-layer models, and k=3 for 72-layer models. This scales the smoothing window proportionally to model depth and prevents false CAZ boundary detection from single anomalous layers. The appropriate value of k should ultimately be determined empirically — for models where ground-truth concept boundaries can be established via ablation, the k value that maximizes boundary prediction accuracy is preferred.

### 3.3 CAZ Boundary Detection

The CAZ boundaries are derived from the velocity profile:

- **CAZ Entry** (l_start): The first layer where v_concept(l) exceeds a sustained positive threshold θ₊.
- **CAZ Peak** (l_max): The layer where S(l) reaches its absolute maximum. This corresponds to the "best layer" of conventional interpretability.
- **CAZ Exit** (l_end): The layer where v_concept(l) becomes consistently negative, marking the onset of post-CAZ degradation.

The conventional best-layer heuristic extracts V_concept at l_max. CAZ-aware extraction uses the full interval [l_start, l_end].

### 3.4 Multi-Layer Concept Extraction

Three extraction methods should be compared empirically:

1. **Delta PCA**: PCA on layer-to-layer residual deltas Δh_l = h_l - h_(l-1) within [l_start, l_end]. This captures what each layer *adds*—the construction process itself.

2. **Windowed PCA**: PCA on raw activations h_l across [l_start, l_end]. This captures the cumulative concept direction as it evolves through the assembly zone.

3. **Single-layer (baseline)**: The standard DoM vector at l_max.

If methods (1) or (2) consistently outperform (3) on downstream steering and classification tasks, this validates the dynamical framing as more than descriptive vocabulary. If they do not, the framework may still provide useful theoretical structure while failing to improve practical extraction—which is itself worth establishing.

---

## 4. Testable Predictions

The CAZ framework generates four predictions that are in principle falsifiable with existing open-weight models and standard interpretability tooling.

### 4.1 The Mid-Stream Ablation Hypothesis

**Prediction 1**: Post-CAZ ablation is destructive; early-CAZ ablation is graceful.

When orthogonal projection ablation is applied after the CAZ, the intervention acts as an amputation. The model has already organized its output strategy around the concept; removing the concept vector causes representational shock—elevated KL divergence, increased perplexity, and degraded coherence on unrelated tasks. Intervening *within* the early CAZ should produce qualitatively different behavior. If the concept is projected out of the residual stream as it is being assembled, subsequent attention heads in later layers can route around the missing information. The model should default to a neutral or safe state while preserving general capabilities.

**Experimental design**: For a given concept (e.g., refusal), extract the concept direction at each layer via DoM. Apply orthogonal projection at each layer *l* independently. Measure: (a) behavioral suppression rate on targeted prompts, (b) KL divergence from the unmodified model on unrelated prompts, (c) perplexity on a held-out benchmark. Plot the ratio (a)/(b)—behavioral suppression per unit of collateral damage—as a function of layer. The CAZ framework predicts this ratio peaks within [l_start, l_end] and degrades sharply post-l_end.

This prediction connects naturally to the abliteration literature [Arditi et al., 2024]. The CAZ framework suggests that systematically varying intervention depth—and measuring the suppression-to-capability-impact ratio as a function of layer—would provide a principled basis for selecting intervention points. The framework predicts that early-CAZ intervention would improve this ratio relative to post-peak intervention.

### 4.2 Architecture-Stable CAZ Positioning

**Prediction 2**: CAZ boundaries are concept-specific but architecture-stable.

Different concepts (refusal, honesty, credibility) should have different CAZ windows within the same model. However, the *relative* positioning of those windows—as a fraction of total model depth—should be consistent across architectures of similar parameter count and training regime. Preliminary cross-architecture results showing convergence at relative layer 12 for credibility encoding across multiple GPT-family architectures are suggestive of this pattern, though not yet confirmed on production-scale models.

### 4.3 CAZ Width and Concept Abstraction

**Prediction 3**: CAZ width correlates with concept abstraction level.

More abstract concepts (e.g., "trustworthiness," "moral valence") should have wider CAZ windows than concrete ones (e.g., "negation," "plurality"), because abstract concepts require more iterative construction across attention layers. This is testable by comparing l_end - l_start for concepts at different levels of semantic abstraction as operationalized by, for example, depth in WordNet or scores on standard concreteness rating datasets.

### 4.4 Post-CAZ Degradation as Logit Interference

**Prediction 4**: Post-CAZ re-entanglement correlates with unembedding matrix structure.

The degradation of clean concept geometry in late layers is not noise but a structural consequence of preparing the residual stream for logit projection. Concepts whose associated vocabulary tokens are distributionally similar in the unembedding space—close in embedding distance—should show more post-CAZ degradation than concepts with distributionally distinct vocabulary. This would explain why some concepts retain clean geometry into late layers (their vocabulary is well-separated) while others degrade early (their vocabulary clusters).

---

## 5. Relationship to Existing Work

**Manifold interpretability**

Gurnee et al. [2025] found curved manifolds in middle layers for character counting and explicitly called for unsupervised geometric discovery methods. The CAZ framework provides a formalism for identifying *where* in the layer stack such manifolds crystallize—the assembly zone is precisely where curved manifold structure should be most geometrically coherent and most amenable to unsupervised detection.

**SAE dark matter**

Engels et al. [2024] investigated the structured residual ("dark matter") left unexplained by sparse autoencoders, finding that approximately 50% of the SAE error vector and over 90% of its norm can be linearly predicted from the input activation—the unexplained residual is structured, not noise—and that larger SAEs largely fail to reconstruct the same contexts as smaller SAEs. Crucially, they distinguish a linearly predictable error component from a genuinely *nonlinear* residual that behaves differently from the rest. The CAZ phase model offers a candidate mechanism for part of this nonlinear component: SAEs are trained on activations at a fixed layer, but in-progress concept construction within the CAZ produces transitional representations that are neither fully syntactic (pre-CAZ) nor fully semantic (post-CAZ). These transitional states may be precisely the activations that resist linear decomposition regardless of SAE scale, connecting the assembly-zone framing to the structured nature of the dark matter Engels et al. observe.

**Abliteration**

Arditi et al. [2024] and Wollschläger et al. [2025] establish the geometry of refusal. The CAZ framework extends this by asking not just *what* the geometry is but *when* it forms, and using that temporal structure to identify optimal intervention points. Wollschläger et al.'s finding that refusal occupies multi-dimensional concept cones raises the open question of whether cone dimensionality varies across the CAZ—whether it narrows during assembly (becoming more extractable) and expands during post-CAZ degradation.

**Representation engineering**

Zou et al. [2023] extract directions for honesty, morality, power-seeking, and related concepts via contrastive stimuli. The CAZ framework provides a theoretical basis for choosing intervention depth rather than treating it as a hyperparameter, and suggests that the full CAZ window may contain richer directional information than any single layer.

---

## 6. Preliminary Empirical Results

The framework has been partially validated on the GPT-2 family using a full empirical pipeline across 8 concepts × 8 model architectures (GPT-2, GPT-2-Medium, GPT-2-Large, GPT-2-XL, GPT-Neo, Pythia, OPT variants), implemented in the companion repository [Rosetta Manifold](https://github.com/jamesrahenry/Rosetta_Manifold).

### 6.1 GPT-2-XL Results

At GPT-2-XL scale (48 layers, 100 contrastive pairs per concept):

| Concept | Type | Peak layer | Relative depth | Peak S |
|---|---|---|---|---|
| temporal\_order | relational | L36 / 48 | 75% | 0.449 |
| causation | relational | L37 / 48 | 77% | 0.488 |
| negation | syntactic | L39 / 48 | 81% | 0.314 |
| certainty | epistemic | L44 / 48 | 92% | 0.500 |
| moral\_valence | affective | L44 / 48 | 92% | 0.294 |
| sentiment | affective | L44 / 48 | 92% | 0.396 |
| credibility | epistemic | L46 / 48 | 96% | 0.736 |
| plurality | syntactic | L47 / 48 | 98% | 0.322 |

### 6.2 Confirmed Findings

**Prediction 1 (Mid-Stream Ablation Hypothesis)** is confirmed at GPT-2 scale: single-layer orthogonal projection ablation at the CAZ peak achieves 100% separation reduction on concept probes. This result does not extend to GPT-2-XL, where single-layer projection is insufficient — consistent with the expectation that 1.5B-scale models require multi-layer or weight-space intervention.

**Broad late-assembly ordering**: A clear two-cluster structure is observed. Relational and syntactic concepts (temporal\_order, causation, negation) assemble in the 75–81% depth range; affective and epistemic concepts (certainty, moral\_valence, sentiment, credibility) cluster in the 92–96% range. This separation is robust across the 8 architectures tested.

**Credibility** exhibits substantially stronger separation signal (S = 0.736) than all other concepts — more than 50% larger than the next strongest (certainty, S = 0.500) — suggesting a particularly well-defined concept direction.

### 6.3 Anomalies and Open Questions

**The within-type syntactic < relational ordering is reversed.** The framework predicted syntactic concepts would assemble earlier than relational ones; the data shows the opposite — causation and temporal\_order (relational) assemble at 75–77%, while negation (syntactic) assembles later at 81%. The ordering between types holds, but the within-type prediction does not.

**Plurality is anomalously deep** (L47, 98% depth) — the deepest concept measured, deeper even than credibility. A syntactic concept assembling in the final layers of a 48-layer model is not explained by the current framework and warrants investigation at larger scale.

**Prediction 2 (Architecture-Stable Positioning)** is partially supported — the broad two-cluster ordering holds across architectures — but proper validation requires same-scale cross-architecture comparison (e.g., GPT-Neo-1.3B vs. OPT-1.3B vs. GPT-2-XL at matched parameter count).

**Predictions 3 and 4** (CAZ width and post-CAZ logit interference) have not yet been tested and require either frontier-scale models or a wider concept vocabulary with operationalized abstraction levels.

### 6.4 Frontier-Scale Validation

Frontier-scale validation (Llama 3 70B, Qwen 2.5 72B, Mistral Large) is pending compute access. The proxy-scale results are sufficient to establish that the CAZ is a real and measurable phenomenon; they are not sufficient to confirm the architecture-stability and abstraction-width predictions at production model scale.

---

## 7. Limitations

Known limitations include:

**Computational cost**

Computing S(l) and C(l) at every layer for large contrastive datasets and large models is expensive. Efficient approximations (random projection, activation sketching) may be necessary for models beyond 7B parameters.

**Smoothing sensitivity**

CAZ boundary detection depends on the smoothing parameter *k* and threshold θ₊. A principled method for setting these—perhaps based on layer count or activation dimensionality—is needed before the framework can be applied systematically.

**Linearity assumption**

The separation metric assumes the concept manifold is approximately linearly separable. For concepts with curved or multi-dimensional structure [Gurnee et al., 2025; Engels et al., 2025], kernel-based or topological metrics may be required. This limitation is shared with most of the current interpretability literature.

**Token position dependence**

The framework does not account for which token positions carry concept information. Zhao et al. [2025] showed that harmfulness and refusal encode at different token positions. CAZ boundaries may vary by token position as well as by layer.

**Causal vs. correlational**

High separation at a layer does not establish that the concept is *used* at that layer for downstream computation. Causal validation via ablation and activation patching is required to distinguish geometrically present concepts from epiphenomenal structure.

---

## 8. Conclusion

The Concept Assembly Zone provides a framework for analyzing how Transformers construct semantic representations across their depth. By shifting from single-layer snapshots to cross-layer flow analysis, it generates specific, testable predictions about concept extraction quality, optimal ablation depth, and the geometric origins of the SAE dark matter.

The framework's value is empirical, not definitional. The minimal viable experiment is straightforward: for a small open-weight model and a well-studied concept such as refusal, compare single-layer extraction against CAZ-windowed extraction on steering effectiveness, and compare ablation at different layer depths against the CAZ-predicted optimal intervention point.

If Prediction 1 holds—if the ratio of behavioral suppression to collateral damage peaks within the CAZ window—this provides both practical guidance for safer ablation and theoretical support for the dynamical framing. If it does not, the framework's descriptive vocabulary may still be useful while its interventional claims are refuted. Either outcome advances the field.

The CAZ framework is offered as a complement to existing interpretability methods: extending the layer-sweep heuristic into a dynamical account, and connecting single-layer feature extraction to the emerging geometric program that the field is converging toward. Whether it survives contact with data is an open question we invite others to test.

---

## References

- Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., & Nanda, N. (2024). Refusal in language models is mediated by a single direction. *arXiv preprint arXiv:2406.11717*. https://arxiv.org/abs/2406.11717

- Elhage, N., Nanda, N., Olsson, C., Henighan, T., Joseph, N., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., DasSarma, N., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Brown, T., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2021). A mathematical framework for transformer circuits. *Transformer Circuits Thread*, Anthropic. https://transformer-circuits.pub/2021/framework/index.html

- Engels, J., Riggs, L., & Tegmark, M. (2024). Decomposing the dark matter of sparse autoencoders. *Transactions on Machine Learning Research (TMLR)*, April 2025. *arXiv preprint arXiv:2410.14670*. https://arxiv.org/abs/2410.14670

- Engels, J., Michaud, E. J., Liao, I., Gurnee, W., & Tegmark, M. (2025). Not all language model features are one-dimensionally linear. *Proceedings of the International Conference on Learning Representations (ICLR 2025)*. *arXiv preprint arXiv:2405.14860*. https://arxiv.org/abs/2405.14860

- Gurnee, W., Ameisen, E., Kauvar, I., Tarng, W., Pearce, A., Olah, C., & Batson, J. (2025). When models manipulate manifolds. *Transformer Circuits Thread*, Anthropic, October 2025. *arXiv preprint arXiv:2601.04480*. https://arxiv.org/abs/2601.04480

- Wollschläger, T., Elstner, J., Geisler, S., Cohen-Addad, V., Günnemann, S., & Gasteiger, J. (2025). The geometry of refusal in large language models: Concept cones and representational independence. *Proceedings of Machine Learning Research (ICML 2025)*, 267, 66945–66970. *arXiv preprint arXiv:2502.17420*. https://arxiv.org/abs/2502.17420

- Zhao, J., Huang, J., Wu, Z., Bau, D., & Shi, W. (2025). Harmfulness and refusal are distinct concepts in language models. *Advances in Neural Information Processing Systems (NeurIPS 2025)*. *arXiv preprint arXiv:2507.11878*. https://arxiv.org/abs/2507.11878

- Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., Goel, S., Li, N., Byun, M. J., Wang, Z., Mallen, A., Basart, S., Koyejo, S., Song, D., Fredrikson, M., Kolter, J. Z., & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. *arXiv preprint arXiv:2310.01405*. https://arxiv.org/abs/2310.01405
