# Architecture Decision Records

This document records the key design decisions made during the from-scratch re-implementation of [Human Motion Diffusion Model](https://arxiv.org/abs/2209.14916) (Tevet et al., ICLR 2023).

Each entry follows the format:
- **Context** — why a decision was needed
- **Decision** — what was chosen
- **Rationale** — why this option was selected
- **Trade-offs** — what was given up, and what might change in the future

---

## ADR-1: Framework — PyTorch over JAX / TensorFlow

**Status**: Accepted

**Context**: Building a from-scratch paper implementation requires a framework with strong community support, easy step-by-step debugging, and wide adoption in academic research.

**Decision**: Use PyTorch.

**Rationale**:
- The official MDM implementation is in PyTorch, so code can be directly cross-referenced
- PyTorch's eager execution mode makes it easy to inspect tensor shapes and values at any point during development
- The majority of recent diffusion model and motion generation papers publish PyTorch code
- Reduces friction when reading or borrowing from `reference/`

**Trade-offs**: JAX offers XLA compilation for faster training on TPUs and cleaner functional programming style. TensorFlow has better production serving tooling (TF Serving, TFLite). Neither was relevant for this learning-focused, single-GPU implementation.

---

## ADR-2: Backbone — Transformer Encoder over MLP / RNN

**Status**: Accepted

**Context**: Motion sequences are temporal data of variable length. Multiple architectural families could in principle be used: feedforward MLPs, recurrent networks (LSTM/GRU), or Transformers.

**Decision**: Use `nn.TransformerEncoder` with multi-head self-attention (8 layers, 8 heads, `d_model=512`).

**Rationale**:
- The MDM paper uses a Transformer, making this choice a direct translation of the architecture
- Self-attention captures long-range temporal dependencies in a single layer (e.g., foot placement on frame 1 correlated with landing on frame 30)
- Transformers naturally handle variable-length sequences via masking, which will be needed when real data with mixed-length clips is added
- Parallelism over the sequence dimension makes training faster than sequential RNNs on GPU

**Trade-offs**: Transformers are quadratic in sequence length — $O(F^2)$ attention cost — which is negligible at 60 frames but could become expensive for multi-minute sequences. RNNs are cheaper for long sequences but harder to parallelize.

---

## ADR-3: Prediction Target — $x_0$ Prediction over Noise ($\varepsilon$) Prediction

**Status**: Accepted

**Context**: Diffusion models can be trained to predict either the added noise $\varepsilon$ (noise prediction, as in original DDPM) or the clean data $x_0$ directly ($x_0$-prediction). Both are mathematically equivalent given:

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon$$

so predicting one uniquely determines the other.

**Decision**: Train the model to predict $x_0$ directly. Loss: $\mathcal{L} = \|x_0 - f_\theta(x_t, t, a)\|^2$.

**Rationale**:
- The MDM paper explicitly adopts $x_0$-prediction ("simple objective" formulation from DDPM)
- For motion data, $x_0$-prediction has a more direct physical interpretation: the model outputs a complete, interpretable pose sequence at each denoising step
- Simpler to implement and evaluate (loss magnitude is directly in pose space, not in noise space)

**Trade-offs**: Ho et al. (DDPM, 2020) found that noise prediction trained slightly more stably in image domains. For motion, the MDM authors found $x_0$-prediction works well empirically. A future experiment comparing both on HumanML3D would be informative.

---

## ADR-4: Conditioning Strategy — Prepend Tokens over Cross-Attention

**Status**: Accepted

**Context**: The model must be conditioned on two signals: action class $a$ and diffusion timestep $t$. Possible approaches include:
1. Prepend condition embeddings as extra tokens at the start of the sequence
2. Add condition embeddings element-wise to every frame embedding
3. Introduce a separate cross-attention module where motion frames attend to condition embeddings

**Decision**: Embed $a$ and $t$ separately, then prepend them as two extra tokens: `seq = cat([c_emb, t_emb, x_emb])`. Strip them from the output after the Transformer.

**Rationale**:
- Directly follows the MDM paper's design; no deviation from the reference
- Simpler to implement than a dedicated cross-attention module
- The Transformer's self-attention automatically lets every motion frame attend to both condition tokens — no special mechanism needed
- Clean separation: positions 0–1 are always conditions; positions 2–F+1 are always motion frames

**Trade-offs**: For richer text conditions (e.g., a sequence of CLIP token embeddings rather than a single vector), cross-attention would be more expressive. Prepend tokens with a single vector are a bottleneck at that point, but sufficient for the single-vector conditions used here.

---

## ADR-5: Noise Schedule — Linear over Cosine

**Status**: Accepted

**Context**: The schedule $\{\beta_t\}$ controls how quickly signal is destroyed across the 1000 diffusion steps. Two common choices:
- **Linear** (DDPM, Ho et al. 2020): $\beta_t = \text{linspace}(0.0001, 0.02, 1000)$
- **Cosine** (Improved DDPM, Nichol & Dhariwal 2021): schedule derived from $\bar{\alpha}_t = \cos^2(\cdot)$, designed to preserve more signal at low noise levels

**Decision**: Use the linear schedule.

**Rationale**:
- The MDM paper uses a linear schedule
- Simpler to implement (`torch.linspace`) and reason about
- Consistent with the reference implementation in `reference/diffusion/`
- Understanding schedule choice is secondary to understanding the Transformer architecture for the learning goal of this repo

**Trade-offs**: The cosine schedule is generally preferred for image generation because the linear schedule destroys almost all signal near $t = T$, making early denoising steps uninformative. When real motion training is added in the future, switching to cosine could improve sample quality.

---

## ADR-6: Activation Function — SiLU over ReLU / GELU (Time Embedding)

**Status**: Accepted

**Context**: The time embedding network (`Linear → activation → Linear`) requires a non-linear activation. Candidates: ReLU, GELU, SiLU (Swish).

**Decision**: Use SiLU — `nn.SiLU()` — in the time embedding MLP.

**Rationale**:
- SiLU ($f(x) = x \cdot \sigma(x)$) is smooth, differentiable everywhere, and non-monotonic, which gives the gradient useful information even for large negative inputs
- Widely used as the default activation in diffusion model time embedding networks (DDPM, Stable Diffusion, etc.)
- Smooth transitions between timestep embeddings are desirable because the conditioning signal changes continuously with $t$

**Trade-offs**: GELU is nearly identical in practice and is the default in BERT-family models. ReLU is faster (no sigmoid computation) but can produce dead neurons in conditioning MLPs. The empirical difference for a 2-layer MLP is negligible; SiLU was chosen for consistency with the broader diffusion model ecosystem.

---

## ADR-7: Simplified Scope — No Text Encoder, No Real Data

**Status**: Accepted

**Context**: The full MDM model conditions on free-form text via CLIP or DistilBERT encoders and trains on large motion-capture datasets (HumanML3D: 14,616 clips, KIT: 3,911 clips). Replicating this from scratch involves substantial data-engineering work unrelated to the diffusion mechanics.

**Decision**: Use action-class conditioning with randomly generated dummy tensors instead of text encoders and real datasets.

**Rationale**:
- The learning goal is to understand diffusion + Transformer mechanics, not to reproduce benchmark numbers
- CLIP text encoding is an independent preprocessing component; its presence or absence does not change the architecture of the denoising model
- Keeping the codebase under ~100 lines per file makes each component easy to read in isolation
- `reference/` provides the complete, runnable implementation for when real training experiments are needed

**Trade-offs**: This implementation cannot generate meaningful motions — the model has no real signal to learn from. The concrete next steps to make it functional are:
1. Implement a real data loader for HumanML3D
2. Replace dummy action labels with CLIP text embeddings (or keep action labels for the action-to-motion task)
3. Implement the reverse diffusion sampling loop (iterative denoising from $x_T \sim \mathcal{N}(0, I)$)
