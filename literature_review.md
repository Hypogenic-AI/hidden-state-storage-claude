# Literature Review: How Much Can You Store in a Hidden State?

## Research Area Overview

This review surveys work on the information capacity of hidden states in neural networks, particularly transformers and LLMs. The central question is: how much information can be encoded in a single hidden-state vector, and how does the encoding scheme affect capacity? We organize findings around five themes: (1) theoretical and empirical capacity limits, (2) superposition and compressed representations, (3) decoding hidden states via unembedding/lens techniques, (4) what hidden states encode beyond next-token, and (5) steganographic encoding in LLM outputs.

---

## Key Papers

### Paper 1: Cramming 1568 Tokens into a Single Vector (Kuratov et al., 2025)
- **ArXiv**: 2502.13063 (ACL 2025, Oral)
- **Key Contribution**: Directly measures how many tokens can be losslessly compressed into a single input embedding vector using per-sample optimization of trainable [mem] vectors with a frozen LLM decoder.
- **Methodology**: Prepend trainable vectors to frozen LLM input, optimize via cross-entropy loss for 5000 steps. Measure decoding capacity, token gain, and information gain (cross-entropy reduction).
- **Key Results**:
  - Llama-3.1-8B: 1568 tokens from a single 4096-dim vector (theoretical max: ~1931 tokens for 16-bit floats)
  - Capacity utilization: only ~5-30% of theoretical maximum is used
  - Information gain is constant per model regardless of text type (natural vs random)
  - Capacity scales linearly with number of [mem] vectors (7168 tokens from 16 vectors with Llama-3.2-1B)
  - Works for non-transformer architectures too (Mamba SSMs show same pattern)
  - Compression limit determined by cross-entropy, not text length
- **Datasets Used**: PG-19 (books), AO3 fanfics (post-Oct 2024), random word sequences from GloVe vocabulary
- **Code Available**: https://github.com/yurakuratov/hidden_capacity
- **Relevance**: **Most directly relevant paper**. Provides empirical upper bounds on hidden state capacity across encoding schemes. The gap between theoretical (d_model × b / log2|V|) and practical capacity is the central finding.

### Paper 2: Toy Models of Superposition (Elhage et al., 2022)
- **ArXiv**: 2209.10652 (Anthropic)
- **Key Contribution**: Demonstrates that neural networks store more features than they have dimensions via "superposition," exploiting feature sparsity. Provides phase diagrams showing when superposition occurs.
- **Methodology**: Train small ReLU autoencoders (x → W^T ReLU(Wx + b)) on synthetic sparse data. Vary feature importance and sparsity. Analyze learned weight geometry.
- **Key Results**:
  - With dense features: model learns orthogonal basis for top-k features (PCA-like)
  - With sparse features: model stores MORE features than dimensions using non-orthogonal directions
  - Phase change: transition from dedicated to superposed representations depends on sparsity
  - Geometric structure: features organize as uniform polytopes (pentagons, tetrahedra)
  - Computation can occur in superposition (e.g., absolute value circuits)
  - Hypothesis: real networks are "noisily simulating" much larger sparse networks
- **Code Available**: https://github.com/anthropics/toy-models-of-superposition
- **Relevance**: Provides theoretical framework for understanding WHY hidden states can store more information than their dimensionality suggests. Superposition is the mechanism enabling high-capacity encoding.

### Paper 3: Eliciting Latent Predictions with the Tuned Lens (Belrose et al., 2023)
- **ArXiv**: 2303.08112
- **Key Contribution**: Trains per-layer affine transformations ("translators") to decode hidden states into vocabulary distributions, fixing the logit lens's unreliability and bias.
- **Methodology**: For each layer l, learn affine transform (A_l, b_l) minimizing KL divergence between tuned lens output and final layer logits. Uses distillation loss to avoid learning extra information.
- **Key Results**:
  - Substantially lower perplexity than logit lens across all layers
  - Unbiased: marginal distribution matches final layer (unlike logit lens which has 4-5 bits bias)
  - Representational drift between layers is smooth but significant
  - Translators transfer across nearby layers and to fine-tuned model variants
  - Features influential on tuned lens are also influential on the model (causal basis extraction)
  - Prediction trajectory converges monotonically to final output
- **Code Available**: https://github.com/AlignmentResearch/tuned-lens
- **Relevance**: Provides the key tool for **unembedding-based decoding** of hidden states. Shows that different layers encode information in drifting bases, requiring learned translators rather than raw unembedding.

### Paper 4: Future Lens (Pal et al., 2023)
- **ArXiv**: 2311.04897 (CoNLL 2023)
- **Key Contribution**: Shows that a single hidden state at position t encodes information about tokens at positions t+2, t+3, etc. — not just the immediate next token.
- **Methodology**: Three methods on GPT-J-6B: (1) linear probes predicting future hidden states, (2) fixed-prompt causal intervention (transplanting hidden states), (3) learned-prompt causal intervention.
- **Key Results**:
  - Learned prompt method: 48.4% accuracy predicting N=1 token ahead from a single hidden state
  - 43.7% for N=2, 46.9% for N=3 (well above bigram baseline of ~20%)
  - Peak accuracy at middle layers (layer 14 of 28), NOT final layer
  - Information is broadly encoded, not just for named entities
  - Accuracy correlates with model confidence in next-token prediction
- **Code Available**: https://github.com/KoyenaPal/future-lens
- **Relevance**: Demonstrates that hidden states store MULTI-TOKEN information, directly relevant to capacity questions. The decoder choice (linear vs learned prompt using transformer layers) dramatically affects what can be extracted.

### Paper 5: Transformer Feed-Forward Layers Are Key-Value Memories (Geva et al., 2021)
- **ArXiv**: 2012.14913 (EMNLP 2021)
- **Key Contribution**: Shows FFN layers operate as key-value memories where keys match input patterns and values induce vocabulary distributions.
- **Relevance**: Explains part of HOW information is stored in hidden states — via distributed key-value patterns in FFN weights, with lower layers storing shallow patterns and upper layers storing semantic ones.

### Paper 6: Analyzing Transformers in Embedding Space (Dar et al., 2023)
- **ArXiv**: 2209.02535 (ACL 2023)
- **Key Contribution**: Projects all transformer parameters (attention, MLP weights) into vocabulary space for zero-pass interpretability.
- **Relevance**: Provides methodology for understanding hidden states through the lens of vocabulary encoding, where weight matrices are interpretable as vocabulary-space operations.

### Paper 7: Information-Theoretic Probing with MDL (Voita & Titov, 2020)
- **ArXiv**: 2003.12298 (EMNLP 2020)
- **Key Contribution**: Recasts probing as compression — measures bits needed to describe labels given representations. More stable and informative than accuracy-based probing.
- **Relevance**: Provides proper information-theoretic framework for measuring how much information hidden states encode about specific properties.

### Paper 8: How Much Do Language Models Memorize? (Morris et al., 2025)
- **ArXiv**: 2505.24832
- **Key Contribution**: Shows GPT-style transformers store approximately 3.5-4 bits of information per model parameter.
- **Relevance**: Provides per-parameter capacity bounds, complementing Kuratov et al.'s per-vector bounds.

### Paper 9: On the Optimal Memorization Capacity of Transformers (Kajitsuka & Sato, 2024)
- **ArXiv**: 2409.17677
- **Key Contribution**: Proves transformers can memorize N input sequences with O(sqrt(N)) parameters in next-token prediction, and shows this is optimal up to log factors.
- **Relevance**: Theoretical capacity bounds for transformers.

### Paper 10: Breaking the Softmax Bottleneck (Yang et al., 2018)
- **ArXiv**: 1711.03953 (ICLR 2018)
- **Key Contribution**: Shows hidden state dimension (d ~ 10^2-3) is much smaller than vocabulary (|V| ~ 10^5), creating a rank-deficient softmax bottleneck limiting expressiveness.
- **Relevance**: Identifies a fundamental constraint on how much the hidden state can express when decoded through the unembedding matrix.

### Paper 11: Patchscopes (Ghandeharioun et al., 2024)
- **ArXiv**: 2401.06102
- **Key Contribution**: Unifies logit lens, tuned lens, and probing under a framework that patches hidden states between inference passes.
- **Relevance**: Shows the LLM itself can be used as a decoder for hidden states, often outperforming vocabulary projection and probing methods.

### Paper 12: nGPT: Normalized Transformer on the Hypersphere (Loshchilov et al., 2024)
- **ArXiv**: 2410.01131
- **Key Contribution**: Normalizes all vectors to unit norm on a hypersphere. Each layer contributes a displacement on the surface.
- **Relevance**: Directly relevant to "fixed vector walks" encoding scheme — where magnitude and direction encode information. nGPT constrains representations to a hypersphere, making direction the primary information carrier.

### Paper 13: The Steganographic Potentials of Language Models (2025)
- **ArXiv**: 2505.03439
- **Key Contribution**: Shows LLMs can develop covert encoding schemes through RL fine-tuning.
- **Relevance**: Relevant to understanding how information can be encoded in token sequences using vocabulary-level encoding schemes.

### Paper 14: A Watermark for Large Language Models (Kirchenbauer et al., 2023)
- **ArXiv**: 2301.10226
- **Key Contribution**: Encodes bits into generated text by partitioning vocabulary into "green" and "red" lists and biasing generation toward green tokens.
- **Relevance**: Demonstrates a practical vocabulary-based encoding scheme that hides information in token selections — relevant to understanding bits-per-token capacity of vocabulary encoding.

### Paper 15: Linear Representation Hypothesis (Park et al., 2023)
- **ArXiv**: 2311.03658
- **Key Contribution**: Formalizes that high-level concepts are represented as linear directions in LLM representation space.
- **Relevance**: If concepts are linear directions, the capacity of a hidden state is bounded by how many near-orthogonal directions can fit in d dimensions — connecting to compressed sensing theory.

### Paper 16: Representation Engineering (Zou et al., 2023)
- **ArXiv**: 2310.01405
- **Key Contribution**: Reads and controls hidden state representations for safety properties (honesty, harmlessness). Shows concepts are encoded as directions.
- **Relevance**: Demonstrates that specific semantic concepts can be encoded/decoded from hidden state directions, providing evidence for directional encoding capacity.

---

## Common Methodologies

- **Per-sample optimization**: Optimize trainable vectors to encode text, use frozen LLM as decoder (Kuratov et al.)
- **Probing classifiers**: Train linear/affine probes on hidden states to predict linguistic properties
- **Logit lens / Tuned lens**: Apply unembedding matrix (possibly with learned affine transform) to decode hidden states
- **Causal intervention**: Transplant hidden states between contexts to test what information transfers
- **Sparse autoencoders**: Decompose hidden states into interpretable features, revealing superposition
- **Information-theoretic analysis**: Measure mutual information, cross-entropy reduction, MDL

## Standard Baselines
- Logit lens (raw unembedding) vs. tuned lens (learned affine + unembedding)
- Linear probes vs. nonlinear probes
- Bigram/n-gram baselines for prediction tasks
- Theoretical capacity bounds: d_model × b / log2|V|
- Standard compression algorithms (zlib, bz2, arithmetic coding)

## Evaluation Metrics
- **Token-level accuracy**: Fraction of correctly decoded tokens
- **Information Gain / CE-reduction**: Bits of cross-entropy reduced by the encoding
- **Precision@k**: Top-k accuracy for predicted tokens
- **Surprisal**: Negative log probability of top predicted token
- **KL divergence**: Between probe output and model output
- **Capacity utilization**: Empirical capacity / theoretical maximum

## Datasets in the Literature
- **PG-19**: Public domain books from Project Gutenberg (Kuratov et al., Belrose et al.)
- **The Pile**: 825GB diverse text corpus (Future Lens, tuned lens evaluation)
- **GovReport**: Government reports (Kuratov et al., for compressed vector structure analysis)
- **Random word sequences**: From GloVe vocabulary (Kuratov et al.)
- **AO3 Fanfics**: Post-training-cutoff fanfiction texts (Kuratov et al.)

## Gaps and Opportunities
1. **Encoding scheme comparison**: No systematic comparison of ASCII, vocabulary, vector walk, and unembedding-based encoding schemes for the same model
2. **Decoder choice impact**: How much does decoder complexity (identity, linear, affine, transformer layer) affect measured capacity?
3. **Quantization effects**: How does reducing precision (fp32 → fp16 → int8) affect practical capacity?
4. **Architecture comparison**: Limited comparison across architectures (transformer, SSM, RNN) for same task
5. **Superposition-aware encoding**: Can knowledge of superposition geometry enable better encoding?

## Recommendations for Our Experiment

Based on literature review:
- **Primary dataset**: PG-19 (well-established, used by key prior work, publicly available)
- **Secondary dataset**: Random token sequences (controls for model language knowledge)
- **Recommended baselines**:
  - Theoretical capacity bound (d × b / log2|V|)
  - Logit lens (raw unembedding)
  - Tuned lens (learned affine + unembedding)
  - Per-sample [mem] vector optimization (Kuratov method)
  - ASCII bit encoding
  - Vocabulary-based encoding
- **Recommended metrics**: Token accuracy, information gain (CE-reduction), capacity utilization ratio
- **Recommended models**: Pythia suite (160M-2.8B) for scaling analysis, Llama-3.2-1B for main experiments
- **Key tools**: TransformerLens, nnsight, hidden_capacity codebase
