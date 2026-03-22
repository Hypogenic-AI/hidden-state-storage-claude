# Research Plan: How Much Can You Store in a Hidden State?

## Motivation & Novelty Assessment

### Why This Research Matters
Hidden states are the fundamental communication channel in transformers — every layer reads from and writes to the residual stream. Understanding the information capacity of a single hidden state vector under different encoding schemes tells us (a) how efficiently transformers use their bandwidth, (b) what's the theoretical ceiling for methods like memory tokens, soft prompts, and state compression, and (c) how decoder complexity trades off against storage capacity.

### Gap in Existing Work
Kuratov et al. (2025) measured capacity empirically using per-sample gradient optimization with a full frozen LLM as decoder — achieving 1568 tokens in a 4096-dim vector. But no one has systematically compared the simpler encoding schemes that form the spectrum from "no learned decoder" to "full transformer decoder":
- Raw ASCII bit-packing (no decoder needed)
- Vocabulary ID packing (no decoder needed)
- Vector walk encoding (simple geometric decoder)
- Unembedding matrix decoding (one matrix multiply)
- Single transformer layer decoding (moderate decoder)

### Our Novel Contribution
A systematic, apples-to-apples comparison of 5 encoding schemes across the decoder-complexity spectrum, measuring information capacity (in bits and tokens) for the same model and hidden state dimensionality. This reveals how much of the capacity comes from the encoding scheme vs. the decoder.

### Experiment Justification
- **Exp 1 (ASCII)**: Baseline — pure bit-packing, no model knowledge. Sets the floor.
- **Exp 2 (Vocab)**: Uses model's vocabulary but naive packing. Tests if vocab structure helps.
- **Exp 3 (Vector walks)**: Novel geometric encoding. Tests if direction/magnitude separation helps.
- **Exp 4 (Unembedding)**: Uses model's learned mapping. Tests single-matrix decoding.
- **Exp 5 (Transformer layer decoder)**: Uses one transformer layer. Tests if moderate compute helps.

## Research Question
How does the information capacity of a transformer hidden state vector vary across encoding schemes of increasing decoder complexity, from raw ASCII bit-packing to using a transformer layer as decoder?

## Hypothesis Decomposition
1. ASCII encoding stores d_model × precision_bits / 8 characters (pure information-theoretic bound)
2. LLM vocabulary encoding stores d_model × precision_bits / log2(|V|) tokens (more info per token than ASCII)
3. Vector walk encoding can exceed naive packing by using geometric structure (direction for position, magnitude for token ID)
4. Unembedding-based decoding leverages learned token geometry to store more tokens than naive packing
5. A single transformer layer decoder significantly increases capacity over unembedding alone, approaching but not matching Kuratov et al.'s full-model results

## Proposed Methodology

### Model
GPT-2 (small, 768-dim hidden states, 50257 vocab) — well-understood, fits easily on GPU, good for systematic study. We'll also test Pythia-410M for validation.

### Encoding Schemes

**1. ASCII Encoding**
- Pack ASCII bytes (7 bits each) into float32 vector dimensions
- Each float32 can store 32 bits = 4 ASCII characters (with quantization)
- For fp16: 16 bits = 2 ASCII characters per dimension
- Theoretical: 768 × 32 / 7 ≈ 3510 ASCII chars (fp32), 768 × 16 / 7 ≈ 1755 (fp16)
- Decode: unpack bits, convert to characters

**2. LLM Vocabulary Encoding**
- Pack token IDs into vector dimensions
- Each fp32 stores floor(32 / log2(50257)) = floor(32/15.62) = 2 token IDs per dim
- Theoretical: 768 × 2 = 1536 tokens (fp32), 768 × 1 = 768 tokens (fp16)
- Decode: unpack, lookup token ID

**3. Vector Walk Encoding**
- Divide d-dimensional space into k subspaces
- In each subspace: direction encodes position, magnitude encodes token ID
- Use learned or fixed codebook of directions
- Capacity depends on angular resolution and magnitude quantization
- Theory: with k subspaces of dim d/k, can encode k tokens with position

**4. Unembedding Decoding**
- Encode: place tokens using the model's embedding matrix
- Sum token embeddings (possibly with position weights) into single vector
- Decode: multiply by unembedding matrix, take argmax for each position
- Use techniques from logit lens / tuned lens
- Capacity limited by linear separability of summed embeddings

**5. Single Transformer Layer Decoder**
- Encode: optimize a single vector via gradient descent
- Decode: pass through one transformer layer + unembedding
- Compare frozen random layer vs. actual model layer
- This is a simplified version of Kuratov et al.'s approach

### Evaluation Metrics
- **Bits stored**: Total recoverable information in bits
- **Tokens stored**: Number of tokens perfectly decoded (accuracy=1.0)
- **Capacity utilization**: Fraction of theoretical maximum
- **Bits per dimension**: Efficiency measure

### Baselines
- Theoretical maximum: d × b / log2(|V|) for token capacity
- Kuratov et al. results (from paper): 1568 tokens for Llama-3.1-8B (4096-dim)
- Random chance

## Expected Outcomes
- ASCII < Vocab < Vector Walk < Unembedding < Transformer Layer
- The gap between each step quantifies the value of decoder complexity
- Unembedding should provide a big jump over naive encoding due to learned geometry
- Transformer layer should approach but not match full-model optimization

## Timeline
- Phase 1 (Planning): 20 min ✓
- Phase 2 (Setup): 10 min
- Phase 3 (Implementation): 60 min
- Phase 4 (Experiments): 60 min
- Phase 5 (Analysis): 30 min
- Phase 6 (Documentation): 20 min

## Potential Challenges
- Vector walk encoding is novel — may need iteration on the specific scheme
- Gradient optimization for single-layer decoder may be tricky to tune
- Need to handle numerical precision carefully for bit-packing schemes
