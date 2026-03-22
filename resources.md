# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "How much can you store in a hidden state?" including papers, datasets, and code repositories.

## Papers
Total papers downloaded: 20

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Cramming 1568 Tokens into a Single Vector | Kuratov et al. | 2025 | papers/2502.13063_*.pdf | Core paper: measures capacity of single embedding vector |
| Toy Models of Superposition | Elhage et al. (Anthropic) | 2022 | papers/2209.10652_*.pdf | How models store more features than dimensions |
| Tuned Lens | Belrose et al. | 2023 | papers/2303.08112_*.pdf | Decoding hidden states via learned affine + unembedding |
| Future Lens | Pal et al. | 2023 | papers/2311.04897_*.pdf | Single hidden states encode multi-token futures |
| How Much Do LMs Memorize? | Morris et al. | 2025 | papers/2505.24832_*.pdf | ~3.5-4 bits per parameter |
| Optimal Memorization Capacity | Kajitsuka, Sato | 2024 | papers/2409.17677_*.pdf | O(sqrt(N)) parameters for N sequences |
| Upper/Lower Memory Capacity | - | 2024 | papers/2405.13718_*.pdf | Formal capacity bounds |
| Softmax Bottleneck | Yang et al. | 2018 | papers/1711.03953_*.pdf | Hidden dim << vocab size bottleneck |
| Analyzing Transformers in Embedding Space | Dar et al. | 2023 | papers/2209.02535_*.pdf | Projecting parameters into vocab space |
| Patchscopes | Ghandeharioun et al. | 2024 | papers/2401.06102_*.pdf | Unified hidden state inspection |
| BERT Classical NLP Pipeline | Tenney et al. | 2019 | papers/1905.05950_*.pdf | Layer-wise linguistic encoding |
| Probing with MDL | Voita, Titov | 2020 | papers/2003.12298_*.pdf | Information-theoretic probing |
| FFN as Key-Value Memories | Geva et al. | 2021 | papers/2012.14913_*.pdf | How FFN stores knowledge |
| Knowledge Neurons | Dai et al. | 2022 | papers/2104.08696_*.pdf | Specific neurons for facts |
| Watermark for LLMs | Kirchenbauer et al. | 2023 | papers/2301.10226_*.pdf | Vocabulary-based bit encoding |
| Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | Directional encoding of concepts |
| Linear Representation Hypothesis | Park et al. | 2023 | papers/2311.03658_*.pdf | Concepts as linear directions |
| nGPT Hypersphere | Loshchilov et al. | 2024 | papers/2410.01131_*.pdf | Vector walks on unit hypersphere |
| Steganographic Potentials of LMs | - | 2025 | papers/2505.03439_*.pdf | Covert encoding via RL |
| In-context Learning & Induction Heads | Olsson et al. (Anthropic) | 2022 | papers/2209.11895_*.pdf | Residual stream communication |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-103 (test) | HuggingFace Salesforce/wikitext | 4,358 samples | Text for compression experiments | datasets/wikitext103/ | Alternative to PG-19 (deprecated loader) |
| Random Word Sequences | Generated (seed=42) | 100 samples, 64-512 words | Capacity isolation (no LM advantage) | datasets/random_words.json | Following Kuratov et al. methodology |

See datasets/README.md for download instructions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| hidden_capacity | github.com/yurakuratov/hidden_capacity | [mem] vector compression | code/hidden_capacity/ | Core experimental code |
| tuned-lens | github.com/AlignmentResearch/tuned-lens | Hidden state decoding | code/tuned-lens/ | Unembedding-based decoder |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Mechanistic interpretability | code/TransformerLens/ | Activation access for 50+ models |
| future-lens | github.com/KoyenaPal/future-lens | Future token prediction | code/future-lens/ | Causal intervention methods |
| nnsight | github.com/ndif-team/nnsight | Neural network intervention | code/nnsight/ | Architecture-agnostic probing |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used web search to identify key papers across 7 research areas (hidden state capacity, superposition, logit/tuned lens, probing, steganography, knowledge storage, representation geometry)
2. Downloaded 20 papers from arXiv covering the full spectrum from theoretical bounds to practical encoding
3. Deep-read 4 core papers (Kuratov, Elhage, Belrose, Pal) with full methodology extraction
4. Identified code repositories from paper citations and GitHub search

### Selection Criteria
- Papers directly measuring or theoretically bounding hidden state information capacity
- Papers demonstrating encoding/decoding schemes for hidden states
- Papers providing tools or frameworks for hidden state analysis
- Preference for recent work (2023-2025) with available code

### Challenges Encountered
- PG-19 dataset HuggingFace loader is deprecated; used WikiText-103 as alternative
- Paper-finder service timed out; relied on comprehensive web search instead

### Gaps and Workarounds
- No single paper systematically compares all encoding schemes mentioned in the hypothesis
- ASCII encoding and fixed vector walk encoding are not well-studied in existing literature
- The "using a Transformer layer as a decoder" approach is implicit in tuned lens but not framed as an encoding scheme comparison

## Recommendations for Experiment Design

Based on gathered resources, we recommend:

1. **Primary dataset(s)**: WikiText-103 test split (available) + random word sequences (available). Optionally download PG-19 for direct comparison with Kuratov et al.

2. **Baseline methods (encoding schemes to compare)**:
   - **ASCII encoding**: Encode text as ASCII bits, pack into vector dimensions
   - **LLM vocabulary encoding**: Map token IDs to vector positions
   - **Fixed vector walks**: Encode token identity as direction and position as magnitude in subspaces
   - **Unembedding-based**: Use the model's embedding matrix to encode, unembedding to decode
   - **Transformer layer decoder**: Use one or more frozen transformer layers as the decoder (à la tuned lens, Kuratov [mem])
   - **Theoretical upper bound**: d_model × b_precision / log2(|V|) tokens

3. **Evaluation metrics**: Token accuracy at threshold 0.99 (Kuratov), information gain (CE-reduction), capacity utilization ratio

4. **Code to adapt/reuse**:
   - `hidden_capacity`: Core [mem] vector optimization loop — adapt for different encoding schemes
   - `TransformerLens`: Access model internals for custom encoding/decoding
   - `tuned-lens`: Pre-trained layer-wise decoders as one decoding scheme
   - `nnsight`: Alternative intervention framework
