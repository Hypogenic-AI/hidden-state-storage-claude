# How Much Can You Store in a Hidden State?

A systematic comparison of information storage capacity in transformer hidden state vectors across five encoding schemes of increasing decoder complexity.

## Key Findings

- **ASCII bit-packing** stores the most raw information: 3,072 bytes in a 768-dim fp32 vector (lossless)
- **LLM vocabulary encoding** stores 1,536 tokens per fp32 vector (2 tokens/dim, lossless)
- **Unembedding-based decoding** with a shared projection achieves 100% accuracy for 200 random tokens at 3 dims/token, failing sharply at 2 dims/token
- **One transformer layer** as decoder only stores ~10 random tokens from a single vector — barely better than position embeddings alone
- **Full model (GPT-2)** with a single [mem] vector stores ~50 random tokens (74% accuracy); with 32 vectors reaches 99.4% on 500 tokens
- **Superposition encoding** (summing token*direction) fails completely for token IDs (unlike sparse binary features)

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install torch transformers numpy scipy matplotlib pandas tqdm einops accelerate

# Run experiments
python src/experiments_v2.py    # Main experiments (natural text)
python src/experiments_v3_random.py  # Random token experiments (key results)
python src/visualize.py         # Generate plots
```

## File Structure

```
REPORT.md                  # Full research report with results
planning.md                # Research plan
src/
  experiments.py           # V1 experiments (initial)
  experiments_v2.py        # V2: corrected single-vector experiments
  experiments_v3_random.py # V3: random token experiments (key)
  visualize.py             # Plot generation
results/
  all_results.json         # V1 results
  capacity_results_v3_random.json  # V3 random token results
  plots/                   # All figures
literature_review.md       # Literature survey
resources.md               # Resource catalog
papers/                    # Downloaded papers
datasets/                  # Test datasets
code/                      # Reference implementations
```

## Model

GPT-2 small (d_model=768, vocab=50,257, 12 layers). See [REPORT.md](REPORT.md) for full details.
