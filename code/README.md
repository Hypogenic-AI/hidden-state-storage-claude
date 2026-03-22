# Cloned Repositories

## Repo 1: hidden_capacity
- **URL**: https://github.com/yurakuratov/hidden_capacity
- **Purpose**: Official code for "Cramming 1568 Tokens into a Single Vector" (ACL 2025)
- **Location**: `code/hidden_capacity/`
- **Key files**: Scripts for training [mem] vectors, evaluating compression capacity
- **Notes**: Requires PyTorch, HuggingFace Transformers. Uses per-sample optimization with frozen LLM. Most directly relevant codebase for reproducing encoding experiments.

## Repo 2: tuned-lens
- **URL**: https://github.com/AlignmentResearch/tuned-lens
- **Purpose**: Train and evaluate tuned lens probes for decoding hidden states
- **Location**: `code/tuned-lens/`
- **Key files**: Tuned lens training, evaluation, and visualization code
- **Notes**: Provides the unembedding-based decoder approach. Can be used to measure what information is decodable at each layer.

## Repo 3: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Mechanistic interpretability library for GPT-style models
- **Location**: `code/TransformerLens/`
- **Key files**: `transformer_lens/` package with HookedTransformer, activation caching
- **Notes**: Primary tool for accessing hidden states, residual stream, attention patterns. Supports 50+ open-source models. Essential for any encoding experiment.

## Repo 4: future-lens
- **URL**: https://github.com/KoyenaPal/future-lens
- **Purpose**: Code for "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State"
- **Location**: `code/future-lens/`
- **Key files**: Linear probes, causal intervention, learned prompt methods
- **Notes**: Demonstrates that single hidden states encode multi-token sequences. Useful for understanding what decoder complexity achieves.

## Repo 5: nnsight
- **URL**: https://github.com/ndif-team/nnsight
- **Purpose**: Architecture-agnostic library for neural network interpretability and intervention
- **Location**: `code/nnsight/`
- **Key files**: Core intervention API, remote execution support
- **Notes**: Alternative to TransformerLens for activation extraction and editing. Supports any PyTorch model.
