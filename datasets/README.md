# Downloaded Datasets

This directory contains datasets for the research project on hidden state information capacity. Large data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-103 (Test Split)

### Overview
- **Source**: HuggingFace `Salesforce/wikitext` (wikitext-103-raw-v1)
- **Size**: 4,358 samples (test split)
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Text for compression/encoding experiments
- **License**: Creative Commons Attribution-ShareAlike
- **Why chosen**: Standard benchmark text corpus, well-suited for measuring hidden state capacity on natural language. Used as an alternative to PG-19 (whose HuggingFace loader is deprecated).

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
dataset.save_to_disk("datasets/wikitext103")
```

### Loading

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext103")
```

## Dataset 2: Random Word Sequences

### Overview
- **Source**: Generated locally (following Kuratov et al., 2025 methodology)
- **Size**: 100 samples, lengths [64, 128, 256, 512] words
- **Format**: JSON
- **Task**: Capacity measurement without language model prediction advantage
- **License**: N/A (synthetic)
- **Why chosen**: Random sequences eliminate the LM's ability to predict tokens from language knowledge, isolating the pure capacity of the [mem] vector encoding. Follows methodology from the "Cramming 1568 Tokens" paper.

### Loading

```python
import json
with open("datasets/random_words.json") as f:
    data = json.load(f)
```

### Notes
- Random words sampled from a vocabulary of ~170 common English words (seed=42)
- Kuratov et al. used top 100K GloVe words; for full replication, download GloVe vocabulary from https://nlp.stanford.edu/data/glove.6B.zip
- The key insight from literature: for random text, Token Gain ≈ Decoding Capacity (no language prediction advantage)

## Dataset 3: PG-19 (Recommended for full experiments)

### Overview
- **Source**: Project Gutenberg books via `pg19` dataset
- **Size**: ~28,000 books (train), test split ~100 books
- **Format**: Plain text
- **Task**: Primary benchmark for text compression into hidden states
- **Why chosen**: Used by Kuratov et al. (2025) and widely used in long-context research

### Download Instructions

PG-19 HuggingFace loader is deprecated. Download directly:
```bash
# Download from Google Cloud Storage
gsutil -m cp -r gs://deepmind-gutenberg/train ./datasets/pg19/train
gsutil -m cp -r gs://deepmind-gutenberg/test ./datasets/pg19/test
```

Or use the raw text files:
```python
# Alternative: download individual books from Project Gutenberg
import requests
# Example: download "The Adventures of Sherlock Holmes"
url = "https://www.gutenberg.org/files/1661/1661-0.txt"
text = requests.get(url).text
```
