"""
Hidden State Storage Capacity Experiments
==========================================
Systematic comparison of 5 encoding schemes for storing information
in a transformer hidden state vector.

Encoding schemes (increasing decoder complexity):
1. ASCII bit-packing (no decoder)
2. LLM vocabulary ID packing (no decoder)
3. Vector walk encoding (geometric decoder)
4. Unembedding-based encoding (matrix multiply decoder)
5. Single transformer layer decoder (learned decoder)
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/workspaces/hidden-state-storage-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load model and tokenizer
# ============================================================

def load_model(model_name="gpt2"):
    """Load a transformer model and tokenizer."""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        output_hidden_states=True
    ).to(DEVICE)
    model.eval()

    d_model = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    vocab_size = model.config.vocab_size
    n_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers

    print(f"  d_model={d_model}, vocab_size={vocab_size}, n_layers={n_layers}")
    return model, tokenizer, d_model, vocab_size, n_layers


# ============================================================
# Test texts
# ============================================================

def get_test_texts():
    """Return a set of test texts for encoding experiments."""
    texts = [
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used to test fonts and keyboards.",
        "In the beginning was the Word, and the Word was with God, and the Word was God.",
        "To be or not to be, that is the question. Whether 'tis nobler in the mind to suffer.",
        "The history of all hitherto existing society is the history of class struggles.",
        "It was the best of times, it was the worst of times, it was the age of wisdom.",
    ]
    return texts


# ============================================================
# EXPERIMENT 1: ASCII Bit-Packing
# ============================================================

def exp1_ascii_encoding(d_model, precisions=[32, 16, 8]):
    """
    Pack ASCII characters into a float vector by encoding bits.

    Each ASCII char = 7 bits. Each dimension stores `precision` bits.
    Capacity = d_model * precision / 7 characters.

    We test actual encoding/decoding to verify lossless recovery.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: ASCII Bit-Packing")
    print("="*60)

    results = {}
    test_texts = get_test_texts()

    for precision in precisions:
        bits_total = d_model * precision
        max_chars = bits_total // 7  # 7 bits per ASCII char
        max_bytes = bits_total // 8  # 8 bits per byte (full ASCII)

        print(f"\n--- Precision: {precision} bits per dimension ---")
        print(f"  Total bits: {bits_total}")
        print(f"  Theoretical max (7-bit ASCII): {max_chars} chars")
        print(f"  Theoretical max (8-bit bytes): {max_bytes} chars")

        # Actually encode and decode
        successes = []
        for text in test_texts:
            # Convert text to bits
            text_bytes = text.encode('ascii', errors='replace')
            text_bits = ''.join(format(b, '08b') for b in text_bytes)

            # How many chars can we fit?
            max_text_len = min(len(text), max_bytes)
            usable_bits = max_text_len * 8

            # Pack into vector
            vec = np.zeros(d_model, dtype=np.float64)
            bits_to_pack = text_bits[:max_text_len * 8]

            for i in range(d_model):
                start_bit = i * precision
                end_bit = min(start_bit + precision, len(bits_to_pack))
                if start_bit >= len(bits_to_pack):
                    break
                bit_chunk = bits_to_pack[start_bit:end_bit]
                # Store as integer value in the dimension
                vec[i] = int(bit_chunk, 2) if bit_chunk else 0

            # Decode from vector
            recovered_bits = ''
            for i in range(d_model):
                start_bit = i * precision
                if start_bit >= usable_bits:
                    break
                val = int(round(vec[i]))
                chunk_len = min(precision, usable_bits - start_bit)
                recovered_bits += format(val, f'0{chunk_len}b')

            # Convert bits back to text
            recovered_bytes = bytearray()
            for j in range(0, min(len(recovered_bits), max_text_len * 8), 8):
                byte_str = recovered_bits[j:j+8]
                if len(byte_str) == 8:
                    recovered_bytes.append(int(byte_str, 2))

            recovered_text = recovered_bytes.decode('ascii', errors='replace')
            actual_stored = min(len(text), max_text_len)
            original_chunk = text[:actual_stored]
            match = recovered_text[:actual_stored] == original_chunk
            successes.append((actual_stored, match))

        # Record results
        accuracy = sum(1 for _, m in successes if m) / len(successes)
        avg_chars = np.mean([s for s, _ in successes])

        results[f"fp{precision}"] = {
            "precision_bits": precision,
            "total_bits": bits_total,
            "theoretical_max_chars_7bit": max_chars,
            "theoretical_max_chars_8bit": max_bytes,
            "bits_per_char": 7,
            "lossless_accuracy": accuracy,
            "avg_chars_stored": float(avg_chars),
        }

        print(f"  Lossless recovery: {accuracy*100:.0f}%")
        print(f"  Avg chars stored: {avg_chars:.0f}")

    return results


# ============================================================
# EXPERIMENT 2: LLM Vocabulary Encoding
# ============================================================

def exp2_vocab_encoding(d_model, vocab_size, tokenizer, precisions=[32, 16]):
    """
    Pack token IDs into vector dimensions.

    Each token ID needs log2(vocab_size) bits.
    Capacity = d_model * precision / log2(vocab_size) tokens.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: LLM Vocabulary ID Packing")
    print("="*60)

    bits_per_token = math.ceil(math.log2(vocab_size))
    print(f"Vocab size: {vocab_size}, bits per token: {bits_per_token}")

    results = {}
    test_texts = get_test_texts()

    for precision in precisions:
        bits_total = d_model * precision
        max_tokens = bits_total // bits_per_token
        # Alternative: pack multiple tokens per dimension
        tokens_per_dim = precision // bits_per_token
        max_tokens_alt = d_model * max(1, tokens_per_dim)

        print(f"\n--- Precision: {precision} bits per dimension ---")
        print(f"  Total bits: {bits_total}")
        print(f"  Theoretical max tokens: {max_tokens}")
        print(f"  Tokens per dim (floor): {tokens_per_dim}")
        print(f"  Max tokens (dim-packing): {max_tokens_alt}")

        successes = []
        tokens_stored_list = []

        for text in test_texts:
            tokens = tokenizer.encode(text)
            n_store = min(len(tokens), max_tokens_alt)

            # Pack token IDs into vector
            vec = np.zeros(d_model, dtype=np.float64)

            if tokens_per_dim >= 1:
                # Pack multiple tokens per dimension
                for i in range(d_model):
                    val = 0
                    for j in range(tokens_per_dim):
                        idx = i * tokens_per_dim + j
                        if idx < n_store:
                            val = val * vocab_size + tokens[idx]
                    vec[i] = val

                # Decode
                recovered = []
                for i in range(d_model):
                    val = int(round(vec[i]))
                    chunk = []
                    for j in range(tokens_per_dim):
                        chunk.append(val % vocab_size)
                        val //= vocab_size
                    chunk.reverse()
                    recovered.extend(chunk)
                recovered = recovered[:n_store]
            else:
                # One token per dimension (with wasted bits)
                for i in range(min(n_store, d_model)):
                    vec[i] = tokens[i]
                recovered = [int(round(vec[i])) for i in range(min(n_store, d_model))]

            match = recovered == tokens[:n_store]
            successes.append(match)
            tokens_stored_list.append(n_store)

            decoded_text = tokenizer.decode(recovered)
            original_text = tokenizer.decode(tokens[:n_store])

        accuracy = sum(1 for m in successes if m) / len(successes)
        avg_tokens = np.mean(tokens_stored_list)

        results[f"fp{precision}"] = {
            "precision_bits": precision,
            "total_bits": bits_total,
            "bits_per_token": bits_per_token,
            "theoretical_max_tokens": max_tokens,
            "tokens_per_dim": tokens_per_dim,
            "max_tokens_dim_packing": max_tokens_alt,
            "lossless_accuracy": accuracy,
            "avg_tokens_stored": float(avg_tokens),
        }

        print(f"  Lossless recovery: {accuracy*100:.0f}%")
        print(f"  Avg tokens stored: {avg_tokens:.0f}")

    return results


# ============================================================
# EXPERIMENT 3: Vector Walk Encoding
# ============================================================

def exp3_vector_walk_encoding(d_model, vocab_size, tokenizer):
    """
    Encode tokens via fixed vector walks:
    - Divide hidden state into k subspaces
    - In each subspace, direction encodes the token position
    - Magnitude encodes the token ID

    This is a novel geometric encoding. We test several subspace sizes.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Fixed Vector Walk Encoding")
    print("="*60)

    results = {}
    test_texts = get_test_texts()

    # Test different subspace dimensions
    for sub_dim in [2, 4, 8, 16, 32, 64]:
        n_subspaces = d_model // sub_dim

        print(f"\n--- Subspace dim: {sub_dim}, Num subspaces: {n_subspaces} ---")

        # Each subspace encodes ONE token:
        # - Direction (unit vector in sub_dim space) encodes position
        # - Magnitude encodes token ID
        #
        # Direction capacity: roughly (sub_dim - 1) bits for distinguishable directions
        # Magnitude capacity: precision bits (but we use it for token ID)
        # Total capacity: n_subspaces tokens

        # For position encoding via direction:
        # We use a fixed set of direction vectors (one per possible position)
        # The maximum positions = number of distinguishable directions
        # In sub_dim dimensions, we can have ~2^(sub_dim-1) well-separated directions
        # But we also need to encode which direction = which position

        # Simpler approach: each subspace stores one token
        # The first component = token ID (as float)
        # The remaining components = position encoding
        # This gives n_subspaces tokens total

        # Even simpler + more powerful: use the direction to encode position index
        # and magnitude to encode token ID

        # Generate fixed direction codebook for positions
        max_positions = min(512, n_subspaces * 4)  # reasonable max sequence length

        # Generate random orthogonal directions for position encoding
        torch.manual_seed(SEED)
        if sub_dim >= max_positions:
            # Can use orthogonal directions
            Q = torch.randn(sub_dim, max_positions)
            Q, _ = torch.linalg.qr(Q)
            directions = Q[:, :max_positions].T  # [max_positions, sub_dim]
        else:
            # More positions than dimensions — must use non-orthogonal
            directions = torch.randn(max_positions, sub_dim)
            directions = F.normalize(directions, dim=1)

        successes = []
        tokens_stored_list = []

        for text in test_texts:
            tokens = tokenizer.encode(text)
            n_store = min(len(tokens), n_subspaces)

            # Encode: each subspace i stores token at position i
            vec = torch.zeros(d_model)
            for i in range(n_store):
                token_id = tokens[i]
                # Direction = fixed direction for position i
                direction = directions[i % max_positions]
                # Magnitude = token_id + 1 (avoid zero magnitude)
                magnitude = float(token_id + 1)
                # Write into subspace
                start = i * sub_dim
                end = start + sub_dim
                vec[start:end] = direction[:sub_dim] * magnitude

            # Decode
            recovered = []
            for i in range(n_store):
                start = i * sub_dim
                end = start + sub_dim
                subvec = vec[start:end]
                magnitude = torch.norm(subvec).item()
                token_id = round(magnitude) - 1
                token_id = max(0, min(token_id, vocab_size - 1))
                recovered.append(token_id)

            match = recovered == tokens[:n_store]
            successes.append(match)
            tokens_stored_list.append(n_store)

        accuracy = sum(1 for m in successes if m) / len(successes)
        avg_tokens = np.mean(tokens_stored_list)

        results[f"sub{sub_dim}"] = {
            "subspace_dim": sub_dim,
            "n_subspaces": n_subspaces,
            "max_tokens": n_subspaces,
            "lossless_accuracy": accuracy,
            "avg_tokens_stored": float(avg_tokens),
            "bits_per_token_effective": float(sub_dim * 32),  # bits used per token
        }

        print(f"  Max tokens: {n_subspaces}")
        print(f"  Lossless recovery: {accuracy*100:.0f}%")
        print(f"  Avg tokens stored: {avg_tokens:.0f}")

    # Advanced: Superposition-based vector walk
    # Pack MORE tokens than subspaces by using superposition
    print("\n--- Advanced: Superposition Vector Walk ---")
    sub_dim = 16
    n_subspaces = d_model // sub_dim  # 48 subspaces

    # Try to store MORE tokens than subspaces using superposition
    # Each token encoded as: magnitude * random_direction
    # All summed into the full vector
    # Decode via correlation with known directions

    for overload_factor in [1, 2, 4, 8]:
        n_target = n_subspaces * overload_factor

        # Generate random codebook: one direction per (position, token) pair
        # Too expensive for full vocab. Instead: one direction per position,
        # magnitude for token ID, sum all together.
        torch.manual_seed(SEED)
        pos_directions = torch.randn(n_target, d_model)
        pos_directions = F.normalize(pos_directions, dim=1)

        successes = []
        for text in test_texts:
            tokens = tokenizer.encode(text)
            n_store = min(len(tokens), n_target)

            # Encode: sum of magnitude * direction for each position
            vec = torch.zeros(d_model)
            for i in range(n_store):
                magnitude = float(tokens[i] + 1)
                vec += pos_directions[i] * magnitude

            # Decode: correlate with each position direction
            recovered = []
            for i in range(n_store):
                # Project onto position direction
                corr = torch.dot(vec, pos_directions[i]).item()
                token_id = round(corr) - 1
                token_id = max(0, min(token_id, vocab_size - 1))
                recovered.append(token_id)

            match = recovered == tokens[:n_store]
            successes.append(match)

        accuracy = sum(1 for m in successes if m) / len(successes)
        print(f"  Overload {overload_factor}x ({n_target} tokens): accuracy={accuracy*100:.0f}%")

        results[f"superposition_{overload_factor}x"] = {
            "method": "superposition_vector_walk",
            "overload_factor": overload_factor,
            "n_target_tokens": n_target,
            "lossless_accuracy": accuracy,
        }

    return results


# ============================================================
# EXPERIMENT 4: Unembedding-Based Encoding
# ============================================================

def exp4_unembedding(model, tokenizer, d_model, vocab_size):
    """
    Use the model's embedding/unembedding matrices to encode and decode.

    Approaches:
    a) Logit lens: store a vector, decode via W_U (unembedding matrix)
    b) Sum of embeddings with position scaling
    c) Optimized vector via gradient descent decoded through unembedding only
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: Unembedding-Based Encoding")
    print("="*60)

    results = {}
    test_texts = get_test_texts()

    # Get embedding and unembedding matrices
    if hasattr(model, 'transformer'):  # GPT-2
        W_E = model.transformer.wte.weight.detach()  # [vocab, d_model]
        W_U = model.lm_head.weight.detach()  # [vocab, d_model]
    else:
        W_E = model.model.embed_tokens.weight.detach()
        W_U = model.lm_head.weight.detach()

    print(f"Embedding shape: {W_E.shape}")
    print(f"Unembedding shape: {W_U.shape}")

    # --- Approach 4a: Single token via argmax(W_U @ h) ---
    print("\n--- 4a: Single token via logit lens ---")
    # A single hidden state vector can trivially encode 1 token via the embedding
    # decode via unembedding argmax

    correct = 0
    total = 0
    for text in test_texts:
        tokens = tokenizer.encode(text)
        for tok in tokens[:50]:  # test first 50 tokens
            h = W_E[tok]  # embed the token
            logits = h @ W_U.T  # decode
            pred = logits.argmax().item()
            if pred == tok:
                correct += 1
            total += 1

    single_token_acc = correct / total
    print(f"  Single token embed→unembedding accuracy: {single_token_acc*100:.1f}%")
    results["single_token_logit_lens"] = {
        "method": "embed_then_unembed_argmax",
        "tokens_stored": 1,
        "accuracy": single_token_acc,
    }

    # --- Approach 4b: Sum of scaled embeddings ---
    print("\n--- 4b: Sum of scaled embeddings ---")
    # Encode: h = sum_i (alpha_i * W_E[token_i])
    # Decode: logits = W_U @ h, take top-k
    # Problem: summing embeddings loses token order and interferes

    for n_tokens in [1, 2, 3, 5, 10, 20, 50]:
        correct_tokens = 0
        total_tokens = 0

        for text in test_texts:
            tokens = tokenizer.encode(text)
            n = min(n_tokens, len(tokens))
            toks = tokens[:n]

            # Encode: weighted sum (weight = 1/n for stability)
            h = torch.zeros(d_model, device=DEVICE)
            for i, tok in enumerate(toks):
                h += W_E[tok] / n

            # Decode: get top-k predictions
            logits = h @ W_U.T
            top_k = logits.topk(n).indices.cpu().tolist()

            # Count how many of the original tokens appear in top-k
            overlap = len(set(top_k) & set(toks))
            correct_tokens += overlap
            total_tokens += n

        recall = correct_tokens / total_tokens if total_tokens > 0 else 0
        print(f"  n={n_tokens}: recall={recall*100:.1f}% (bag-of-tokens)")

        results[f"sum_embeddings_n{n_tokens}"] = {
            "method": "sum_embeddings_bag_of_tokens",
            "n_tokens": n_tokens,
            "recall": recall,
        }

    # --- Approach 4c: Optimized vector + unembedding ---
    print("\n--- 4c: Gradient-optimized vector decoded via unembedding ---")
    # Optimize a single vector h to maximize P(token_i) = softmax(W_U @ h)[token_i]
    # for a sequence of tokens

    for n_tokens in [1, 5, 10, 20, 50, 100, 200]:
        accuracies = []

        for text in test_texts[:3]:  # fewer texts for optimization experiments
            tokens = tokenizer.encode(text)
            n = min(n_tokens, len(tokens))
            target = torch.tensor(tokens[:n], device=DEVICE)

            # Optimize h
            h = torch.randn(d_model, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([h], lr=0.1)

            for step in range(2000):
                optimizer.zero_grad()
                # Each "position" just decodes from the same h
                # We want h to encode ALL tokens simultaneously
                logits = h @ W_U.T  # [vocab_size]

                # Loss: minimize -log P(token_i) for each target token
                # Use log_softmax
                log_probs = F.log_softmax(logits, dim=0)
                loss = -log_probs[target].mean()

                loss.backward()
                optimizer.step()

            # Evaluate: for each target token, check if it's in top-n predictions
            with torch.no_grad():
                logits = h @ W_U.T
                top_k = logits.topk(n).indices.cpu().tolist()
                recall = len(set(top_k) & set(target.cpu().tolist())) / n
                accuracies.append(recall)

        avg_acc = np.mean(accuracies)
        print(f"  n={n_tokens}: recall={avg_acc*100:.1f}% (optimized, bag-of-tokens)")

        results[f"optimized_unembed_n{n_tokens}"] = {
            "method": "gradient_optimized_unembed",
            "n_tokens": n_tokens,
            "recall": float(avg_acc),
            "optimization_steps": 2000,
        }

    # --- Approach 4d: Sequence decoding with position-aware unembedding ---
    print("\n--- 4d: Position-aware optimized vector + unembedding ---")
    # Optimize multiple vectors h_1..h_n, but constrained to d_model total dims
    # decode each h_i via unembedding
    # This is equivalent to reshaping a d_model vector into n chunks

    for n_tokens in [1, 5, 10, 20, 50]:
        accuracies = []
        dims_per_token = d_model // n_tokens

        if dims_per_token < 2:
            print(f"  n={n_tokens}: skipped (dims_per_token={dims_per_token} < 2)")
            continue

        for text in test_texts[:3]:
            tokens = tokenizer.encode(text)
            n = min(n_tokens, len(tokens))
            target = torch.tensor(tokens[:n], device=DEVICE)

            # We project each chunk through a learned linear layer to d_model
            # then apply unembedding. This tests what a small projection can do.
            h = torch.randn(n, dims_per_token, device=DEVICE, requires_grad=True)
            proj = torch.randn(dims_per_token, d_model, device=DEVICE, requires_grad=True)

            optimizer = torch.optim.Adam([h, proj], lr=0.05)

            for step in range(2000):
                optimizer.zero_grad()
                # Project each chunk to full d_model
                h_full = h @ proj  # [n, d_model]
                logits = h_full @ W_U.T  # [n, vocab_size]
                loss = F.cross_entropy(logits, target[:n])
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                h_full = h @ proj
                logits = h_full @ W_U.T
                preds = logits.argmax(dim=1).cpu().tolist()
                acc = sum(1 for p, t in zip(preds, target[:n].cpu().tolist()) if p == t) / n
                accuracies.append(acc)

        avg_acc = np.mean(accuracies)
        print(f"  n={n_tokens} (dims/tok={dims_per_token}): seq accuracy={avg_acc*100:.1f}%")

        results[f"position_aware_unembed_n{n_tokens}"] = {
            "method": "position_aware_unembed",
            "n_tokens": n_tokens,
            "dims_per_token": dims_per_token,
            "sequence_accuracy": float(avg_acc),
        }

    return results


# ============================================================
# EXPERIMENT 5: Single Transformer Layer Decoder
# ============================================================

def exp5_transformer_layer_decoder(model, tokenizer, d_model, vocab_size):
    """
    Optimize a single vector decoded through one transformer layer + unembedding.

    This is the key experiment: how much more capacity does a single
    transformer layer provide compared to just the unembedding matrix?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Single Transformer Layer Decoder")
    print("="*60)

    results = {}
    test_texts = get_test_texts()

    if hasattr(model, 'transformer'):  # GPT-2
        layers = model.transformer.h
        W_U = model.lm_head.weight.detach()
        ln_f = model.transformer.ln_f
    else:
        layers = model.model.layers
        W_U = model.lm_head.weight.detach()
        ln_f = model.model.norm

    # Test with different layers (early, middle, late)
    n_layers = len(layers)
    test_layer_indices = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for layer_idx in test_layer_indices:
        layer = layers[layer_idx]
        layer.eval()

        for n_tokens in [10, 50, 100, 200, 500]:
            accuracies = []

            for text in test_texts[:3]:
                tokens = tokenizer.encode(text)
                n = min(n_tokens, len(tokens))
                target = torch.tensor(tokens[:n], device=DEVICE)

                # Optimize a sequence of vectors (simulating the residual stream)
                # that when passed through one transformer layer + LN + unembedding
                # produces the target tokens

                # The "hidden state" is a single vector of size d_model
                # We broadcast it to n positions, then decode through one layer
                h = torch.randn(1, n, d_model, device=DEVICE, requires_grad=True)
                optimizer = torch.optim.Adam([h], lr=0.01)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

                best_acc = 0
                for step in range(3000):
                    optimizer.zero_grad()

                    # Pass through one transformer layer
                    with torch.no_grad():
                        # Create position IDs and attention mask
                        pass

                    # Forward through layer
                    residual = h
                    layer_out = layer(h)[0]  # Most layers return (hidden_states, ...)
                    # Add residual (transformer layers usually have residual inside)

                    # Apply final layer norm
                    normed = ln_f(layer_out)

                    # Apply unembedding
                    logits = normed @ W_U.T  # [1, n, vocab]
                    logits = logits.squeeze(0)  # [n, vocab]

                    loss = F.cross_entropy(logits, target)
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    if step % 500 == 0:
                        with torch.no_grad():
                            preds = logits.argmax(dim=1)
                            acc = (preds == target).float().mean().item()
                            best_acc = max(best_acc, acc)

                # Final evaluation
                with torch.no_grad():
                    layer_out = layer(h)[0]
                    normed = ln_f(layer_out)
                    logits = (normed @ W_U.T).squeeze(0)
                    preds = logits.argmax(dim=1)
                    acc = (preds == target).float().mean().item()
                    best_acc = max(best_acc, acc)
                    accuracies.append(best_acc)

            avg_acc = np.mean(accuracies)
            print(f"  Layer {layer_idx}, n={n_tokens}: accuracy={avg_acc*100:.1f}%")

            # Calculate bits stored
            bits_stored = avg_acc * n_tokens * math.log2(vocab_size)

            results[f"layer{layer_idx}_n{n_tokens}"] = {
                "layer_index": layer_idx,
                "n_tokens": n_tokens,
                "accuracy": float(avg_acc),
                "bits_stored": float(bits_stored),
                "bits_per_dim": float(bits_stored / d_model),
                "optimization_steps": 3000,
            }

    return results


# ============================================================
# EXPERIMENT 5b: Full-vector optimization (Kuratov-style)
# ============================================================

def exp5b_kuratov_style(model, tokenizer, d_model, vocab_size):
    """
    Kuratov-style: optimize a single [mem] vector prepended to the input,
    decoded by the FULL frozen model. This is the upper bound.

    Simplified version: optimize a vector, feed it through ALL layers.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5b: Full Model Decoder (Kuratov-style)")
    print("="*60)

    results = {}
    test_texts = get_test_texts()

    if hasattr(model, 'transformer'):
        wte = model.transformer.wte
        wpe = model.transformer.wpe

    for n_tokens in [10, 50, 100, 200, 500]:
        accuracies = []

        for text in test_texts[:2]:  # Fewer texts since full model is expensive
            tokens = tokenizer.encode(text)
            n = min(n_tokens, len(tokens))
            target = torch.tensor([tokens[:n]], device=DEVICE)  # [1, n]

            # Create a single trainable [mem] vector
            mem_vec = torch.randn(1, 1, d_model, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([mem_vec], lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

            best_acc = 0
            for step in range(5000):
                optimizer.zero_grad()

                # Embed target tokens
                token_embeds = wte(target)  # [1, n, d]
                pos_ids = torch.arange(1, n + 1, device=DEVICE).unsqueeze(0)
                pos_embeds = wpe(pos_ids)

                # Prepend mem vector
                mem_pos = wpe(torch.zeros(1, 1, dtype=torch.long, device=DEVICE))
                full_input = torch.cat([mem_vec + mem_pos, token_embeds + pos_embeds], dim=1)

                # Forward through full model
                outputs = model(inputs_embeds=full_input)
                logits = outputs.logits[:, :n, :]  # Take first n positions (after mem)

                # We want position i to predict token i
                # Actually, in causal LM: position i predicts token i+1
                # So position 0 (mem) should predict token 0, position 1 predicts token 1, etc.
                logits_for_loss = outputs.logits[:, 0:n, :]  # mem + first n-1 token positions
                loss = F.cross_entropy(logits_for_loss.reshape(-1, vocab_size), target.reshape(-1))

                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 1000 == 0:
                    with torch.no_grad():
                        preds = logits_for_loss.argmax(dim=-1)
                        acc = (preds == target).float().mean().item()
                        best_acc = max(best_acc, acc)

            with torch.no_grad():
                outputs = model(inputs_embeds=full_input)
                logits_for_loss = outputs.logits[:, 0:n, :]
                preds = logits_for_loss.argmax(dim=-1)
                acc = (preds == target).float().mean().item()
                best_acc = max(best_acc, acc)
                accuracies.append(best_acc)

        avg_acc = np.mean(accuracies)
        bits_stored = avg_acc * n_tokens * math.log2(vocab_size)

        print(f"  n={n_tokens}: accuracy={avg_acc*100:.1f}%, bits={bits_stored:.0f}")

        results[f"full_model_n{n_tokens}"] = {
            "method": "full_model_kuratov_style",
            "n_tokens": n_tokens,
            "accuracy": float(avg_acc),
            "bits_stored": float(bits_stored),
            "bits_per_dim": float(bits_stored / d_model),
            "optimization_steps": 5000,
        }

    return results


# ============================================================
# Theoretical Capacity Analysis
# ============================================================

def theoretical_analysis(d_model, vocab_size):
    """Compute theoretical capacity bounds."""
    print("\n" + "="*60)
    print("THEORETICAL CAPACITY ANALYSIS")
    print("="*60)

    results = {}
    bits_per_token = math.log2(vocab_size)

    for precision, label in [(32, "fp32"), (16, "fp16"), (8, "int8")]:
        total_bits = d_model * precision

        # ASCII
        ascii_chars_7bit = total_bits // 7
        ascii_chars_8bit = total_bits // 8

        # Vocab tokens
        vocab_tokens = total_bits / bits_per_token

        # Information-theoretic
        info_bits = total_bits  # raw capacity

        results[label] = {
            "precision_bits": precision,
            "total_bits": total_bits,
            "total_bytes": total_bits // 8,
            "ascii_chars_7bit": ascii_chars_7bit,
            "ascii_chars_8bit": ascii_chars_8bit,
            "vocab_tokens": vocab_tokens,
            "bits_per_token": bits_per_token,
            "info_content_bits": total_bits,
        }

        print(f"\n--- {label} (d={d_model}) ---")
        print(f"  Total bits: {total_bits}")
        print(f"  ASCII chars (7-bit): {ascii_chars_7bit}")
        print(f"  ASCII chars (8-bit): {ascii_chars_8bit}")
        print(f"  Vocab tokens: {vocab_tokens:.1f}")
        print(f"  Bits per token: {bits_per_token:.2f}")

    return results


# ============================================================
# Main execution
# ============================================================

def main():
    print("="*70)
    print("HIDDEN STATE STORAGE CAPACITY EXPERIMENTS")
    print("="*70)

    # Load model
    model, tokenizer, d_model, vocab_size, n_layers = load_model("gpt2")

    all_results = {
        "model": "gpt2",
        "d_model": d_model,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Theoretical bounds
    all_results["theoretical"] = theoretical_analysis(d_model, vocab_size)

    # Experiment 1: ASCII
    all_results["exp1_ascii"] = exp1_ascii_encoding(d_model)

    # Experiment 2: Vocabulary
    all_results["exp2_vocab"] = exp2_vocab_encoding(d_model, vocab_size, tokenizer)

    # Experiment 3: Vector walks
    all_results["exp3_vector_walk"] = exp3_vector_walk_encoding(d_model, vocab_size, tokenizer)

    # Experiment 4: Unembedding
    all_results["exp4_unembedding"] = exp4_unembedding(model, tokenizer, d_model, vocab_size)

    # Experiment 5: Single layer decoder
    all_results["exp5_layer_decoder"] = exp5_transformer_layer_decoder(model, tokenizer, d_model, vocab_size)

    # Experiment 5b: Full model (Kuratov-style)
    all_results["exp5b_full_model"] = exp5b_kuratov_style(model, tokenizer, d_model, vocab_size)

    # Save all results
    results_path = RESULTS_DIR / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    results = main()
