"""
Hidden State Storage Capacity - V2: Finding Capacity Limits
============================================================
Corrected experiments that test how much info a SINGLE d_model vector can store.

Key difference from v1: all experiments use a SINGLE vector of d_model dims.
The question is: how many tokens can be recovered from that one vector?
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("/workspaces/hidden-state-storage-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name="gpt2"):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    d_model = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    vocab_size = model.config.vocab_size
    n_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers
    print(f"  d_model={d_model}, vocab={vocab_size}, layers={n_layers}")
    return model, tokenizer, d_model, vocab_size, n_layers


def get_long_text(tokenizer, min_tokens=2000):
    """Generate a long text for capacity testing."""
    texts = [
        "The quick brown fox jumps over the lazy dog. " * 50,
        "In the beginning was the Word, and the Word was with God. " * 50,
        "To be or not to be, that is the question. " * 50,
        "The history of all hitherto existing society is the history of class struggles. " * 30,
    ]
    full_text = " ".join(texts)
    tokens = tokenizer.encode(full_text)
    return tokens[:min_tokens]


# ============================================================
# EXPERIMENT 1: Theoretical bounds (no decoder needed)
# ============================================================

def exp_theoretical(d_model, vocab_size):
    """Pure information-theoretic capacity bounds."""
    print("\n" + "="*60)
    print("THEORETICAL CAPACITY BOUNDS")
    print("="*60)

    bits_per_token = math.log2(vocab_size)
    results = {}

    for precision, label in [(32, "fp32"), (16, "fp16"), (8, "int8")]:
        total_bits = d_model * precision
        ascii_7bit = total_bits // 7
        ascii_8bit = total_bits // 8
        vocab_tokens = total_bits / bits_per_token

        print(f"\n{label}: {total_bits} bits total")
        print(f"  ASCII (7-bit): {ascii_7bit} chars")
        print(f"  ASCII (8-bit): {ascii_8bit} bytes")
        print(f"  Vocab tokens:  {vocab_tokens:.1f} (log2|V|={bits_per_token:.2f})")

        results[label] = {
            "total_bits": total_bits,
            "ascii_7bit_chars": ascii_7bit,
            "ascii_8bit_chars": ascii_8bit,
            "vocab_tokens": round(vocab_tokens, 1),
        }

    return results


# ============================================================
# EXPERIMENT 2: ASCII encoding (verified implementation)
# ============================================================

def exp_ascii(d_model):
    """Bit-pack ASCII into a vector. Verify lossless encode/decode."""
    print("\n" + "="*60)
    print("EXPERIMENT: ASCII Bit-Packing (Verified)")
    print("="*60)

    text = ("The quick brown fox jumps over the lazy dog. " * 200)
    text_bytes = text.encode('ascii')

    results = {}
    for precision in [32, 16, 8]:
        max_bytes = (d_model * precision) // 8
        n_store = min(len(text_bytes), max_bytes)

        # Encode
        vec = np.zeros(d_model, dtype=np.float64)
        byte_arr = np.frombuffer(text_bytes[:n_store], dtype=np.uint8)

        # Pack bytes_per_dim bytes into each dimension
        bytes_per_dim = precision // 8
        for i in range(d_model):
            val = 0
            for j in range(bytes_per_dim):
                idx = i * bytes_per_dim + j
                if idx < n_store:
                    val = (val << 8) | int(byte_arr[idx])
            vec[i] = val

        # Decode
        recovered = bytearray()
        for i in range(d_model):
            val = int(round(vec[i]))
            chunk = []
            for j in range(bytes_per_dim):
                chunk.append(val & 0xFF)
                val >>= 8
            chunk.reverse()
            recovered.extend(chunk)

        recovered = bytes(recovered[:n_store])
        match = recovered == text_bytes[:n_store]

        print(f"\nfp{precision}: stored {n_store} bytes ({n_store} ASCII chars), lossless={match}")

        results[f"fp{precision}"] = {
            "precision": precision,
            "bytes_stored": n_store,
            "chars_stored": n_store,
            "lossless": bool(match),
            "total_bits_used": d_model * precision,
            "efficiency": n_store * 8 / (d_model * precision),
        }

    return results


# ============================================================
# EXPERIMENT 3: Vocab encoding (pack token IDs)
# ============================================================

def exp_vocab(d_model, vocab_size, tokenizer):
    """Pack token IDs into vector dimensions."""
    print("\n" + "="*60)
    print("EXPERIMENT: Vocabulary ID Packing")
    print("="*60)

    tokens = get_long_text(tokenizer)
    bits_per_token = math.ceil(math.log2(vocab_size))  # 16 bits for GPT-2

    results = {}
    for precision in [32, 16]:
        tokens_per_dim = precision // bits_per_token
        max_tokens = d_model * max(1, tokens_per_dim)
        n_store = min(len(tokens), max_tokens)

        # Encode
        vec = np.zeros(d_model, dtype=np.int64)
        if tokens_per_dim >= 2:
            for i in range(d_model):
                val = 0
                for j in range(tokens_per_dim):
                    idx = i * tokens_per_dim + j
                    if idx < n_store:
                        val = val * vocab_size + tokens[idx]
                vec[i] = val
        else:
            for i in range(min(n_store, d_model)):
                vec[i] = tokens[i]

        # Decode
        recovered = []
        if tokens_per_dim >= 2:
            for i in range(d_model):
                val = int(vec[i])
                chunk = []
                for j in range(tokens_per_dim):
                    chunk.append(val % vocab_size)
                    val //= vocab_size
                chunk.reverse()
                recovered.extend(chunk)
        else:
            recovered = [int(vec[i]) for i in range(min(n_store, d_model))]

        recovered = recovered[:n_store]
        match = recovered == tokens[:n_store]
        decoded_text = tokenizer.decode(recovered[:20])

        print(f"\nfp{precision}: stored {n_store} tokens, lossless={match}")
        print(f"  tokens_per_dim={tokens_per_dim}, sample: '{decoded_text}...'")

        results[f"fp{precision}"] = {
            "precision": precision,
            "bits_per_token": bits_per_token,
            "tokens_per_dim": max(1, tokens_per_dim),
            "tokens_stored": n_store,
            "lossless": bool(match),
        }

    return results


# ============================================================
# EXPERIMENT 4: Vector Walk Encoding
# ============================================================

def exp_vector_walk(d_model, vocab_size, tokenizer):
    """
    Encode tokens via geometric structure in a SINGLE vector.

    Method: Divide d_model into subspaces. Each subspace encodes one token.
    Within each subspace: store the token ID directly in one dimension,
    and use the others for position information or redundancy.

    The key limit: d_model / sub_dim tokens per vector.
    With sub_dim=1: d_model tokens (just packing IDs, same as vocab encoding)
    With sub_dim=2: d_model/2 tokens with direction encoding
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Vector Walk Encoding")
    print("="*60)

    tokens = get_long_text(tokenizer)
    results = {}

    # Method A: Pure magnitude encoding (1 dim per token)
    # This is actually equivalent to vocab encoding with 1 token per dim
    print("\n--- Method A: Magnitude only (1 dim/token) ---")
    n_store = min(len(tokens), d_model)
    vec = torch.zeros(d_model)
    for i in range(n_store):
        vec[i] = float(tokens[i])

    recovered = [int(round(vec[i].item())) for i in range(n_store)]
    match = recovered == tokens[:n_store]
    print(f"  Stored {n_store} tokens, lossless={match}")
    results["magnitude_only"] = {"tokens": n_store, "lossless": bool(match), "dims_per_token": 1}

    # Method B: Direction + Magnitude (2 dims per token)
    print("\n--- Method B: Direction + Magnitude (2 dims/token) ---")
    n_store = min(len(tokens), d_model // 2)
    vec = torch.zeros(d_model)
    for i in range(n_store):
        # Direction encodes position (angle = 2π * position / max_positions)
        angle = 2 * math.pi * i / d_model
        magnitude = float(tokens[i] + 1)
        vec[2*i] = magnitude * math.cos(angle)
        vec[2*i + 1] = magnitude * math.sin(angle)

    # Decode
    recovered = []
    for i in range(n_store):
        x, y = vec[2*i].item(), vec[2*i+1].item()
        magnitude = math.sqrt(x**2 + y**2)
        token_id = round(magnitude) - 1
        recovered.append(max(0, min(token_id, vocab_size - 1)))

    match = recovered == tokens[:n_store]
    print(f"  Stored {n_store} tokens, lossless={match}")
    results["dir_mag_2d"] = {"tokens": n_store, "lossless": bool(match), "dims_per_token": 2}

    # Method C: Superposition encoding (try to exceed 1 token per dim)
    print("\n--- Method C: Superposition encoding ---")
    # Use random projection: each token gets a random direction in d_model space
    # Store: vec = sum_i (token_i * direction_i)
    # Decode: project onto each direction

    for n_target in [d_model // 2, d_model, d_model * 2, d_model * 4]:
        n_store = min(len(tokens), n_target)
        torch.manual_seed(SEED)
        directions = torch.randn(n_target, d_model)
        directions = F.normalize(directions, dim=1)

        # Encode
        vec = torch.zeros(d_model)
        for i in range(n_store):
            vec += float(tokens[i]) * directions[i]

        # Decode via projection
        recovered = []
        for i in range(n_store):
            proj = torch.dot(vec, directions[i]).item()
            token_id = round(proj)
            token_id = max(0, min(token_id, vocab_size - 1))
            recovered.append(token_id)

        acc = sum(1 for a, b in zip(recovered, tokens[:n_store]) if a == b) / n_store
        ratio = n_store / d_model
        print(f"  n={n_store} ({ratio:.1f}x d_model): accuracy={acc*100:.1f}%")

        results[f"superposition_{n_target}"] = {
            "tokens": n_store,
            "overload_ratio": round(ratio, 2),
            "accuracy": round(acc, 4),
        }

    # Method D: Compressed sensing recovery (OMP/LASSO for sparse signals)
    print("\n--- Method D: Compressed sensing (sparse recovery) ---")
    # If we know tokens come from a finite vocab, we can use sparse recovery
    # Treat as: vec = D @ x, where D is dictionary, x is sparse indicator
    # This only works well when sequence has few UNIQUE tokens

    for n_target in [50, 100, 200, 500]:
        n_store = min(len(tokens), n_target)
        toks = tokens[:n_store]
        unique_tokens = list(set(toks))
        n_unique = len(unique_tokens)

        # Create measurement matrix: random projection to d_model
        torch.manual_seed(SEED)
        A = torch.randn(d_model, n_store) / math.sqrt(d_model)

        # Signal: one-hot token ID at each position
        signal = torch.zeros(n_store)
        for i in range(n_store):
            signal[i] = float(toks[i])

        # Measurement
        measurement = A @ signal

        # Recovery via pseudoinverse (works when n_store <= d_model)
        if n_store <= d_model:
            recovered_signal = torch.linalg.lstsq(A, measurement).solution
            recovered = [max(0, min(round(x.item()), vocab_size-1)) for x in recovered_signal]
            acc = sum(1 for a, b in zip(recovered, toks) if a == b) / n_store
        else:
            acc = 0.0

        print(f"  n={n_store} (unique={n_unique}): lstsq accuracy={acc*100:.1f}%")
        results[f"compressed_sensing_{n_target}"] = {
            "tokens": n_store,
            "unique_tokens": n_unique,
            "accuracy": round(acc, 4),
        }

    return results


# ============================================================
# EXPERIMENT 5: Unembedding-based decoding
# ============================================================

def exp_unembedding(model, tokenizer, d_model, vocab_size):
    """
    Optimize a SINGLE d_model vector to encode n tokens,
    decoded ONLY through the unembedding matrix W_U.

    The decoder is: for position i, predict token_i = argmax(W_U @ h)
    But with a single h, all positions get the same prediction!

    So we need position-aware decoding:
    Option A: h encodes a bag of tokens (unordered)
    Option B: learn a small position-dependent projection, then unembed
    Option C: use different subsets of h dims for different positions
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Unembedding-Based Decoding")
    print("="*60)

    if hasattr(model, 'transformer'):
        W_E = model.transformer.wte.weight.detach()
        W_U = model.lm_head.weight.detach()
    else:
        W_E = model.model.embed_tokens.weight.detach()
        W_U = model.lm_head.weight.detach()

    results = {}
    tokens = get_long_text(tokenizer)

    # --- Method A: Bag-of-tokens (single vector, unordered) ---
    print("\n--- A: Optimized bag-of-tokens via unembedding ---")
    for n_tokens in [1, 5, 10, 20, 50, 100, 200, 500, 768]:
        n = min(n_tokens, len(tokens))
        target = torch.tensor(tokens[:n], device=DEVICE)

        h = torch.randn(d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([h], lr=0.1)

        for step in range(3000):
            optimizer.zero_grad()
            logits = h @ W_U.T  # [vocab]
            log_probs = F.log_softmax(logits, dim=0)
            loss = -log_probs[target].mean()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            logits = h @ W_U.T
            top_k_preds = logits.topk(n).indices.cpu().tolist()
            target_set = set(target.cpu().tolist())
            recall = len(set(top_k_preds) & target_set) / len(target_set)

        unique_targets = len(target_set)
        print(f"  n={n} (unique={unique_targets}): recall={recall*100:.1f}%")
        results[f"bag_n{n}"] = {
            "n_tokens": n, "unique_targets": unique_targets,
            "recall": round(recall, 4), "method": "bag_of_tokens"
        }

    # --- Method B: Position-aware via learned linear transform ---
    print("\n--- B: Position-aware (learned W_pos @ h → unembed) ---")
    for n_tokens in [5, 10, 20, 50, 100, 200, 384, 500]:
        n = min(n_tokens, len(tokens))
        target = torch.tensor(tokens[:n], device=DEVICE)

        # h is single vector, W_pos maps h to per-position representations
        h = torch.randn(d_model, device=DEVICE, requires_grad=True)
        W_pos = torch.randn(n, d_model, d_model, device=DEVICE) * 0.01
        W_pos.requires_grad_(True)

        optimizer = torch.optim.Adam([h, W_pos], lr=0.01)

        for step in range(3000):
            optimizer.zero_grad()
            # Each position gets its own linear transform of h
            h_per_pos = torch.einsum('npd,d->np', W_pos, h)  # [n, d_model]
            logits = h_per_pos @ W_U.T  # [n, vocab]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            h_per_pos = torch.einsum('npd,d->np', W_pos, h)
            logits = h_per_pos @ W_U.T
            preds = logits.argmax(dim=1)
            acc = (preds == target).float().mean().item()

        print(f"  n={n}: accuracy={acc*100:.1f}%")

        # Note: this has n*d*d + d parameters, which is way more than d
        # The real question: how much info is in h alone?
        # W_pos is shared across texts, h is text-specific
        total_params_h = d_model
        total_params_wpos = n * d_model * d_model
        print(f"    (h params={total_params_h}, W_pos params={total_params_wpos})")

        results[f"posaware_n{n}"] = {
            "n_tokens": n, "accuracy": round(acc, 4),
            "h_params": total_params_h, "wpos_params": total_params_wpos,
            "method": "position_aware_linear"
        }

    # --- Method C: Subspace partition (pure h, no extra params) ---
    print("\n--- C: Subspace partition (dims/token from single h) ---")
    for n_tokens in [1, 5, 10, 20, 50, 100, 200, 384, 768]:
        n = min(n_tokens, len(tokens))
        if n == 0:
            continue
        dims_per_tok = d_model // n
        if dims_per_tok < 1:
            print(f"  n={n}: skipped (not enough dims)")
            continue

        target = torch.tensor(tokens[:n], device=DEVICE)

        # Optimize h and a FIXED shared projection from dims_per_tok → d_model
        h = torch.randn(d_model, device=DEVICE, requires_grad=True)
        proj = torch.randn(dims_per_tok, d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([h, proj], lr=0.05)

        best_acc = 0.0
        for step in range(5000):
            optimizer.zero_grad()
            # Split h into n chunks, project each, unembed
            chunks = h[:n * dims_per_tok].reshape(n, dims_per_tok)
            h_proj = chunks @ proj  # [n, d_model]
            logits = h_proj @ W_U.T  # [n, vocab]
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    chunks = h[:n * dims_per_tok].reshape(n, dims_per_tok)
                    h_proj = chunks @ proj
                    logits = h_proj @ W_U.T
                    acc = (logits.argmax(dim=1) == target).float().mean().item()
                    best_acc = max(best_acc, acc)

        with torch.no_grad():
            chunks = h[:n * dims_per_tok].reshape(n, dims_per_tok)
            h_proj = chunks @ proj
            logits = h_proj @ W_U.T
            acc = (logits.argmax(dim=1) == target).float().mean().item()
            best_acc = max(best_acc, acc)

        print(f"  n={n} (dims/tok={dims_per_tok}): accuracy={best_acc*100:.1f}%")
        # Real info in h: d_model floats. Proj is shared (amortized across texts).
        results[f"subspace_n{n}"] = {
            "n_tokens": n, "dims_per_token": dims_per_tok,
            "accuracy": round(best_acc, 4),
            "h_params": d_model, "proj_params": dims_per_tok * d_model,
            "method": "subspace_partition"
        }

    return results


# ============================================================
# EXPERIMENT 6: One Transformer Layer as Decoder
# ============================================================

def exp_one_layer_decoder(model, tokenizer, d_model, vocab_size):
    """
    Optimize a SINGLE vector h of d_model dims.
    Decode by broadcasting h to n positions, passing through ONE frozen
    transformer layer + layer norm + unembedding.

    Key: the transformer layer is FROZEN (it's the decoder, not learned per-sample).
    Only h is optimized per-sample.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: One Transformer Layer Decoder (Single Vector)")
    print("="*60)

    if hasattr(model, 'transformer'):
        layers = model.transformer.h
        W_U = model.lm_head.weight.detach()
        ln_f = model.transformer.ln_f
        wpe = model.transformer.wpe
    else:
        layers = model.model.layers
        W_U = model.lm_head.weight.detach()
        ln_f = model.model.norm
        wpe = None

    n_layers_total = len(layers)
    results = {}
    tokens = get_long_text(tokenizer)

    for layer_idx in [0, n_layers_total // 2, n_layers_total - 1]:
        layer = layers[layer_idx]
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)

        for n_tokens in [10, 50, 100, 200, 500, 768, 1000]:
            n = min(n_tokens, len(tokens))
            target = torch.tensor(tokens[:n], device=DEVICE)

            # Single vector h, broadcast to n positions
            h = torch.randn(d_model, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([h], lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

            best_acc = 0.0
            for step in range(5000):
                optimizer.zero_grad()

                # Broadcast h to [1, n, d_model]
                h_seq = h.unsqueeze(0).unsqueeze(0).expand(1, n, d_model).clone()

                # Add position embeddings if available
                if wpe is not None:
                    pos_ids = torch.arange(n, device=DEVICE).unsqueeze(0)
                    h_seq = h_seq + wpe(pos_ids)

                # Pass through frozen layer
                with torch.no_grad():
                    layer_out = layer(h_seq)[0]

                # But we need gradients w.r.t. h, so we can't use no_grad
                # Redo with grad
                h_seq_grad = h.unsqueeze(0).unsqueeze(0).expand(1, n, d_model).clone()
                if wpe is not None:
                    pos_ids = torch.arange(n, device=DEVICE).unsqueeze(0)
                    h_seq_grad = h_seq_grad + wpe(pos_ids)

                layer_out = layer(h_seq_grad)[0]
                normed = ln_f(layer_out)
                logits = (normed @ W_U.T).squeeze(0)  # [n, vocab]

                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 1000 == 0:
                    with torch.no_grad():
                        acc = (logits.argmax(dim=1) == target).float().mean().item()
                        best_acc = max(best_acc, acc)

            # Final eval
            with torch.no_grad():
                h_seq = h.unsqueeze(0).unsqueeze(0).expand(1, n, d_model)
                if wpe is not None:
                    h_seq = h_seq + wpe(torch.arange(n, device=DEVICE).unsqueeze(0))
                layer_out = layer(h_seq)[0]
                normed = ln_f(layer_out)
                logits = (normed @ W_U.T).squeeze(0)
                acc = (logits.argmax(dim=1) == target).float().mean().item()
                best_acc = max(best_acc, acc)

            bits = best_acc * n * math.log2(vocab_size)
            print(f"  Layer {layer_idx}, n={n}: acc={best_acc*100:.1f}%, bits={bits:.0f}")
            results[f"L{layer_idx}_n{n}"] = {
                "layer": layer_idx, "n_tokens": n,
                "accuracy": round(best_acc, 4),
                "bits_stored": round(bits, 1),
                "bits_per_dim": round(bits / d_model, 2),
            }

        # Re-enable grad for layer params
        for p in layer.parameters():
            p.requires_grad_(True)

    return results


# ============================================================
# EXPERIMENT 7: Full model decoder (Kuratov-style, single vector)
# ============================================================

def exp_full_model_single_vec(model, tokenizer, d_model, vocab_size):
    """
    Kuratov-style: single trainable [mem] vector prepended to the sequence,
    full frozen model as decoder.
    """
    print("\n" + "="*60)
    print("EXPERIMENT: Full Model (Kuratov-style, Single Vector)")
    print("="*60)

    if hasattr(model, 'transformer'):
        wte = model.transformer.wte
        wpe = model.transformer.wpe
    else:
        raise NotImplementedError("Only GPT-2 supported for now")

    results = {}
    tokens = get_long_text(tokenizer)

    for n_tokens in [10, 50, 100, 200, 500, 768, 1000, 1500]:
        n = min(n_tokens, len(tokens))
        target = torch.tensor([tokens[:n]], device=DEVICE)

        # Single trainable vector
        mem = torch.randn(1, 1, d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([mem], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

        best_acc = 0.0
        for step in range(5000):
            optimizer.zero_grad()

            # Embed target tokens
            token_embeds = wte(target)  # [1, n, d]
            pos_ids = torch.arange(1, n + 1, device=DEVICE).unsqueeze(0)
            pos_embeds = wpe(pos_ids)
            mem_pos = wpe(torch.zeros(1, 1, dtype=torch.long, device=DEVICE))

            # Concat: [mem] + token embeddings
            full_input = torch.cat([mem + mem_pos, token_embeds + pos_embeds], dim=1)

            # Forward through full model
            outputs = model(inputs_embeds=full_input)
            # Position 0 (mem) predicts token 0, position 1 predicts token 1, etc.
            logits = outputs.logits[:, 0:n, :]  # [1, n, vocab]

            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    best_acc = max(best_acc, acc)
                    if step == 0 or step % 2000 == 0:
                        print(f"    step {step}: loss={loss.item():.3f} acc={acc*100:.1f}%")

        with torch.no_grad():
            full_input = torch.cat([mem + mem_pos, token_embeds + pos_embeds], dim=1)
            outputs = model(inputs_embeds=full_input)
            logits = outputs.logits[:, 0:n, :]
            preds = logits.argmax(dim=-1)
            acc = (preds == target).float().mean().item()
            best_acc = max(best_acc, acc)

        bits = best_acc * n * math.log2(vocab_size)
        print(f"  n={n}: acc={best_acc*100:.1f}%, tokens_correct={int(best_acc*n)}, bits={bits:.0f}")

        results[f"n{n}"] = {
            "n_tokens": n, "accuracy": round(best_acc, 4),
            "tokens_correct": int(best_acc * n),
            "bits_stored": round(bits, 1),
            "bits_per_dim": round(bits / d_model, 2),
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("HIDDEN STATE CAPACITY V2: FINDING LIMITS")
    print("="*70)
    print(f"Device: {DEVICE}")

    model, tokenizer, d_model, vocab_size, n_layers = load_model("gpt2")

    all_results = {
        "model": "gpt2",
        "d_model": d_model,
        "vocab_size": vocab_size,
        "n_layers": n_layers,
    }

    all_results["theoretical"] = exp_theoretical(d_model, vocab_size)
    all_results["ascii"] = exp_ascii(d_model)
    all_results["vocab"] = exp_vocab(d_model, vocab_size, tokenizer)
    all_results["vector_walk"] = exp_vector_walk(d_model, vocab_size, tokenizer)
    all_results["unembedding"] = exp_unembedding(model, tokenizer, d_model, vocab_size)
    all_results["one_layer"] = exp_one_layer_decoder(model, tokenizer, d_model, vocab_size)
    all_results["full_model"] = exp_full_model_single_vec(model, tokenizer, d_model, vocab_size)

    # Save
    out_path = RESULTS_DIR / "capacity_results_v2.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")
    return all_results


if __name__ == "__main__":
    results = main()
