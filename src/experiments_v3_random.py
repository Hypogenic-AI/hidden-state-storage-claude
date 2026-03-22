"""
V3: Random token experiments to measure TRUE storage capacity
============================================================
Using random token sequences eliminates the model's language modeling advantage,
isolating the pure information storage capacity of the hidden state.
"""

import json
import math
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


def load_model(model_name="gpt2"):
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, output_hidden_states=True
    ).to(DEVICE)
    model.eval()
    d = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    V = model.config.vocab_size
    L = model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers
    max_pos = model.config.n_positions if hasattr(model.config, 'n_positions') else 1024
    print(f"  d={d}, V={V}, L={L}, max_pos={max_pos}")
    return model, tokenizer, d, V, L, max_pos


def random_tokens(n, vocab_size, seed=42):
    """Generate random token IDs uniformly from vocabulary."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, vocab_size, size=n).tolist()


# ============================================================
# Full Model (Kuratov-style) with RANDOM tokens
# ============================================================

def exp_full_model_random(model, tokenizer, d_model, vocab_size, max_pos):
    """Full model decoder with random tokens — no LM advantage."""
    print("\n" + "="*60)
    print("FULL MODEL + RANDOM TOKENS")
    print("="*60)

    wte = model.transformer.wte
    wpe = model.transformer.wpe
    max_seq = max_pos - 1  # Reserve 1 position for mem token

    results = {}

    for n_tokens in [10, 50, 100, 200, 300, 500, 700, 900, 1023]:
        n = min(n_tokens, max_seq)
        tokens = random_tokens(n, vocab_size, seed=SEED)
        target = torch.tensor([tokens], device=DEVICE)

        mem = torch.randn(1, 1, d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([mem], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

        best_acc = 0.0
        for step in range(5000):
            optimizer.zero_grad()
            token_embeds = wte(target)
            pos_ids = torch.arange(1, n + 1, device=DEVICE).unsqueeze(0)
            pos_embeds = wpe(pos_ids)
            mem_pos = wpe(torch.zeros(1, 1, dtype=torch.long, device=DEVICE))
            full_input = torch.cat([mem + mem_pos, token_embeds + pos_embeds], dim=1)

            outputs = model(inputs_embeds=full_input)
            logits = outputs.logits[:, 0:n, :]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    best_acc = max(best_acc, acc)

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
# One Layer decoder with RANDOM tokens
# ============================================================

def exp_one_layer_random(model, tokenizer, d_model, vocab_size, max_pos):
    """Single layer decoder with random tokens."""
    print("\n" + "="*60)
    print("ONE LAYER DECODER + RANDOM TOKENS")
    print("="*60)

    layers = model.transformer.h
    W_U = model.lm_head.weight.detach()
    ln_f = model.transformer.ln_f
    wpe = model.transformer.wpe
    n_layers_total = len(layers)

    results = {}

    for layer_idx in [0, n_layers_total // 2, n_layers_total - 1]:
        layer = layers[layer_idx]
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)

        for n_tokens in [10, 50, 100, 200, 500, 768]:
            n = min(n_tokens, max_pos)
            tokens = random_tokens(n, vocab_size, seed=SEED)
            target = torch.tensor(tokens, device=DEVICE)

            h = torch.randn(d_model, device=DEVICE, requires_grad=True)
            optimizer = torch.optim.Adam([h], lr=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

            best_acc = 0.0
            for step in range(5000):
                optimizer.zero_grad()
                h_seq = h.unsqueeze(0).unsqueeze(0).expand(1, n, d_model).clone()
                pos_ids = torch.arange(n, device=DEVICE).unsqueeze(0)
                h_seq = h_seq + wpe(pos_ids)

                layer_out = layer(h_seq)[0]
                normed = ln_f(layer_out)
                logits = (normed @ W_U.T).squeeze(0)
                loss = F.cross_entropy(logits, target)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % 1000 == 0:
                    with torch.no_grad():
                        acc = (logits.argmax(dim=1) == target).float().mean().item()
                        best_acc = max(best_acc, acc)

            with torch.no_grad():
                h_seq = h.unsqueeze(0).unsqueeze(0).expand(1, n, d_model)
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
            }

        for p in layer.parameters():
            p.requires_grad_(True)

    return results


# ============================================================
# Unembedding subspace partition with RANDOM tokens
# ============================================================

def exp_unembed_subspace_random(model, d_model, vocab_size):
    """Subspace partition with random tokens — true capacity test."""
    print("\n" + "="*60)
    print("UNEMBEDDING SUBSPACE PARTITION + RANDOM TOKENS")
    print("="*60)

    W_U = model.lm_head.weight.detach()
    results = {}

    for n_tokens in [1, 5, 10, 20, 50, 100, 200, 384, 500, 768]:
        dims_per_tok = d_model // n_tokens
        if dims_per_tok < 1:
            continue

        n = n_tokens
        tokens = random_tokens(n, vocab_size, seed=SEED)
        target = torch.tensor(tokens, device=DEVICE)

        h = torch.randn(d_model, device=DEVICE, requires_grad=True)
        proj = torch.randn(dims_per_tok, d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([h, proj], lr=0.05)

        best_acc = 0.0
        for step in range(5000):
            optimizer.zero_grad()
            chunks = h[:n * dims_per_tok].reshape(n, dims_per_tok)
            h_proj = chunks @ proj
            logits = h_proj @ W_U.T
            loss = F.cross_entropy(logits, target)
            loss.backward()
            optimizer.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    acc = (logits.argmax(dim=1) == target).float().mean().item()
                    best_acc = max(best_acc, acc)

        with torch.no_grad():
            chunks = h[:n * dims_per_tok].reshape(n, dims_per_tok)
            h_proj = chunks @ proj
            logits = h_proj @ W_U.T
            acc = (logits.argmax(dim=1) == target).float().mean().item()
            best_acc = max(best_acc, acc)

        bits = best_acc * n * math.log2(vocab_size)
        print(f"  n={n} (dims/tok={dims_per_tok}): acc={best_acc*100:.1f}%, bits={bits:.0f}")

        results[f"n{n}"] = {
            "n_tokens": n, "dims_per_token": dims_per_tok,
            "accuracy": round(best_acc, 4), "bits_stored": round(bits, 1),
        }

    return results


# ============================================================
# Multiple mem vectors (scaling analysis)
# ============================================================

def exp_multi_mem_vectors(model, tokenizer, d_model, vocab_size, max_pos):
    """Test how capacity scales with number of [mem] vectors."""
    print("\n" + "="*60)
    print("SCALING: Multiple [mem] Vectors + Full Model")
    print("="*60)

    wte = model.transformer.wte
    wpe = model.transformer.wpe

    results = {}
    n_target = 500  # Fixed target length

    tokens = random_tokens(n_target, vocab_size, seed=SEED)
    target = torch.tensor([tokens], device=DEVICE)

    for n_mem in [1, 2, 4, 8, 16, 32]:
        n = min(n_target, max_pos - n_mem)
        target_trunc = target[:, :n]

        mem = torch.randn(1, n_mem, d_model, device=DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([mem], lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

        best_acc = 0.0
        for step in range(5000):
            optimizer.zero_grad()
            token_embeds = wte(target_trunc)
            pos_ids = torch.arange(n_mem, n_mem + n, device=DEVICE).unsqueeze(0)
            pos_embeds = wpe(pos_ids)
            mem_pos = wpe(torch.arange(n_mem, device=DEVICE).unsqueeze(0))
            full_input = torch.cat([mem + mem_pos, token_embeds + pos_embeds], dim=1)

            outputs = model(inputs_embeds=full_input)
            logits = outputs.logits[:, n_mem-1:n_mem+n-1, :]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_trunc.reshape(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 1000 == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target_trunc).float().mean().item()
                    best_acc = max(best_acc, acc)

        with torch.no_grad():
            full_input = torch.cat([mem + mem_pos, token_embeds + pos_embeds], dim=1)
            outputs = model(inputs_embeds=full_input)
            logits = outputs.logits[:, n_mem-1:n_mem+n-1, :]
            preds = logits.argmax(dim=-1)
            acc = (preds == target_trunc).float().mean().item()
            best_acc = max(best_acc, acc)

        total_params = n_mem * d_model
        bits = best_acc * n * math.log2(vocab_size)
        print(f"  n_mem={n_mem}: acc={best_acc*100:.1f}%, bits={bits:.0f}, bits/param={bits/total_params:.2f}")

        results[f"mem{n_mem}"] = {
            "n_mem_vectors": n_mem, "n_tokens": n,
            "total_params": total_params,
            "accuracy": round(best_acc, 4),
            "bits_stored": round(bits, 1),
            "bits_per_param": round(bits / total_params, 2),
        }

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("="*70)
    print("V3: RANDOM TOKEN CAPACITY EXPERIMENTS")
    print("="*70)

    model, tokenizer, d_model, vocab_size, n_layers, max_pos = load_model("gpt2")

    all_results = {
        "model": "gpt2", "d_model": d_model,
        "vocab_size": vocab_size, "max_pos": max_pos,
    }

    all_results["unembed_subspace_random"] = exp_unembed_subspace_random(model, d_model, vocab_size)
    all_results["one_layer_random"] = exp_one_layer_random(model, tokenizer, d_model, vocab_size, max_pos)
    all_results["full_model_random"] = exp_full_model_random(model, tokenizer, d_model, vocab_size, max_pos)
    all_results["multi_mem_scaling"] = exp_multi_mem_vectors(model, tokenizer, d_model, vocab_size, max_pos)

    out = RESULTS_DIR / "capacity_results_v3_random.json"
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {out}")

    return all_results


if __name__ == "__main__":
    main()
