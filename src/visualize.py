"""Generate all plots for the hidden state capacity research."""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("/workspaces/hidden-state-storage-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Load results
with open(RESULTS_DIR / "capacity_results_v3_random.json") as f:
    v3 = json.load(f)

d_model = 768
vocab_size = 50257
bits_per_token = math.log2(vocab_size)

plt.rcParams.update({
    'font.size': 12, 'figure.figsize': (10, 6),
    'axes.grid': True, 'grid.alpha': 0.3,
})

# ============================================================
# PLOT 1: Capacity comparison across encoding schemes
# ============================================================

fig, ax = plt.subplots(figsize=(12, 7))

# Data: (scheme_name, max_tokens_at_100%, bits_stored, color)
schemes = [
    ("ASCII (fp32)\n3072 bytes", 3072, 3072*8, '#2196F3'),
    ("ASCII (fp16)\n1536 bytes", 1536, 1536*8, '#64B5F6'),
    ("Vocab ID (fp32)\n1536 tokens", 1536, 1536*bits_per_token, '#4CAF50'),
    ("Vocab ID (fp16)\n768 tokens", 768, 768*bits_per_token, '#81C784'),
    ("Vector Walk\n(1 dim/tok)\n768 tokens", 768, 768*bits_per_token, '#FF9800'),
    ("Subspace+Unembed\n(3 dims/tok)\n200 tokens", 200, 200*bits_per_token, '#9C27B0'),
    ("1 Layer Decoder\n(single vec)\n~10 tokens", 10, 10*bits_per_token, '#F44336'),
    ("Full Model\n(single vec)\n~50 tokens", 50, 50*0.74*bits_per_token, '#E91E63'),
    ("Full Model\n(32 mem vecs)\n~500 tokens", 500, 500*0.994*bits_per_token, '#795548'),
]

names = [s[0] for s in schemes]
bits = [s[2] for s in schemes]
colors = [s[3] for s in schemes]

bars = ax.bar(range(len(schemes)), bits, color=colors, edgecolor='black', linewidth=0.5)

# Add theoretical maximum line
theo_max = d_model * 32  # fp32
ax.axhline(y=theo_max, color='red', linestyle='--', linewidth=2, label=f'Theoretical max (fp32): {theo_max} bits')

ax.set_xticks(range(len(schemes)))
ax.set_xticklabels(names, rotation=0, ha='center', fontsize=9)
ax.set_ylabel('Information Stored (bits)')
ax.set_title('Hidden State Storage Capacity: Encoding Scheme Comparison\n(GPT-2, d_model=768, random tokens where applicable)')
ax.legend(fontsize=10)

# Add bit values on bars
for bar, b in zip(bars, bits):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
            f'{b:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "capacity_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved capacity_comparison.png")


# ============================================================
# PLOT 2: Accuracy vs sequence length for different decoders
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

# One layer decoder (last layer, random)
one_layer_data = v3.get("one_layer_random", {})
ol_n = []
ol_acc = []
for key, val in sorted(one_layer_data.items()):
    if "L11" in key:
        ol_n.append(val["n_tokens"])
        ol_acc.append(val["accuracy"] * 100)

ax.plot(ol_n, ol_acc, 'o-', color='#F44336', linewidth=2, markersize=6, label='1 Layer Decoder (Layer 11)')

# Full model (random)
fm_data = v3.get("full_model_random", {})
fm_n = []
fm_acc = []
for key, val in sorted(fm_data.items(), key=lambda x: x[1]["n_tokens"]):
    fm_n.append(val["n_tokens"])
    fm_acc.append(val["accuracy"] * 100)

ax.plot(fm_n, fm_acc, 's-', color='#E91E63', linewidth=2, markersize=6, label='Full Model (12 layers)')

# Subspace partition (random)
sub_data = v3.get("unembed_subspace_random", {})
sub_n = []
sub_acc = []
for key, val in sorted(sub_data.items(), key=lambda x: x[1]["n_tokens"]):
    sub_n.append(val["n_tokens"])
    sub_acc.append(val["accuracy"] * 100)

ax.plot(sub_n, sub_acc, '^-', color='#9C27B0', linewidth=2, markersize=6, label='Subspace + Unembed')

ax.set_xlabel('Number of Tokens to Store')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Token Recovery Accuracy vs Sequence Length\n(Single d=768 vector, Random Tokens)')
ax.legend(fontsize=10)
ax.set_xscale('log')
ax.set_ylim(-5, 105)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "accuracy_vs_length.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved accuracy_vs_length.png")


# ============================================================
# PLOT 3: Multi-mem vector scaling
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

mem_data = v3.get("multi_mem_scaling", {})
mem_n = []
mem_acc = []
mem_bits = []
mem_bpp = []

for key, val in sorted(mem_data.items(), key=lambda x: x[1]["n_mem_vectors"]):
    mem_n.append(val["n_mem_vectors"])
    mem_acc.append(val["accuracy"] * 100)
    mem_bits.append(val["bits_stored"])
    mem_bpp.append(val["bits_per_param"])

ax1.plot(mem_n, mem_acc, 'o-', color='#795548', linewidth=2, markersize=8)
ax1.set_xlabel('Number of [mem] Vectors')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Accuracy vs Number of Memory Vectors\n(500 random tokens, full GPT-2 decoder)')
ax1.set_xscale('log', base=2)

ax2.bar(range(len(mem_n)), mem_bpp, color='#795548', edgecolor='black', linewidth=0.5)
ax2.set_xticks(range(len(mem_n)))
ax2.set_xticklabels(mem_n)
ax2.set_xlabel('Number of [mem] Vectors')
ax2.set_ylabel('Bits per Parameter')
ax2.set_title('Storage Efficiency:\nBits per Learnable Parameter')

for i, (n, bpp) in enumerate(zip(mem_n, mem_bpp)):
    ax2.text(i, bpp + 0.02, f'{bpp:.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "multi_mem_scaling.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved multi_mem_scaling.png")


# ============================================================
# PLOT 4: Decoder complexity spectrum (summary figure)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

# For each decoder, what's the max # tokens stored at ≥99% accuracy?
# Also compute effective bits stored

decoders = [
    ("No Decoder\n(ASCII fp32)", 3072, 3072*8, 0, '#2196F3'),
    ("No Decoder\n(Vocab ID fp32)", 1536, 1536*bits_per_token, 0, '#4CAF50'),
    ("No Decoder\n(Magnitude/dim)", 768, 768*bits_per_token, 0, '#FF9800'),
    ("Shared Proj\n+ Unembed\n(3 dims/tok)", 200, 200*bits_per_token, 768*3, '#9C27B0'),
    ("1 Layer\n(broadcast vec)", 10, 10*bits_per_token, 0, '#F44336'),
    ("Full Model\n(1 mem vec)", 50, 578, 0, '#E91E63'),  # 74% of 50
    ("Full Model\n(16 mem vecs)", 362, 5653, 16*768, '#795548'),  # 72.4% of 500
    ("Full Model\n(32 mem vecs)", 497, 7762, 32*768, '#8D6E63'),  # 99.4% of 500
]

x = range(len(decoders))
names = [d[0] for d in decoders]
tokens = [d[1] for d in decoders]
bits_vals = [d[2] for d in decoders]
extra_params = [d[3] for d in decoders]
colors = [d[4] for d in decoders]

ax.bar(x, bits_vals, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(y=d_model*32, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Theoretical max fp32: {d_model*32} bits')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8, ha='center')
ax.set_ylabel('Bits Stored (Lossless or Near-Lossless)')
ax.set_title('Information Storage in a 768-dim Hidden State\nby Encoding/Decoding Scheme')
ax.legend()

for i, (b, t) in enumerate(zip(bits_vals, tokens)):
    ax.text(i, b + 300, f'{b:.0f} bits\n({t} tokens)', ha='center', fontsize=7, fontweight='bold')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "decoder_complexity_spectrum.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved decoder_complexity_spectrum.png")


# ============================================================
# PLOT 5: Bits per dimension efficiency
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

eff_data = [
    ("ASCII fp32", 3072*8 / 768),
    ("ASCII fp16", 1536*8 / 768),
    ("Vocab fp32", 1536*bits_per_token / 768),
    ("Vocab fp16", 768*bits_per_token / 768),
    ("Vec Walk (1d)", 768*bits_per_token / 768),
    ("Subspace+Unembed", 200*bits_per_token / 768),
    ("1 Layer (random)", 10*bits_per_token / 768),
    ("Full Model 1vec", 578 / 768),
    ("Full Model 16vec", 5653 / (16*768)),
    ("Full Model 32vec", 7762 / (32*768)),
]

names = [d[0] for d in eff_data]
bpd = [d[1] for d in eff_data]
colors_eff = ['#2196F3', '#64B5F6', '#4CAF50', '#81C784', '#FF9800',
              '#9C27B0', '#F44336', '#E91E63', '#795548', '#8D6E63']

ax.barh(range(len(eff_data)), bpd, color=colors_eff, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(eff_data)))
ax.set_yticklabels(names)
ax.set_xlabel('Bits per Dimension')
ax.set_title('Storage Efficiency: Bits per Hidden State Dimension')
ax.axvline(x=32, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='fp32 (32 bits/dim)')
ax.axvline(x=16, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='fp16 (16 bits/dim)')
ax.legend()

for i, b in enumerate(bpd):
    ax.text(b + 0.5, i, f'{b:.1f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "bits_per_dimension.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved bits_per_dimension.png")

print("\nAll plots generated!")
