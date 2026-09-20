from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def add_box(ax, x, y, w, h, text, fc="#eeeeee", ec="#666666",
            fontsize=10, weight="normal", rounded=False, lw=1.2):
    if rounded:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.01",
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
    else:
        patch = Rectangle(
            (x, y), w, h,
            facecolor=fc, edgecolor=ec, linewidth=lw
        )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight
    )
    return patch


def add_arrow(ax, x1, y1, x2, y2, lw=1.2, style="-|>", color="#444444",
              mutation_scale=12, linestyle="-"):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        linestyle=linestyle
    )
    ax.add_patch(arrow)
    return arrow


def add_dashed_connector(ax, x1, y1, x2, y2):
    add_arrow(
        ax, x1, y1, x2, y2,
        lw=1.4,
        style="-",
        color="#555555",
        linestyle=(0, (4, 3)),
        mutation_scale=10
    )


fig, ax = plt.subplots(figsize=(15.5, 7.7))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


# ---------------- Left panel ----------------
left_x, left_y, left_w, left_h = 0.03, 0.12, 0.30, 0.78
add_box(ax, left_x, left_y, left_w, left_h, "", fc="#f5f5f5", ec="#9a9a9a", lw=1.5)
ax.text(left_x + 0.008, left_y + left_h + 0.01, "Qwen2.5ForCausalLM",
        fontsize=12, ha="left", color="#444444")

model_x, model_y, model_w, model_h = 0.05, 0.18, 0.25, 0.66
add_box(ax, model_x, model_y, model_w, model_h, "", fc="#ededed", ec="#9a9a9a", lw=1.3)
ax.text(model_x + 0.008, model_y + model_h + 0.008, "Qwen2Model",
        fontsize=11, ha="left", color="#555555")

add_box(ax, 0.09, 0.77, 0.14, 0.06, "Embedding",
        fc="#cfc6df", ec="#8d7fa0", fontsize=12)

stack_x, stack_y, stack_w, stack_h = 0.09, 0.28, 0.16, 0.42
add_box(ax, stack_x, stack_y, stack_w, stack_h, "", fc="#e6e6e6", ec="#9a9a9a")

for i in range(6):
    dx = i * 0.010
    dy = i * 0.015
    bx = stack_x + 0.03 + dx
    by = stack_y + 0.03 + dy
    bw = 0.09
    bh = 0.18
    add_box(ax, bx, by, bw, bh, "", fc="#f0d98c", ec="#ad9a5a", lw=1.0)

    add_box(ax, bx + 0.015, by + 0.115, 0.06, 0.028, "Norm",
            fc="#f0b26b", ec="#a46a2a", fontsize=7)
    add_box(ax, bx + 0.015, by + 0.075, 0.06, 0.032, "Attention",
            fc="#d8e3f4", ec="#7f92af", fontsize=7)
    add_box(ax, bx + 0.015, by + 0.038, 0.06, 0.028, "Norm",
            fc="#f0b26b", ec="#a46a2a", fontsize=7)
    add_box(ax, bx + 0.015, by + 0.006, 0.06, 0.026, "MLP",
            fc="#e6d2e2", ec="#9a7f96", fontsize=7)

add_box(ax, 0.11, 0.20, 0.12, 0.055, "Final Norm",
        fc="#cfd9e8", ec="#7f8fa6", fontsize=12)

add_box(ax, 0.11, 0.11, 0.12, 0.055, "LM Head",
        fc="#ead8be", ec="#a48b68", fontsize=12)

# Highlight one layer for zoom
highlight_x, highlight_y, highlight_w, highlight_h = 0.18, 0.39, 0.095, 0.19
ax.add_patch(Rectangle(
    (highlight_x, highlight_y), highlight_w, highlight_h,
    fill=False, edgecolor="#666666", linewidth=1.6, linestyle=(0, (3, 2))
))


# ---------------- Middle panel ----------------
mid_x, mid_y, mid_w, mid_h = 0.40, 0.12, 0.23, 0.78
add_box(ax, mid_x, mid_y, mid_w, mid_h, "", fc="#f1da8a", ec="#b99b47", lw=1.5)

add_box(ax, 0.42, 0.80, 0.17, 0.08, "input_layernorm\nweight",
        fc="#e9ab61", ec="#a56a29", fontsize=11, weight="bold")

add_box(ax, 0.42, 0.53, 0.17, 0.22, "", fc="#cfdcf0", ec="#7f8fa6")
ax.text(0.43, 0.71, "self_attn", fontsize=12, fontweight="bold", ha="left")
for i, name in enumerate(["q_proj", "k_proj", "v_proj", "o_proj"]):
    y = 0.66 - i * 0.042
    ax.text(0.43, y, name, fontsize=11, ha="left", va="center")
    add_box(ax, 0.505, y - 0.015, 0.075, 0.03, "nn.Linear",
            fc="#e5e5e5", ec="#9a9a9a", fontsize=9)

add_box(ax, 0.42, 0.44, 0.17, 0.07, "post_attn_layernorm\nweight",
        fc="#e9ab61", ec="#a56a29", fontsize=11, weight="bold")

add_box(ax, 0.42, 0.18, 0.17, 0.22, "", fc="#ead7e6", ec="#9e8598")
ax.text(0.43, 0.35, "mlp", fontsize=12, fontweight="bold", ha="left")
for i, name in enumerate(["gate_proj", "up_proj", "down_proj"]):
    y = 0.305 - i * 0.05
    ax.text(0.43, y, name, fontsize=11, ha="left", va="center")
    add_box(ax, 0.505, y - 0.015, 0.075, 0.03, "nn.Linear",
            fc="#e5e5e5", ec="#9a9a9a", fontsize=9)

add_box(ax, 0.40, 0.07, 0.23, 0.05, "QwenDecoderLayer",
        fc="#ece1b7", ec="#c4b279", fontsize=13)

# ---------------- Right panel ----------------
right_x, right_y, right_w, right_h = 0.70, 0.12, 0.28, 0.78
add_box(ax, right_x, right_y, right_w, right_h, "", fc="#cfdcf0", ec="#7f8fa6", lw=1.5)

add_box(ax, 0.75, 0.83, 0.08, 0.06, "Linear Layers\nfrom self_attn / mlp",
        fc="#e8c2c0", ec="#b47f7a", fontsize=10, rounded=True, weight="bold")

add_box(ax, 0.75, 0.72, 0.08, 0.07, "Group of\nWeight Tensors",
        fc="#e6e6e6", ec="#9a9a9a", fontsize=10)

add_box(ax, 0.89, 0.72, 0.06, 0.06, "Group Scaling\nFactor",
        fc="#d5e8c6", ec="#92ad73", fontsize=10, rounded=True, weight="bold")

add_box(ax, 0.75, 0.58, 0.08, 0.06, "FP16\nBaseline",
        fc="#f7f7f7", ec="#9a9a9a", fontsize=10)

add_box(ax, 0.89, 0.58, 0.06, 0.06, "INT8\nbitsandbytes",
        fc="#f7f7f7", ec="#9a9a9a", fontsize=10)

add_box(ax, 0.82, 0.46, 0.06, 0.06, "4 bit NF4\nbitsandbytes",
        fc="#f7f7f7", ec="#9a9a9a", fontsize=10)

add_box(ax, 0.75, 0.33, 0.08, 0.06, "Bangla Tasks",
        fc="#e5e5e5", ec="#9a9a9a", fontsize=11, weight="bold")

add_box(ax, 0.71, 0.25, 0.07, 0.05, "Stance",
        fc="#ffffff", ec="#9a9a9a", fontsize=10)
add_box(ax, 0.81, 0.25, 0.07, 0.05, "NLI",
        fc="#ffffff", ec="#9a9a9a", fontsize=10)
add_box(ax, 0.91, 0.25, 0.07, 0.05, "Fake News",
        fc="#ffffff", ec="#9a9a9a", fontsize=10)

add_box(ax, 0.77, 0.13, 0.14, 0.08, "Evaluation Metrics\nMacro F1, Memory,\nLatency, Throughput",
        fc="#f1d2d8", ec="#b98895", fontsize=10, rounded=True, weight="bold")

add_box(ax, 0.70, 0.07, 0.28, 0.05, "Quantization and Evaluation Pipeline",
        fc="#bdd1e8", ec="#7f96af", fontsize=13, weight="bold")

# Arrows in right panel
add_arrow(ax, 0.79, 0.83, 0.79, 0.79)
add_arrow(ax, 0.79, 0.72, 0.92, 0.72)
add_arrow(ax, 0.79, 0.72, 0.79, 0.64)
add_arrow(ax, 0.92, 0.72, 0.92, 0.64)
add_arrow(ax, 0.79, 0.58, 0.82, 0.39)
add_arrow(ax, 0.92, 0.58, 0.85, 0.39)
add_arrow(ax, 0.85, 0.46, 0.79, 0.39)
add_arrow(ax, 0.79, 0.33, 0.745, 0.30)
add_arrow(ax, 0.79, 0.33, 0.845, 0.30)
add_arrow(ax, 0.79, 0.33, 0.945, 0.30)
add_arrow(ax, 0.745, 0.25, 0.81, 0.21)
add_arrow(ax, 0.845, 0.25, 0.84, 0.21)
add_arrow(ax, 0.945, 0.25, 0.87, 0.21)

# Dashed zoom connectors
add_dashed_connector(ax, highlight_x + highlight_w, highlight_y + highlight_h, mid_x, mid_y + mid_h)
add_dashed_connector(ax, highlight_x + highlight_w, highlight_y, mid_x, mid_y)

add_dashed_connector(ax, mid_x + mid_w, mid_y + mid_h, right_x, right_y + right_h)
add_dashed_connector(ax, mid_x + mid_w, mid_y, right_x, right_y)

# Caption
fig.text(
    0.5, 0.03,
    "Figure 1: Overview of the Qwen2.5 architecture, decoder layer components, and the quantization and evaluation pipeline used in this study.",
    ha="center",
    fontsize=17,
    fontfamily="serif"
)

png_path = OUT_DIR / "methodology_architecture_figure.png"
pdf_path = OUT_DIR / "methodology_architecture_figure.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

print("Saved:")
print(png_path)
print(pdf_path)