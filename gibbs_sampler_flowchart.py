import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.path import Path

fig, ax = plt.subplots(figsize=(10, 11), dpi=300)
ax.axis("off")

boxes = [
    (
        "Step 0: Initialization of Parameters\n(W_t^0, mu_t^0, U_{i,t}^0, V_{i,t}^0, Sigma_t^0, etc.)",
        0.88,
    ),
    (
        "Block 1: Dynamic Probit Network Layer (W_t)\nTruncated normal draws z_{ij,t}; Latent coordinates (u_i, v_j, u_{i,t}); Row-normalize W_t",
        0.72,
    ),
    (
        "Block 2: Dynamic Factor Trend (mu_t)\nState vector alpha_t = [lambda_t', delta_t']' via FFBS; Static factor loadings F",
        0.56,
    ),
    (
        "Block 3: Dynamic Low-Rank Lag Coefficients (B_{i,t})\nBilinear factors U_{i,t}, V_{i,t} via row-decoupled Woodbury FFBS",
        0.40,
    ),
    (
        "Block 4: Triangular Multivariate Stochastic Volatility\nCholesky entries a_j; Log-volatilities h_{j,t} via 7-component mixture FFBS",
        0.24,
    ),
    (
        "Block 5: Structural Identification & Retraction\nContractive spectral check rho(F_t) <= 1 - epsilon; SVD & sign anchors",
        0.08,
    ),
]

box_pad = 0.012
arrow_gap = 0.008
box_width = 0.72
box_height = 0.075
x_left = 0.08
x_mid = x_left + box_width / 2
visual_half_h = box_height / 2 + box_pad
visual_right = x_left + box_width + box_pad

for i, (text, y) in enumerate(boxes):
    rect = patches.FancyBboxPatch(
        (x_left, y - box_height / 2),
        box_width,
        box_height,
        boxstyle=f"round,pad={box_pad},rounding_size=0.015",
        facecolor="#F7F9FA",
        edgecolor="#1F2937",
        linewidth=1.4,
    )
    ax.add_patch(rect)

    parts = text.split("\n", 1)
    ax.text(
        x_mid,
        y + 0.015,
        parts[0],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    if len(parts) > 1:
        ax.text(
            x_mid,
            y - 0.02,
            parts[1],
            ha="center",
            va="center",
            fontsize=9,
            color="#374151",
        )

    if i < len(boxes) - 1:
        next_y = boxes[i + 1][1]
        y_start = y - visual_half_h - arrow_gap
        y_end = next_y + visual_half_h + arrow_gap
        ax.plot(
            [x_mid, x_mid],
            [y_start, y_end],
            color="#1F2937",
            lw=1.5,
            solid_capstyle="butt",
            zorder=2,
        )
        ax.annotate(
            "",
            xy=(x_mid, y_end),
            xytext=(x_mid, y_end + 0.01),
            arrowprops=dict(
                arrowstyle="->", color="#1F2937", lw=1.5, mutation_scale=12
            ),
        )

# Feedback loop stays fully to the right of the boxes.
loop_x = visual_right + 0.10
loop_gap = 0.025
block1_y = boxes[1][1]
block5_y = boxes[-1][1]
path = Path(
    [
        (visual_right + loop_gap, block5_y),
        (loop_x, block5_y),
        (loop_x, block1_y),
        (visual_right + loop_gap, block1_y),
    ],
    [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO],
)
ax.add_patch(
    patches.FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        color="#2563EB",
        linewidth=1.5,
        linestyle="--",
        mutation_scale=15,
        shrinkA=0,
        shrinkB=0,
    )
)
ax.text(
    loop_x + 0.045,
    0.40,
    "Iterate MCMC\nuntil convergence",
    ha="center",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#2563EB",
    rotation=-90,
)

ax.set_xlim(0.0, 1.08)
ax.set_ylim(0.0, 1.0)
plt.tight_layout()
plt.savefig("gibbs_sampler_flowchart.png", dpi=300, bbox_inches="tight")
print("Saved as gibbs_sampler_flowchart.png")
