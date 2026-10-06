"""Task 1: sketch of the four XOR points  ->  results/xor_points.png"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(4.4, 4.4))
pts = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
for (x1, x2), y in pts.items():
    ax.scatter(x1, x2, s=260, marker="o" if y == 1 else "s",
               c="#d95f02" if y == 1 else "#1b9e77", edgecolors="black", zorder=3)
    ax.annotate(f"({x1},{x2})  y={y}", (x1, x2), textcoords="offset points",
                xytext=(0, -24) if x2 == 0 else (0, 16), ha="center", fontsize=9)
ax.plot([-0.35, 0.85], [0.85, -0.35], "--", color="grey", lw=1)
ax.text(0.30, 0.52, "example line x1 + x2 = 0.5:\n(1,1) ends up on the same\n"
        "side as the class-1 points", fontsize=8, color="dimgrey")
ax.set_xlim(-0.4, 1.4)
ax.set_ylim(-0.4, 1.4)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_aspect("equal")
ax.set_title("XOR: class 1 (circles) vs class 0 (squares)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
fig.savefig(os.path.join(HERE, "results", "xor_points.png"), dpi=130)
