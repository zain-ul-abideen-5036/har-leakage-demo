"""
Generates the article banner image (1600x840, suitable for Medium's header).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
import numpy as np

fig = plt.figure(figsize=(16, 8.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16); ax.set_ylim(0, 8.4)
ax.axis("off")

# Background gradient (dark navy to deep teal)
grad = np.linspace(0, 1, 256).reshape(1, -1)
grad = np.vstack([grad] * 2)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("bg", ["#0B132B", "#1B3A4B", "#12444A"])
ax.imshow(grad, extent=[0, 16, 0, 8.4], aspect="auto", cmap=cmap, zorder=0)

rng = np.random.RandomState(3)

# Left cluster: scattered, overlapping dots (representing leaked/naive split, chaotic)
leak_colors = ["#E76F51", "#F4A261"]
for _ in range(70):
    x = rng.uniform(0.8, 6.8)
    y = rng.uniform(1.2, 7.2)
    r = rng.uniform(0.05, 0.13)
    c = leak_colors[rng.randint(0, 2)]
    ax.add_patch(Circle((x, y), r, color=c, alpha=0.55, zorder=2))

# Right cluster: cleanly separated groups (representing the fix, orderly)
fix_colors = ["#2A9D8F", "#8ECAE6", "#94D2BD"]
centers = [(10.5, 6.0), (12.8, 4.2), (10.8, 2.0)]
for cx, cy in centers:
    color = fix_colors[rng.randint(0, len(fix_colors))]
    for _ in range(16):
        x = cx + rng.normal(0, 0.45)
        y = cy + rng.normal(0, 0.45)
        r = rng.uniform(0.07, 0.12)
        ax.add_patch(Circle((x, y), r, color=color, alpha=0.8, zorder=2))

# Dividing arrow / transformation cue
ax.annotate("", xy=(9.3, 4.2), xytext=(7.3, 4.2),
             arrowprops=dict(arrowstyle="-|>", color="#F4F4F4", lw=3, alpha=0.85))

# Title text
ax.text(0.6, 7.55, "THE DATA LEAKAGE BUG", fontsize=34, fontweight="bold",
         color="#FFFFFF", zorder=5, family="sans-serif")
ax.text(0.6, 6.75, "That Makes Your Model Look 30 Points Better Than It Is",
         fontsize=17, color="#D8E2E6", zorder=5, family="sans-serif")

ax.text(0.6, 1.0, "A Human Activity Recognition case study, with real code and real numbers",
         fontsize=13, color="#9FB8BE", style="italic", zorder=5)

plt.savefig("assets/banner.png", dpi=150, bbox_inches="tight",
            facecolor="#0B132B", pad_inches=0)
print("saved assets/banner.png")
