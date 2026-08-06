"""
Builds every figure used in the article, from real saved results.

Run this after src/experiment.py has produced results/results.json and
results/dataset.npz.

    python -m src.make_figures
"""

import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch
from sklearn.decomposition import PCA

from src.generate_data import ACTIVITIES

PALETTE = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261", "#8E44AD", "#E9C46A"]
LEAK_COLOR = "#E76F51"
FIX_COLOR = "#2A9D8F"


def figure1_split_diagram(out_path="figures/figure1_split_diagram.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    participant_labels = ["P1", "P2", "P3", "P4", "P5", "P6"]

    def draw_box(ax, x, y, w, h, label):
        box = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=2, edgecolor="#333333", facecolor="#F7F7F7",
        )
        ax.add_patch(box)
        ax.text(x + w / 2, y + h + 0.12, label, ha="center", va="bottom",
                 fontsize=14, fontweight="bold", color="#333333")

    def scatter_dots(ax, x, y, w, h, dot_indices, seed):
        rng = np.random.RandomState(seed)
        xs = rng.uniform(x + 0.15, x + w - 0.15, len(dot_indices))
        ys = rng.uniform(y + 0.15, y + h - 0.15, len(dot_indices))
        for xi, yi, idx in zip(xs, ys, dot_indices):
            ax.add_patch(Circle((xi, yi), 0.09, facecolor=PALETTE[idx],
                                  edgecolor="white", linewidth=1.2, zorder=3))

    axA = axes[0]
    axA.set_xlim(0, 6); axA.set_ylim(0, 5); axA.axis("off")
    axA.set_title("A) Naive Random Split", fontsize=15, fontweight="bold", color="#E63946", pad=18)
    draw_box(axA, 0.3, 1, 2.4, 3, "Train")
    draw_box(axA, 3.3, 1, 2.4, 3, "Test")
    rng_master = np.random.RandomState(1)
    train_idx = rng_master.choice(range(6), size=14, replace=True)
    test_idx = rng_master.choice(range(6), size=8, replace=True)
    scatter_dots(axA, 0.3, 1, 2.4, 3, train_idx, seed=10)
    scatter_dots(axA, 3.3, 1, 2.4, 3, test_idx, seed=20)
    axA.text(3, 0.5, "Same participants appear in both sets", ha="center",
              fontsize=11, style="italic", color="#E63946")

    axB = axes[1]
    axB.set_xlim(0, 6); axB.set_ylim(0, 5); axB.axis("off")
    axB.set_title("B) Subject Level Split", fontsize=15, fontweight="bold", color="#2A9D8F", pad=18)
    draw_box(axB, 0.3, 1, 2.4, 3, "Train")
    draw_box(axB, 3.3, 1, 2.4, 3, "Test")
    train_idx_b = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    test_idx_b = np.array([4, 4, 4, 4, 5, 5, 5, 5])
    scatter_dots(axB, 0.3, 1, 2.4, 3, train_idx_b, seed=30)
    scatter_dots(axB, 3.3, 1, 2.4, 3, test_idx_b, seed=40)
    axB.text(3, 0.5, "Zero participant overlap between sets", ha="center",
              fontsize=11, style="italic", color="#2A9D8F")

    handles = [mpatches.Patch(color=PALETTE[i], label=participant_labels[i]) for i in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, title="Each color represents one participant")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def figure2_accuracy_comparison(results, out_path="figures/figure2_accuracy_comparison.png"):
    naive = {"Random Forest": results["naive"]["rf_acc"], "SVM": results["naive"]["svm_acc"]}
    grouped = {"Random Forest": results["grouped"]["rf_acc"], "SVM": results["grouped"]["svm_acc"]}

    models = ["Random Forest", "SVM"]
    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 6))
    bars1 = ax.bar(x - width / 2, [naive[m] for m in models], width,
                    label="Naive Split (leaked)", color=LEAK_COLOR, edgecolor="white")
    bars2 = ax.bar(x + width / 2, [grouped[m] for m in models], width,
                    label="Subject Level Split (correct)", color=FIX_COLOR, edgecolor="white")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h),
                         xytext=(0, 5), textcoords="offset points",
                         ha="center", fontsize=12, fontweight="bold", color="#333333")

    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13, fontweight="bold")
    ax.set_ylim(0, 100)
    ax.set_title("The Gap Is the Size of the Problem the Naive Split Was Hiding",
                  fontsize=13, color="#333333", pad=16)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def figure3_embedding(dataset, out_path="figures/figure3_subject_vs_activity_embedding.png"):
    X, y, groups = dataset["X"], dataset["y"], dataset["groups"]

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Highlight a clean subset of subjects distinctly; fade the rest to
    # light gray so the clustering pattern reads clearly rather than
    # turning into visual noise with 30 separate colors.
    axA = axes[0]
    highlighted_subjects = list(range(8))
    highlight_colors = plt.cm.tab10(np.linspace(0, 1, len(highlighted_subjects)))

    other_mask = ~np.isin(groups, highlighted_subjects)
    axA.scatter(Z[other_mask, 0], Z[other_mask, 1], s=8, color="#DDDDDD", alpha=0.5, zorder=1)

    for i, s in enumerate(highlighted_subjects):
        mask = groups == s
        axA.scatter(Z[mask, 0], Z[mask, 1], s=16, color=highlight_colors[i],
                     alpha=0.85, zorder=2, label=f"Subject {s}")

    axA.set_title("Colored by Subject Identity", fontsize=14, fontweight="bold", color="#E63946")
    axA.set_xlabel("PCA Component 1"); axA.set_ylabel("PCA Component 2")
    axA.legend(loc="upper right", frameon=False, fontsize=8, ncol=2, title="8 of 30 subjects shown")
    axA.text(0.02, 0.02, "Each subject forms its own tight cluster:\nthe model can \u201crecognize the person\u201d",
              transform=axA.transAxes, fontsize=10, va="bottom", color="#E63946", style="italic")

    axB = axes[1]
    cmap_class = ["#457B9D", "#F4A261", "#2A9D8F", "#8E44AD"]
    for c in range(len(ACTIVITIES)):
        mask = y == c
        axB.scatter(Z[mask, 0], Z[mask, 1], s=10, color=cmap_class[c], alpha=0.55, label=ACTIVITIES[c])
    axB.set_title("Colored by Activity Class", fontsize=14, fontweight="bold", color="#2A9D8F")
    axB.set_xlabel("PCA Component 1"); axB.set_ylabel("PCA Component 2")
    axB.legend(loc="upper right", frameon=False, fontsize=9)
    axB.text(0.02, 0.02, "Activity classes overlap heavily:\nthis is the signal we actually want",
              transform=axB.transAxes, fontsize=10, va="bottom", color="#2A9D8F", style="italic")

    fig.suptitle("The Subject Signal Is Stronger Than the Activity Signal",
                  fontsize=13, color="#333333", y=1.02)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def figure4_confusion_matrices(results, out_path="figures/figure4_confusion_matrices.png"):
    naive_cm = np.array(results["naive"]["rf_confusion"])
    grouped_cm = np.array(results["grouped"]["rf_confusion"])

    def normalize(cm):
        return cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax, cm, title, cmap in [
        (axes[0], normalize(naive_cm), "Naive Split (Random Forest)", "Reds"),
        (axes[1], normalize(grouped_cm), "Subject Level Split (Random Forest)", "Greens"),
    ]:
        im = ax.imshow(cm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(ACTIVITIES))); ax.set_xticklabels(ACTIVITIES, rotation=30, ha="right")
        ax.set_yticks(range(len(ACTIVITIES))); ax.set_yticklabels(ACTIVITIES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=12, fontweight="bold")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > 0.5 else "#333333"
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color=color, fontsize=10)

    fig.suptitle("Where the Errors Actually Land Once Leakage Is Removed",
                  fontsize=13, color="#333333", y=1.03)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    with open("results/results.json") as f:
        results = json.load(f)
    dataset = np.load("results/dataset.npz")

    figure1_split_diagram()
    figure2_accuracy_comparison(results)
    figure3_embedding(dataset)
    figure4_confusion_matrices(results)


if __name__ == "__main__":
    main()
