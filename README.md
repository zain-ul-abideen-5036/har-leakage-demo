<div align="center">

# HAR Subject-Level Data Leakage Demo

**A fully reproducible demonstration of the data leakage bug that makes wearable-sensor activity classifiers look 30 accuracy points better than they actually are.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)

[Read the full article on Medium](https://medium.com/@zainulabideen5/the-data-leakage-bug-that-makes-your-model-look-30-points-better-than-it-is-0160407cdaee) &nbsp;·&nbsp; [Quick start](#quick-start) &nbsp;·&nbsp; [Project structure](#project-structure) &nbsp;·&nbsp; [The core fix](#the-core-fix-isolated)

</div>

---

## The result, in one table

Same synthetic dataset. Same two models. One line of splitting code changed.

| Model | Naive random split | Subject-level split | Drop |
|---|---:|---:|---:|
| Random Forest | 78.3% | 47.5% | −30.8 pts |
| SVM | 77.2% | 54.0% | −23.2 pts |

The lower numbers are the correct ones. The higher numbers are what you get when your model partly learns to recognize *people* instead of *activities*.

> **This repository is the companion code for the Medium article** *"The Data Leakage Bug That Makes Your Model Look 30 Points Better Than It Is."* The article explains the reasoning and the figures in depth; this repo is where you go to run it yourself, verify the numbers, and adapt it to your own data. **[Read the article →](https://medium.com/@zainulabideen5/the-data-leakage-bug-that-makes-your-model-look-30-points-better-than-it-is-0160407cdaee)**

---

## Why this matters

Wearable sensor datasets are not collections of independent samples. They're a small number of people, each contributing many correlated samples, and every person moves in their own idiosyncratic way. A random train/test shuffle has no concept of "person," so it happily splits one individual's samples across both sides of the line, handing a model the answer key before it's ever evaluated.

This repository builds a synthetic dataset with that exact structure on purpose, so the mechanism can be measured directly instead of argued about in the abstract, and shows the fix: a single group-aware split.

---

## Quick start

```bash
git clone https://github.com/zain-ul-abideen-5036/har-leakage-demo.git
cd har-leakage-demo
pip install -r requirements.txt

# Step 1 — run the experiment (naive split vs. subject-level split)
python -m src.experiment

# Step 2 — generate every figure from the real saved results
python -m src.make_figures

# Optional — regenerate the banner image
python -m src.make_banner
```

`src/experiment.py` prints both accuracy numbers to the console and writes `results/results.json` and `results/dataset.npz`. `src/make_figures.py` reads those files and produces every figure used in the article, so the figures are never separated from the numbers that generated them.

Want to confirm this isn't a cherry-picked result? Change `seed` in `src/generate_data.py` and rerun both steps. The pattern holds across seeds, because it's structural, not incidental.

---

## Project structure

```
har-leakage-demo/
├── article.md                  Full write-up (companion to the Medium article)
├── src/
│   ├── generate_data.py        Synthetic HAR dataset generator
│   ├── experiment.py           Naive split vs. subject-level split experiment
│   ├── make_figures.py         Builds all figures from real saved results
│   └── make_banner.py          Builds the article banner image
├── figures/                    Generated figures (from make_figures.py)
├── results/                    Generated results.json and dataset.npz
├── assets/                     Generated banner image
├── requirements.txt
└── LICENSE
```

---

## The core fix, isolated

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "Leakage detected!"
```

One import, one extra argument passed to the split, and one assertion that turns a silent assumption into a loud failure the moment it's ever violated.

---

## Read more

The full reasoning, including a PCA visualization of why the subject signal dominates the activity signal, and a confusion-matrix breakdown of where the errors actually land once leakage is removed, is in [`article.md`](article.md) and the published version on Medium.

## License

MIT. Use this however is useful to you. See [LICENSE](LICENSE).
