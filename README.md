# HAR Subject Level Data Leakage Demo

A small, fully reproducible demonstration of subject level data leakage in Human Activity Recognition (HAR), the bug where a naive random train/test split lets a model partially "recognize the person" instead of learning the activity itself.

This repository contains the exact code behind the article *The Data Leakage Bug That Makes Your Model Look 30 Points Better Than It Is*.

## The result in one line

Same synthetic dataset, same two models, one line of splitting code changed:

| Model | Naive random split | Subject level split |
|---|---|---|
| Random Forest | 78.3% | 47.5% |
| SVM | 77.2% | 54.0% |

The lower numbers are the correct ones.

## Why this matters

Wearable sensor datasets are not collections of independent samples. They are a small number of people, each contributing many correlated samples. A random shuffle ignores that structure, and a model trained on the resulting split can partly succeed by recognizing individuals rather than activities. This repository builds a synthetic dataset with that exact structure, deliberately, so the mechanism can be isolated and measured directly rather than argued about in the abstract.

## Project structure

```
har-leakage-demo/
├── article.md                  Full write-up
├── src/
│   ├── generate_data.py        Synthetic HAR dataset generator
│   ├── experiment.py           Naive split vs subject level split experiment
│   ├── make_figures.py         Builds all figures from real saved results
│   └── make_banner.py          Builds the article banner image
├── figures/                    Generated figures (created by make_figures.py)
├── results/                    Generated results.json and dataset.npz
├── assets/                     Generated banner image
└── requirements.txt
```

## Running it yourself

```bash
git clone https://github.com/<your-username>/har-leakage-demo.git
cd har-leakage-demo
pip install -r requirements.txt

# Step 1: run the experiment (naive split vs subject level split)
python -m src.experiment

# Step 2: generate every figure from the real saved results
python -m src.make_figures

# Optional: regenerate the banner image
python -m src.make_banner
```

Running `src/experiment.py` prints both accuracy numbers to the console and writes `results/results.json` and `results/dataset.npz`. `src/make_figures.py` reads those files and produces every figure used in the article, so the figures are never separated from the numbers that generated them.

Change the random seed in `src/generate_data.py` to confirm the pattern isn't a cherry picked result. It holds across seeds, because it's structural, not incidental.

## The core fix, isolated

```python
from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

assert set(groups[train_idx]).isdisjoint(set(groups[test_idx])), "Leakage detected!"
```

One import, one extra argument passed to the split, and one assertion that turns a silent assumption into a loud failure if it's ever violated.

## License

MIT. Use this however is useful to you.
