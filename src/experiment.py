"""
Naive random split vs subject level split, on the same synthetic HAR
dataset, using the same two classifiers. Saves numeric results and
confusion matrices to results/results.json and results/confusion.npz
so that make_figures.py can build the article figures from real,
reproducible output.

Run:
    python -m src.experiment
"""

import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.generate_data import ACTIVITIES, generate_har_dataset


def evaluate(X, y, groups, train_idx, test_idx, label):
    scaler = StandardScaler().fit(X[train_idx])
    X_train, X_test = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
    y_train, y_test = y[train_idx], y[test_idx]

    rf = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
    svm = SVC(kernel="rbf", C=2.0, random_state=42).fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    svm_pred = svm.predict(X_test)

    rf_acc = float((rf_pred == y_test).mean() * 100)
    svm_acc = float((svm_pred == y_test).mean() * 100)

    print(f"{label:22s} | RF: {rf_acc:5.1f}%  | SVM: {svm_acc:5.1f}%  | test windows: {len(test_idx)}")

    return {
        "rf_acc": rf_acc,
        "svm_acc": svm_acc,
        "rf_confusion": confusion_matrix(y_test, rf_pred, labels=range(len(ACTIVITIES))).tolist(),
        "svm_confusion": confusion_matrix(y_test, svm_pred, labels=range(len(ACTIVITIES))).tolist(),
    }


def main():
    X, y, groups = generate_har_dataset()
    print(f"Total windows: {len(X)}  |  Subjects: {len(set(groups))}  |  Classes: {len(ACTIVITIES)}")
    print()

    # Naive random split. Ignores subject identity. This is the bug.
    idx = np.arange(len(X))
    np.random.RandomState(7).shuffle(idx)
    split = int(len(idx) * 0.8)
    train_idx_naive, test_idx_naive = idx[:split], idx[split:]
    naive_results = evaluate(X, y, groups, train_idx_naive, test_idx_naive, "Naive random split")

    # Subject level split. This is the fix.
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx_grp, test_idx_grp = next(gss.split(X, y, groups))

    assert set(groups[train_idx_grp]).isdisjoint(set(groups[test_idx_grp])), "Leakage detected!"
    print(
        f"Zero overlap assertion passed: {len(set(groups[train_idx_grp]))} train subjects, "
        f"{len(set(groups[test_idx_grp]))} test subjects, no overlap."
    )

    grouped_results = evaluate(X, y, groups, train_idx_grp, test_idx_grp, "Subject level split")

    results = {"naive": naive_results, "grouped": grouped_results, "activities": ACTIVITIES}

    with open("results/results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save raw arrays for the embedding figure (figure 3)
    np.savez("results/dataset.npz", X=X, y=y, groups=groups)

    print()
    print("Saved results/results.json and results/dataset.npz")


if __name__ == "__main__":
    main()
