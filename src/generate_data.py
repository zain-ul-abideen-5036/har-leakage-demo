"""
Synthetic Human Activity Recognition (HAR) dataset generator.

Simulates wearable accelerometer style summary features for multiple
subjects performing multiple activities. Each subject carries an
idiosyncratic "movement signature" that is added on top of the true
activity signal. This mirrors real HAR data, where individual
differences in gait and movement style often rival or exceed the
activity class signal itself.

This structure is what makes subject level data leakage possible: if a
subject's windows appear in both train and test, a model can partly
learn to recognize the person rather than the activity.
"""

import numpy as np

ACTIVITIES = ["walking", "running", "sitting", "stairs"]


def generate_har_dataset(
    n_subjects: int = 30,
    windows_per_subject_per_class: int = 25,
    n_features: int = 8,
    class_signal_std: float = 1.0,
    subject_signal_std: float = 2.2,
    noise_std: float = 1.0,
    seed: int = 42,
):
    """Generate a synthetic HAR dataset with subject level structure.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)          activity class labels
    groups : np.ndarray, shape (n_samples,)     subject id per sample
    """
    rng = np.random.RandomState(seed)
    n_classes = len(ACTIVITIES)

    class_centers = rng.normal(0, class_signal_std, size=(n_classes, n_features))
    subject_signatures = rng.normal(0, subject_signal_std, size=(n_subjects, n_features))

    X, y, groups = [], [], []
    for subj in range(n_subjects):
        for cls in range(n_classes):
            for _ in range(windows_per_subject_per_class):
                noise = rng.normal(0, noise_std, size=n_features)
                X.append(class_centers[cls] + subject_signatures[subj] + noise)
                y.append(cls)
                groups.append(subj)

    return np.array(X), np.array(y), np.array(groups)
