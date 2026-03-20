"""
Experimental pipeline for WEASEL 2.0 improvements.

Evaluates baseline WEASEL 2.0 and proposed variants on UCR datasets.
Records accuracy, fit time, predict time, peak memory, and feature count.

Usage:
    python run_experiments.py --datasets subset --seed 1379 --n_jobs 4
    python run_experiments.py --datasets all   --seed 1379 --n_jobs 4

Structure:
    1. CONFIGURATION  -- dataset lists, classifier definitions
    2. MEASUREMENT    -- timing, memory, accuracy for one (clf, dataset) pair
    3. PERSISTENCE    -- saving/loading individual results for resumability
    4. MAIN LOOP      -- iterate over all combinations
"""

import argparse
import os
import time
import tracemalloc
import warnings

import numpy as np
import pandas as pd


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# 30 datasets spanning different domains, sizes, and series lengths.
# Use for development and fast iteration before committing to the full run.
SUBSET_DATASETS = [
    "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car",
    "CBF", "Chinatown", "Coffee", "Computers", "CricketX",
    "DiatomSizeReduction", "ECG200", "ECG5000", "ECGFiveDays",
    "FaceFour", "Fish", "GunPoint", "Ham", "Herring",
    "ItalyPowerDemand", "Meat", "MoteStrain", "OliveOil",
    "Plane", "ShapeletSim", "SonyAIBORobotSurface1",
    "Strawberry", "ToeSegmentation1", "Trace", "Wafer",
]


def get_dataset_names(mode: str) -> list:
    """
    Return the list of UCR dataset names to evaluate.

    Parameters
    ----------
    mode : str
        "subset" for 30 representative datasets,
        "all" for the full UCR univariate archive.
    """
    if mode == "subset":
        return SUBSET_DATASETS

    from aeon.datasets.tsc_datasets import univariate_equal_length
    return sorted(list(univariate_equal_length))


def make_classifier(name: str, seed: int, n_jobs: int):
    """
    Construct a fresh, unfitted classifier instance by name.

    Called once per (classifier, dataset) evaluation so that
    no fitted state leaks between datasets.

    Parameters
    ----------
    name : str
        One of the keys in CLASSIFIER_NAMES.
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel threads for SFA transforms.

    Returns
    -------
    classifier
        An unfitted classifier instance.
    """
    from weasel.classification.dictionary_based import WEASEL_V2
    from weasel_variants import (
        WEASEL_V2_TFIDF,
        WEASEL_V2_SublinearTF,
        WEASEL_V2_SGD,
        WEASEL_V2_NoDiff,
    )

    registry = {
        "WEASEL2_baseline":     lambda: WEASEL_V2(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_tfidf":        lambda: WEASEL_V2_TFIDF(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_sublinear_tf": lambda: WEASEL_V2_SublinearTF(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_sgd":          lambda: WEASEL_V2_SGD(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_no_diff":      lambda: WEASEL_V2_NoDiff(random_state=seed, n_jobs=n_jobs),
    }

    if name not in registry:
        raise ValueError(
            f"Unknown classifier '{name}'. "
            f"Available: {list(registry.keys())}"
        )

    return registry[name]()


# The order here determines iteration order and table column order.
CLASSIFIER_NAMES = [
    "WEASEL2_baseline",
    "WEASEL2_tfidf",
    "WEASEL2_sublinear_tf",
    "WEASEL2_sgd",
    "WEASEL2_no_diff",
]


# ============================================================================
# 2. MEASUREMENT
# ============================================================================

def load_dataset(dataset_name: str):
    """
    Load a single UCR dataset using aeon's loader.

    Returns
    -------
    X_train, y_train, X_test, y_test

    Raises
    ------
    RuntimeError
        If the dataset cannot be loaded.
    """
    from aeon.datasets import load_classification

    try:
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
    except Exception as e:
        raise RuntimeError(f"Could not load dataset '{dataset_name}': {e}")

    return X_train, y_train, X_test, y_test


def measure_fit(clf, X_train, y_train) -> dict:
    """
    Fit the classifier. Measure wall-clock time and peak memory.

    Returns
    -------
    dict with keys: fit_time_s, peak_fit_mem_mb
    """
    tracemalloc.start()
    t0 = time.perf_counter()

    clf.fit(X_train, y_train)

    fit_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "fit_time_s": round(fit_time, 4),
        "peak_fit_mem_mb": round(peak_mem / (1024 * 1024), 2),
    }


def measure_predict(clf, X_test):
    """
    Predict on test data. Measure wall-clock time.

    Returns
    -------
    y_pred : np.ndarray
    predict_time_s : float
    """
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    predict_time = time.perf_counter() - t0

    return y_pred, round(predict_time, 4)


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    return round(float(np.mean(y_pred == y_true)), 6)


def get_feature_count(clf) -> int:
    """Extract feature count from a fitted classifier, or -1 if unavailable."""
    if hasattr(clf, "total_features_count"):
        return int(clf.total_features_count)
    return -1


def evaluate(clf, dataset_name: str) -> dict:
    """
    Run one complete evaluation: load data, fit, predict, record metrics.

    Parameters
    ----------
    clf : classifier
        An unfitted classifier instance.
    dataset_name : str
        Name of the UCR dataset.

    Returns
    -------
    dict
        All recorded metrics for this (classifier, dataset) pair.
    """
    X_train, y_train, X_test, y_test = load_dataset(dataset_name)

    # Dataset metadata
    record = {
        "dataset": dataset_name,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "ts_length": X_train.shape[-1],
        "n_classes": len(np.unique(y_train)),
    }

    # Fit
    fit_metrics = measure_fit(clf, X_train, y_train)
    record.update(fit_metrics)

    # Feature count
    record["n_features"] = get_feature_count(clf)

    # Predict
    y_pred, predict_time = measure_predict(clf, X_test)
    record["predict_time_s"] = predict_time

    # Accuracy
    record["accuracy"] = compute_accuracy(y_test, y_pred)

    return record


# ============================================================================
# 3. PERSISTENCE
# ============================================================================

def result_path(results_dir: str, clf_name: str, dataset_name: str) -> str:
    """Path to the individual result CSV for one (classifier, dataset) pair."""
    return os.path.join(results_dir, f"{clf_name}__{dataset_name}.csv")


def result_exists(results_dir: str, clf_name: str, dataset_name: str) -> bool:
    """Check whether this combination has already been evaluated."""
    return os.path.exists(result_path(results_dir, clf_name, dataset_name))


def save_result(results_dir: str, clf_name: str, record: dict) -> None:
    """Save a single result to its own CSV file."""
    path = result_path(results_dir, clf_name, record["dataset"])
    pd.DataFrame([record]).to_csv(path, index=False)


def load_result(results_dir: str, clf_name: str, dataset_name: str) -> dict:
    """Load a previously saved result."""
    path = result_path(results_dir, clf_name, dataset_name)
    return pd.read_csv(path).to_dict(orient="records")[0]


def consolidate_results(results_dir: str, output_path: str) -> pd.DataFrame:
    """
    Read all individual CSVs from results_dir and write one master CSV.
    """
    all_files = sorted([
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv")
    ])

    if not all_files:
        print("No result files found.")
        return pd.DataFrame()

    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    df = df.sort_values(["classifier", "dataset"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Consolidated {len(all_files)} files into {output_path}")

    return df


# ============================================================================
# 4. MAIN LOOP
# ============================================================================

def run_all(dataset_names, classifier_names, seed, n_jobs, results_dir):
    """
    Iterate over every (classifier, dataset) pair.
    Skips pairs that already have saved results (resumability).
    """
    total = len(dataset_names) * len(classifier_names)
    done = 0
    failed = 0

    for dataset_name in dataset_names:
        for clf_name in classifier_names:
            done += 1
            tag = f"[{done}/{total}]"

            # --- Skip if already computed ---
            if result_exists(results_dir, clf_name, dataset_name):
                prev = load_result(results_dir, clf_name, dataset_name)
                print(f"{tag} {clf_name} on {dataset_name} ... "
                      f"CACHED (acc={prev['accuracy']:.4f})")
                continue

            # --- Build a fresh classifier ---
            try:
                clf = make_classifier(clf_name, seed, n_jobs)
            except Exception as e:
                print(f"{tag} {clf_name} on {dataset_name} ... "
                      f"INIT FAILED: {e}")
                failed += 1
                continue

            # --- Evaluate ---
            print(f"{tag} {clf_name} on {dataset_name} ... ",
                  end="", flush=True)
            try:
                record = evaluate(clf, dataset_name)
                record["classifier"] = clf_name
                save_result(results_dir, clf_name, record)

                print(f"acc={record['accuracy']:.4f}  "
                      f"fit={record['fit_time_s']:.2f}s  "
                      f"predict={record['predict_time_s']:.2f}s  "
                      f"mem={record['peak_fit_mem_mb']:.1f}MB")

            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

    print(f"\nFinished. {done - failed} succeeded, {failed} failed.")


def main():
    parser = argparse.ArgumentParser(
        description="Run WEASEL 2.0 improvement experiments on UCR datasets."
    )
    parser.add_argument(
        "--datasets", default="subset", choices=["subset", "all"],
        help="'subset' = 30 representative datasets, "
             "'all' = full UCR archive."
    )
    parser.add_argument(
        "--seed", type=int, default=1379,
        help="Random seed passed to all classifiers."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=4,
        help="Number of parallel threads for SFA transforms."
    )
    parser.add_argument(
        "--results_dir", default="results/raw",
        help="Directory for individual per-experiment CSV files."
    )
    parser.add_argument(
        "--output", default="results/all_results.csv",
        help="Path for the consolidated master CSV."
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    dataset_names = get_dataset_names(args.datasets)

    print(f"Datasets:    {len(dataset_names)} ({args.datasets})")
    print(f"Classifiers: {len(CLASSIFIER_NAMES)}")
    print(f"Total runs:  {len(dataset_names) * len(CLASSIFIER_NAMES)}")
    print(f"Seed:        {args.seed}")
    print(f"n_jobs:      {args.n_jobs}")
    print(f"Results dir: {args.results_dir}")
    print()

    run_all(
        dataset_names, CLASSIFIER_NAMES,
        args.seed, args.n_jobs, args.results_dir,
    )

    consolidate_results(args.results_dir, args.output)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
