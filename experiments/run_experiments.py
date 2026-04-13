import os
import time
import warnings
import tracemalloc
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import pandas as pd
from aeon.datasets import load_classification
from weasel.classification.dictionary_based import WEASEL_V2
from weasel_variants import (
    WEASEL_V2_TFIDF,
    WEASEL_V2_SublinearTF,
    WEASEL_V2_SGD,
    WEASEL_V2_NoDiff,
    WEASEL_V2_FixedEnsemble,
    WEASEL_V2_HalfEnsemble,
    WEASEL_V2_Config,
    WEASEL_V2_AdaptiveEnsemble
)

np.seterr(divide="ignore", over="ignore", invalid="ignore")
warnings.filterwarnings("ignore", category=RuntimeWarning)


SUBSET_DATASETS = [
    # Small train, short series
    "ItalyPowerDemand",       # n=67,  len=24,   2 classes
    "SonyAIBORobotSurface1",  # n=20,  len=70,   2 classes

    # Small train, medium series
    "ArrowHead",              # n=36,  len=175,  3 classes
    "GunPoint",               # n=50,  len=150,  2 classes
    "Coffee",                 # n=28,  len=286,  2 classes

    # Small train, long series
    "HouseTwenty",            # n=40,  len=2000, 2 classes
    "ACSF1",                  # n=100, len=1460, 10 classes
    "Rock",                   # n=20,  len=2844, 4 classes

    # Medium train, medium series
    "Adiac",                  # n=390, len=391,  37 classes
    "Fish",                   # n=175, len=463,  7 classes
    "SwedishLeaf",            # n=500, len=128,  15 classes

    # Medium train, long series
    "UWaveGestureLibraryX",   # n=896, len=315,  8 classes
    "InlineSkate",            # n=100, len=1882, 7 classes
    "Haptics",                # n=155, len=1092, 5 classes

    # Medium train, short series
    "DistalPhalanxOutlineCorrect",  # n=600, len=80, 2 classes
    "MiddlePhalanxTW",              # n=399, len=80, 6 classes

    # Medium train, long series, many classes
    "EOGHorizontalSignal",    # n=362, len=1250, 12 classes

    # Large train
    "ElectricDevices",        # n=8926, len=96,  7 classes
    "FordA",                  # n=3601, len=500, 2 classes
    "Crop",                   # n=7200, len=46,  24 classes

    # Many classes
    "FiftyWords",             # n=450, len=270,  50 classes
    "Phoneme",                # n=214, len=1024, 39 classes
]

def make_classifier(name, seed, n_jobs):
    registry = {
        "WEASEL2_baseline": lambda: WEASEL_V2(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_tfidf": lambda: WEASEL_V2_TFIDF(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_sublinear_tf": lambda: WEASEL_V2_SublinearTF(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_sgd": lambda: WEASEL_V2_SGD(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_no_diff": lambda: WEASEL_V2_NoDiff(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r10": lambda: WEASEL_V2_FixedEnsemble(r_max=10, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r20": lambda: WEASEL_V2_FixedEnsemble(r_max=20, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r25": lambda: WEASEL_V2_FixedEnsemble(r_max=25, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r30": lambda: WEASEL_V2_FixedEnsemble(r_max=30, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r50": lambda: WEASEL_V2_FixedEnsemble(r_max=50,  random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r75": lambda: WEASEL_V2_FixedEnsemble(r_max=75,  random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r100": lambda: WEASEL_V2_FixedEnsemble(r_max=100, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r125": lambda: WEASEL_V2_FixedEnsemble(r_max=125, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r150": lambda: WEASEL_V2_FixedEnsemble(r_max=150, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_r175": lambda: WEASEL_V2_FixedEnsemble(r_max=175, random_state=seed, n_jobs=n_jobs),
        "WEASEL2_half": lambda: WEASEL_V2_HalfEnsemble(random_state=seed, n_jobs=n_jobs),
        "WEASEL2_w12": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=12, n_jobs=n_jobs),
        "WEASEL2_w24": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=24, n_jobs=n_jobs),
        "WEASEL2_w44": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=44, n_jobs=n_jobs),
        "WEASEL2_w64": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=64, n_jobs=n_jobs),
        "WEASEL2_w84": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=84, n_jobs=n_jobs),
        "WEASEL2_w100": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=100, n_jobs=n_jobs),
        "WEASEL2_w124": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=124, n_jobs=n_jobs),
        "WEASEL2_w200": lambda: WEASEL_V2_Config(ensemble_size=50, random_state=seed, max_window=200, n_jobs=n_jobs),
        "WEASEL2_AdaptiveEnsemble": lambda: WEASEL_V2_AdaptiveEnsemble(random_state=seed, n_jobs=n_jobs),

    }

    return registry[name]()


def evaluate(clf_name, dataset_name, seed, n_jobs):
    """Load data, fit, predict, and return a metrics dict."""
    X_train, y_train = load_classification(dataset_name, split="train")
    X_test, y_test = load_classification(dataset_name, split="test")

    clf = make_classifier(clf_name, seed, n_jobs)

    # Fit
    tracemalloc.start()
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    ensemble_size = getattr(clf, "ensemble_size", None)
    max_window = getattr(clf, "max_window", None)

    # Predict
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    predict_time = time.perf_counter() - t0

    return {
        "classifier": clf_name,
        "dataset": dataset_name,
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "ts_length": X_train.shape[-1],
        "n_classes": len(np.unique(y_train)),
        "ensemble_size": ensemble_size,
        "max_window": max_window,
        "n_features": int(clf.total_features_count) if hasattr(clf, "total_features_count") else -1,
        "fit_time_s": round(fit_time, 4),
        "predict_time_s": round(predict_time, 4),
        "predict_time_per_sample_ms": round(predict_time / X_test.shape[0] * 1000, 4),
        "peak_fit_mem_mb": round(peak_mem / (1024 * 1024), 2),
        "accuracy": round(float(np.mean(y_pred == y_test)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_test, y_pred)), 6),
        "train_cv_accuracy": round(float(clf.cross_val_score), 6) if hasattr(clf, "cross_val_score") else -1,
    }


if __name__ == "__main__":
    SEED = 1379
    N_JOBS = 4
    RESULTS_DIR = "results/adaptive_ensemble/"
    OUTPUT_CSV = "results/adaptive_ensemble.csv"
    USE_SUBSET = False

    CLASSIFIERS = [
        "WEASEL2_baseline",
        "WEASEL2_tfidf",
        "WEASEL2_sublinear_tf",
        "WEASEL2_sgd",
        "WEASEL2_no_diff",
        "WEASEL2_r10",
        "WEASEL2_r20",
        "WEASEL2_r25",
        "WEASEL2_r30",
        "WEASEL2_r25",
        "WEASEL2_r50",
        "WEASEL2_r75",
        "WEASEL2_r100",
        "WEASEL2_r125",
        "WEASEL2_r150",
        "WEASEL2_r175",
        "WEASEL2_half",
        "WEASEL2_w12",
        "WEASEL2_w24",
        "WEASEL2_w44",
        "WEASEL2_w64",
        "WEASEL2_w84",
        "WEASEL2_w100",
        "WEASEL2_w124",
        "WEASEL2_w200",
        "WEASEL2_AdaptiveEnsemble",
    ]

    if USE_SUBSET:
        datasets = SUBSET_DATASETS
    else:
        from aeon.datasets.tsc_datasets import univariate_equal_length
        datasets = sorted(list(univariate_equal_length))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    total = len(datasets) * len(CLASSIFIERS)
    done = 0
    failed = 0

    for dataset in datasets:
        for clf_name in CLASSIFIERS:
            done += 1
            tag = f"[{done}/{total}]"
            result_file = os.path.join(RESULTS_DIR, f"{clf_name}__{dataset}.csv")

            # Skip if already done
            if os.path.exists(result_file):
                cached = pd.read_csv(result_file).to_dict(orient="records")[0]
                print(f"{tag} {clf_name} on {dataset} ... CACHED (acc={cached['accuracy']:.4f})")
                continue

            print(f"{tag} {clf_name} on {dataset} ... ", end="", flush=True)
            try:
                record = evaluate(clf_name, dataset, SEED, N_JOBS)
                pd.DataFrame([record]).to_csv(result_file, index=False)
                print(
                    f"acc={record['accuracy']:.4f}  "
                    f"fit={record['fit_time_s']:.2f}s  "
                    f"predict={record['predict_time_s']:.2f}s  "
                    f"mem={record['peak_fit_mem_mb']:.1f}MB"
                )
            except Exception as e:
                print(f"FAILED: {e}")
                failed += 1

    print(f"\nDone. {done - failed} succeeded, {failed} failed.")

    # Consolidate all CSVs into one
    all_files = sorted(
        os.path.join(RESULTS_DIR, f)
        for f in os.listdir(RESULTS_DIR)
        if f.endswith(".csv")
    )
    if all_files:
        df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
        df = df.sort_values(["classifier", "dataset"]).reset_index(drop=True)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Consolidated {len(all_files)} files -> {OUTPUT_CSV}")