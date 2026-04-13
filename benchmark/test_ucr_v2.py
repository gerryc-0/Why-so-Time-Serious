import os

os.environ["KMP_WARNINGS"] = "off"

import itertools
import sys
import time
from warnings import simplefilter

simplefilter(action="ignore", category=FutureWarning)
simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from weasel.classification.dictionary_based import WEASEL_V2, WEASEL, BOSSEnsemble, MUSE, MUSE_V2
from aeon.classification.hybrid import HIVECOTEV2, HIVECOTEV1
from aeon.classification.convolution_based import RocketClassifier, MiniRocketClassifier, MultiRocketClassifier
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.classification.dictionary_based import ContractableBOSS, TemporalDictionaryEnsemble
from aeon.classification.convolution_based import HydraClassifier

# list of datasets to reproduce - importing from UCR using aeon library
ucr_114 = [
    "ACSF1", "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME",
    "Car", "CBF", "Chinatown", "ChlorineConcentration", "CinCECGTorso",
    "Coffee", "Computers", "CricketX", "CricketY", "CricketZ", "Crop",
    "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200",
    "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal",
    "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR",
    "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "GunPoint", "GunPointAgeSpan",
    "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "Ham",
    "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate",
    "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
    "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW",
    "MixedShapesRegularTrain", "MixedShapesSmallTrain", "MoteStrain",
    "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2", "OliveOil",
    "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme",
    "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure",
    "PigCVP", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW",
    "RefrigerationDevices", "Rock", "ScreenType", "SemgHandGenderCh2",
    "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShakeGestureWiimoteZ",
    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SmoothSubspace",
    "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG",
    "TwoPatterns", "UMD", "UWaveGestureLibraryAll", "UWaveGestureLibraryX",
    "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer", "Wine",
    "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
]

used_dataset = ucr_114
# used_dataset = ["Crop", "ElectricDevices", "FordB", "StarLightCurves"]

def get_classifiers(threads_to_use):
    clfs = {
        "WEASEL":      WEASEL(random_state=1379, n_jobs=threads_to_use),
        "WEASEL 2.0":  WEASEL_V2(random_state=1379, n_jobs=threads_to_use),
        "cBOSS (D)":   ContractableBOSS(random_state=1379, n_jobs=threads_to_use),
        "Hydra (K/D)": HydraClassifier(random_state=1379, n_jobs=threads_to_use),
        "MultiRocket": MultiRocketClassifier(random_state=1379, n_jobs=threads_to_use),
        "R_DST":       RDSTClassifier(random_state=1379, n_jobs=threads_to_use),
        "Rocket":      RocketClassifier(random_state=1379, n_jobs=threads_to_use),
        "MiniRocket":  MiniRocketClassifier(random_state=1379, n_jobs=threads_to_use),
    }
    return clfs


# config based on hardware / time constraints
parallel_jobs = 1
threads_to_use = 4


if __name__ == "__main__":

    def _parallel_fit(dataset_name, clf_name):
        simplefilter(action="ignore", category=FutureWarning)
        simplefilter(action="ignore", category=UserWarning)

        from aeon.datasets import load_classification
        X_train, y_train = load_classification(dataset_name, split="train")
        X_test, y_test = load_classification(dataset_name, split="test")
        try:
            return make_run(X_test, X_train, clf_name, dataset_name, y_test, y_train)
        except Exception as e:
            print(f"Skipping {dataset_name} due to error: {e}")
            return {}


    def make_run(X_test, X_train, clf_name, dataset_name, y_test, y_train):
        """Run experiments."""
        sum_scores = {
            clf_name: {
                "dataset": [],
                "all_scores": [],
                "train_scores": [],
                "all_fit": [],
                "all_pred": [],
                "fit_time": 0.0,
                "pred_time": 0.0,
            }
        }

        clf = get_classifiers(threads_to_use)[clf_name]
        print(f"Running {clf_name} on {dataset_name} ... ")
        fit_time = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_time = np.round(time.perf_counter() - fit_time, 5)

        pred_time = time.perf_counter()
        acc = clf.score(X_test, y_test)
        train_acc = clf.cross_val_score if hasattr(clf, "cross_val_score") else 0
        pred_time = np.round(time.perf_counter() - pred_time, 5)
        print(
            f"{clf_name},{dataset_name},"
            + f"{np.round(acc, 3)},"
            + f"{np.round(fit_time, 2)},"
            + f"{np.round(pred_time, 2)}"
            + (
                f",{clf.total_features_count}"
                if hasattr(clf, "total_features_count")
                else f""
            )
            + (
                f",{clf.steps[0][-1].total_features_count}"
                if hasattr(clf, "steps")
                   and (len(clf.steps) > 0)
                   and hasattr(clf.steps[0][-1], "total_features_count")
                else f""
            ),
            flush=True,
        )
        sum_scores[clf_name]["dataset"].append(dataset_name)
        sum_scores[clf_name]["all_scores"].append(acc)
        sum_scores[clf_name]["train_scores"].append(train_acc)
        sum_scores[clf_name]["all_fit"].append(fit_time)
        sum_scores[clf_name]["all_pred"].append(pred_time)
        sum_scores[clf_name]["fit_time"] += sum_scores[clf_name]["fit_time"] + fit_time
        sum_scores[clf_name]["pred_time"] += (
                sum_scores[clf_name]["pred_time"] + pred_time
        )

        return sum_scores


    parallel_res = Parallel(n_jobs=parallel_jobs, timeout=9999999, batch_size=1)(
        delayed(_parallel_fit)(dataset, clf_name)
        for dataset, clf_name in itertools.product(
            used_dataset, get_classifiers(threads_to_use)
        )
    )

    sum_scores = {}
    for result in parallel_res:
        if not sum_scores:
            sum_scores = result
        else:
            for name, data in result.items():
                if name not in sum_scores:
                    sum_scores[name] = {}
                for key, value in data.items():
                    if key not in sum_scores[name]:
                        if type(value) == list:
                            sum_scores[name][key] = []
                        else:
                            sum_scores[name][key] = 0
                    sum_scores[name][key] += value

    print("\n\n---- Final results -----")

    for name, _ in sum_scores.items():
        print("---- Name", name, "-----")
        print(
            "Total mean-accuracy:", np.round(np.mean(sum_scores[name]["all_scores"]), 3)
        )
        print(
            "Total std-accuracy:", np.round(np.std(sum_scores[name]["all_scores"]), 3)
        )
        print(
            "Total median-accuracy:",
            np.round(np.median(sum_scores[name]["all_scores"]), 3),
        )
        print("Total fit_time:", np.round(sum_scores[name]["fit_time"], 2))
        print("Total pred_time:", np.round(sum_scores[name]["pred_time"], 2))
        print("-----------------")

    csv_timings = []
    csv_scores = []
    for name, _ in sum_scores.items():
        all_accs = sum_scores[name]["all_scores"]
        all_train_accs = sum_scores[name]["train_scores"]
        all_datasets = sum_scores[name]["dataset"]
        for acc, train_acc, dataset_name in zip(all_accs, all_train_accs, all_datasets):
            csv_scores.append((name, dataset_name, acc, train_acc))

        all_fit = np.round(sum_scores[name]["all_fit"], 2)
        all_pred = np.round(sum_scores[name]["all_pred"], 2)
        for fit, pred, dataset_name in zip(all_fit, all_pred, all_datasets):
            csv_timings.append((name, dataset_name, fit, pred))

    log_no = 1
    while os.path.exists("ucr-8-classifiers-accuracy-%s.csv" % log_no):
        log_no += 1

    pd.DataFrame.from_records(
        csv_scores,
        columns=["Classifier", "Dataset", "Accuracy", "Train-Acc"],
    ).to_csv("ucr-8-classifiers-accuracy-%s.csv" % log_no, index=False)

    pd.DataFrame.from_records(
        csv_timings,
        columns=["Classifier", "Dataset", "Fit-Time", "Predict-Time"],
    ).to_csv("ucr-8-classifiers-runtime-%s.csv" % log_no, index=False)
