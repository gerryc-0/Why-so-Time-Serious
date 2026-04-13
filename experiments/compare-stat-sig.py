import pandas as pd
from scipy.stats import wilcoxon

# paths
BASELINE_CSV = "results/all_results.csv"
BASELINE_CLF = "WEASEL2_baseline"

NEW_CSV      = "results/all_var_results.csv"
OUTPUT_CSV   = "results/comparison_all_var_results.csv"

if NEW_CSV == OUTPUT_CSV:
    raise Exception("WARNING new CSV must be different from output CSV")

METRICS = [
    "fit_time_s",
    "predict_time_s",
    "predict_time_per_sample_ms",
    "peak_fit_mem_mb",
    "accuracy",
    "balanced_accuracy",
    "train_cv_accuracy",
    "n_features",
]

LOWER_IS_BETTER = {"fit_time_s", "predict_time_s", "predict_time_per_sample_ms", "peak_fit_mem_mb", "n_features"}

NEGLIGIBLE = {
    "fit_time_s":                  0.5,
    "predict_time_s":              0.05,
    "predict_time_per_sample_ms":  0.05,
    "peak_fit_mem_mb":             1.0,
    "accuracy":                    0.005,
    "balanced_accuracy":           0.005,
    "train_cv_accuracy":           0.005,
    "n_features":                  100,
}

# UCR metadata: dataset -> (train_size, series_length, n_classes)
UCR_META = {
    "ACSF1":                          ("DEVICE",       100,  1460, 10),
    "Adiac":                          ("IMAGE",        390,  391,  37),
    "ArrowHead":                      ("IMAGE",        36,   175,  3),
    "BME":                            ("SIMULATED",    30,   128,  3),
    "Beef":                           ("SPECTRO",      30,   470,  5),
    "BeetleFly":                      ("IMAGE",        20,   512,  2),
    "BirdChicken":                    ("IMAGE",        20,   512,  2),
    "CBF":                            ("SIMULATED",    30,   128,  3),
    "Car":                            ("SENSOR",       60,   577,  4),
    "Chinatown":                      ("TRAFFIC",      20,   24,   2),
    "ChlorineConcentration":          ("SIMULATED",    467,  166,  3),
    "CinCECGTorso":                   ("ECG",          40,   1639, 4),
    "Coffee":                         ("SPECTRO",      28,   286,  2),
    "Computers":                      ("DEVICE",       250,  720,  2),
    "CricketX":                       ("MOTION",       390,  300,  12),
    "CricketY":                       ("MOTION",       390,  300,  12),
    "CricketZ":                       ("MOTION",       390,  300,  12),
    "Crop":                           ("IMAGE",        7200, 46,   24),
    "DiatomSizeReduction":            ("IMAGE",        16,   345,  4),
    "DistalPhalanxOutlineAgeGroup":   ("IMAGE",        400,  80,   3),
    "DistalPhalanxOutlineCorrect":    ("IMAGE",        600,  80,   2),
    "DistalPhalanxTW":                ("IMAGE",        400,  80,   6),
    "ECG200":                         ("ECG",          100,  96,   2),
    "ECG5000":                        ("ECG",          500,  140,  5),
    "ECGFiveDays":                    ("ECG",          23,   136,  2),
    "EOGHorizontalSignal":            ("EOG",          362,  1250, 12),
    "EOGVerticalSignal":              ("EOG",          362,  1250, 12),
    "Earthquakes":                    ("SENSOR",       322,  512,  2),
    "ElectricDevices":                ("DEVICE",       8926, 96,   7),
    "EthanolLevel":                   ("SPECTRO",      504,  1751, 4),
    "FaceAll":                        ("IMAGE",        560,  131,  14),
    "FaceFour":                       ("IMAGE",        24,   350,  4),
    "FacesUCR":                       ("IMAGE",        200,  131,  14),
    "FiftyWords":                     ("IMAGE",        450,  270,  50),
    "Fish":                           ("IMAGE",        175,  463,  7),
    "FordA":                          ("SENSOR",       3601, 500,  2),
    "FordB":                          ("SENSOR",       3636, 500,  2),
    "FreezerRegularTrain":            ("SENSOR",       150,  301,  2),
    "FreezerSmallTrain":              ("SENSOR",       28,   301,  2),
    "GunPoint":                       ("MOTION",       50,   150,  2),
    "GunPointAgeSpan":                ("MOTION",       135,  150,  2),
    "GunPointMaleVersusFemale":       ("MOTION",       135,  150,  2),
    "GunPointOldVersusYoung":         ("MOTION",       136,  150,  2),
    "Ham":                            ("SPECTRO",      109,  431,  2),
    "HandOutlines":                   ("IMAGE",        1000, 2709, 2),
    "Haptics":                        ("MOTION",       155,  1092, 5),
    "Herring":                        ("IMAGE",        64,   512,  2),
    "HouseTwenty":                    ("DEVICE",       40,   2000, 2),
    "InlineSkate":                    ("MOTION",       100,  1882, 7),
    "InsectEPGRegularTrain":          ("EPG",          62,   601,  3),
    "InsectEPGSmallTrain":            ("EPG",          17,   601,  3),
    "InsectWingbeatSound":            ("SOUND",        220,  256,  11),
    "ItalyPowerDemand":               ("SENSOR",       67,   24,   2),
    "LargeKitchenAppliances":         ("DEVICE",       375,  720,  3),
    "Lightning2":                     ("SENSOR",       60,   637,  2),
    "Lightning7":                     ("SENSOR",       70,   319,  7),
    "Mallat":                         ("SIMULATED",    55,   1024, 8),
    "Meat":                           ("SPECTRO",      60,   448,  3),
    "MedicalImages":                  ("IMAGE",        381,  99,   10),
    "MiddlePhalanxOutlineAgeGroup":   ("IMAGE",        400,  80,   3),
    "MiddlePhalanxOutlineCorrect":    ("IMAGE",        600,  80,   2),
    "MiddlePhalanxTW":                ("IMAGE",        399,  80,   6),
    "MixedShapesRegularTrain":        ("IMAGE",        500,  1024, 5),
    "MixedShapesSmallTrain":          ("IMAGE",        100,  1024, 5),
    "MoteStrain":                     ("SENSOR",       20,   84,   2),
    "NonInvasiveFetalECGThorax1":     ("ECG",          1800, 750,  42),
    "NonInvasiveFetalECGThorax2":     ("ECG",          1800, 750,  42),
    "OSULeaf":                        ("IMAGE",        200,  427,  6),
    "OliveOil":                       ("SPECTRO",      30,   570,  4),
    "PhalangesOutlinesCorrect":       ("IMAGE",        1800, 80,   2),
    "Phoneme":                        ("SOUND",        214,  1024, 39),
    "PickupGestureWiimoteZ":          ("SENSOR",       50,   None, 10),
    "PigAirwayPressure":              ("HEMODYNAMICS", 104,  2000, 52),
    "PigArtPressure":                 ("HEMODYNAMICS", 104,  2000, 52),
    "PigCVP":                         ("HEMODYNAMICS", 104,  2000, 52),
    "Plane":                          ("SENSOR",       105,  144,  7),
    "PowerCons":                      ("DEVICE",       180,  144,  2),
    "ProximalPhalanxOutlineAgeGroup": ("IMAGE",        400,  80,   3),
    "ProximalPhalanxOutlineCorrect":  ("IMAGE",        600,  80,   2),
    "ProximalPhalanxTW":              ("IMAGE",        400,  80,   6),
    "RefrigerationDevices":           ("DEVICE",       375,  720,  3),
    "Rock":                           ("SPECTRO",      20,   2844, 4),
    "ScreenType":                     ("DEVICE",       375,  720,  3),
    "SemgHandGenderCh2":              ("SPECTRO",      300,  1500, 2),
    "SemgHandMovementCh2":            ("SPECTRO",      450,  1500, 6),
    "SemgHandSubjectCh2":             ("SPECTRO",      450,  1500, 5),
    "ShakeGestureWiimoteZ":           ("SENSOR",       50,   None, 10),
    "ShapeletSim":                    ("SIMULATED",    20,   500,  2),
    "ShapesAll":                      ("IMAGE",        600,  512,  60),
    "SmallKitchenAppliances":         ("DEVICE",       375,  720,  3),
    "SmoothSubspace":                 ("SIMULATED",    150,  15,   3),
    "SonyAIBORobotSurface1":          ("SENSOR",       20,   70,   2),
    "SonyAIBORobotSurface2":          ("SENSOR",       27,   65,   2),
    "StarLightCurves":                ("SENSOR",       1000, 1024, 3),
    "Strawberry":                     ("SPECTRO",      613,  235,  2),
    "SwedishLeaf":                    ("IMAGE",        500,  128,  15),
    "Symbols":                        ("IMAGE",        25,   398,  6),
    "SyntheticControl":               ("SIMULATED",    300,  60,   6),
    "ToeSegmentation1":               ("MOTION",       40,   277,  2),
    "ToeSegmentation2":               ("MOTION",       36,   343,  2),
    "Trace":                          ("SENSOR",       100,  275,  4),
    "TwoLeadECG":                     ("ECG",          23,   82,   2),
    "TwoPatterns":                    ("SIMULATED",    1000, 128,  4),
    "UMD":                            ("SIMULATED",    36,   150,  3),
    "UWaveGestureLibraryAll":         ("MOTION",       896,  945,  8),
    "UWaveGestureLibraryX":           ("MOTION",       896,  315,  8),
    "UWaveGestureLibraryY":           ("MOTION",       896,  315,  8),
    "UWaveGestureLibraryZ":           ("MOTION",       896,  315,  8),
    "Wafer":                          ("SENSOR",       1000, 152,  2),
    "Wine":                           ("SPECTRO",      57,   234,  2),
    "WordSynonyms":                   ("IMAGE",        267,  270,  25),
    "Worms":                          ("MOTION",       181,  900,  5),
    "WormsTwoClass":                  ("MOTION",       181,  900,  2),
    "Yoga":                           ("IMAGE",        300,  426,  2),
}

# Bin boundaries
TRAIN_SIZE_BINS  = [(0, 100, "small (n<100)"), (100, 500, "medium (100-500)"), (500, 99999, "large (n>500)")]
SERIES_LEN_BINS = [(0, 200,   "short (<200)"), (200, 800, "medium (200-800)"), (800, 99999, "long (>800)")]
N_CLASSES_BINS   = [(0, 2, "binary (2)"), (2, 9, "mid (3-9)"), (9, 99999, "many (10+)")]


def assign_bin(value, bins):
    if value is None:
        return "unknown"
    for lo, hi, label in bins:
        if lo == 0 and value <= hi:
            return label
        elif lo < value <= hi:
            return label
    return "other"


def wilcoxon_str(deltas):
    non_zero = deltas[deltas.abs() > 1e-9]
    if len(non_zero) >= 10:
        _, p_val = wilcoxon(non_zero)
        return "p={:.3f} {}".format(p_val, "*" if p_val < 0.05 else "ns")
    return "n={} (too few)".format(len(non_zero))


def print_metric_summary(sub, available_metrics, label=""):
    if label:
        print(f"    [{label}]  n={len(sub)}")
    for m in available_metrics:
        col = f"{m}_delta"
        if col not in sub.columns:
            continue
        deltas = sub[col]
        mean_d = deltas.mean()
        neg_thresh = NEGLIGIBLE[m]
        if m in LOWER_IS_BETTER:
            n_better = (deltas < -neg_thresh).sum()
            n_worse = (deltas >  neg_thresh).sum()
        else:
            n_better = (deltas >  neg_thresh).sum()
            n_worse = (deltas < -neg_thresh).sum()
        n_negligible = len(deltas) - n_better - n_worse
        sig = wilcoxon_str(deltas)
        print(f"      {m:<35} mean delta: {mean_d:.6f}   better: {n_better:>3}  worse: {n_worse:>3}  negligible: {n_negligible:>3}  {sig}")


# load data
baseline_df = pd.read_csv(BASELINE_CSV)
new_df = pd.read_csv(NEW_CSV)

baseline_rows = baseline_df[baseline_df["classifier"] == BASELINE_CLF].set_index("dataset")
new_rows = new_df[new_df["classifier"] != BASELINE_CLF].copy()

common = set(baseline_rows.index) & set(new_rows["dataset"])
missing_from_baseline = set(new_rows["dataset"]) - set(baseline_rows.index)
if missing_from_baseline:
    print(f"[warn] {len(missing_from_baseline)} dataset(s) not in baseline, skipping: {sorted(missing_from_baseline)}")

available_metrics = [m for m in METRICS if m in baseline_rows.columns and m in new_rows.columns]

rows = []
for _, new_row in new_rows[new_rows["dataset"].isin(common)].iterrows():
    dataset = new_row["dataset"]
    b = baseline_rows.loc[dataset]
    row = {"classifier": new_row["classifier"], "dataset": dataset}
    for m in available_metrics:
        b_val = float(b[m])
        n_val = float(new_row[m])
        delta = n_val - b_val
        row[f"{m}_baseline"] = round(b_val, 6)
        row[f"{m}_new"] = round(n_val, 6)
        row[f"{m}_delta"] = round(delta, 6)
    if dataset in UCR_META:
        ds_type, train_n, length, n_cls = UCR_META[dataset]
        row["dataset_type"] = ds_type
        row["train_size_bin"] = assign_bin(train_n, TRAIN_SIZE_BINS)
        row["series_len_bin"] = assign_bin(length,  SERIES_LEN_BINS)
        row["n_classes_bin"] = assign_bin(n_cls,   N_CLASSES_BINS)
    else:
        row["dataset_type"] = row["train_size_bin"] = row["series_len_bin"] = row["n_classes_bin"] = "unknown"
    rows.append(row)

out = pd.DataFrame(rows).sort_values(["classifier", "dataset"])
out.to_csv(OUTPUT_CSV, index=False)

# console summary
print(f"\nBaseline : {BASELINE_CLF}")
print(f"Datasets : {len(common)} matched\n")

for clf in sorted(out["classifier"].unique()):
    sub = out[out["classifier"] == clf]
    print(f"  {clf}  ({len(sub)} datasets)")
    print_metric_summary(sub, available_metrics)
    print()

    for bin_col, bin_label in [
        ("dataset_type",   "by dataset type"),
        ("train_size_bin", "by training size"),
        ("series_len_bin", "by series length"),
        ("n_classes_bin",  "by number of classes"),
    ]:
        print(f"    -- {bin_label} --")
        for bin_val in sorted(sub[bin_col].unique()):
            grp = sub[sub[bin_col] == bin_val]
            print_metric_summary(grp, available_metrics, label=bin_val)
        print()

print(f"Saved to: {OUTPUT_CSV}")