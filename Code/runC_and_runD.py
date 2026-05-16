"""
DDoS Ensemble Voting Strategies - Run C and Run D (5x Scaled)
==============================================================
Scales Run A and Run B dataset by 5x (99,999 -> 499,995 flows)
under strict conditions.


LOCKED from Run A/B (DO NOT CHANGE):
    - Top 4 base models: Decision Tree, Logistic Regression, SVM, MLP
    - Static weights:    0.9641 for all four (equal, from Run A weighted performance score)
    - Weighted performance score formula: 0.35 F1 + 0.30 MCC + 0.20 Recall + 0.10 Prec + 0.05 Acc
    - CV splits:         5-fold StratifiedKFold
    - Random state:      42
    - Meta-learner:      Logistic Regression (Stacking)
    - Same hyperparameters as Run A/B


CHANGED between Run C and Run D:
    - Run C: 27 features, 499,995 flows (scaled Run A)
    - Run D: 22 features, 499,995 flows (scaled Run B, flags removed)


Usage:
    python runC_and_runD.py
"""


import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Force unbuffered stdout/stderr so every print() streams live to the
# terminal and to log files (tee / nohup / redirect). This means you
# see progress in real time regardless of how the script is launched.
# ---------------------------------------------------------------------
import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
    # Python < 3.7 fallback
    import functools
    print = functools.partial(print, flush=True)  # noqa: A001


import time
import numpy as np
import pandas as pd
from collections import Counter


from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR_Meta


from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



# ============================================================
# Configuration
# ============================================================
DATA_PATH_27 = "runC_27feat_499995flows.csv"  # Run C
DATA_PATH_22 = "runD_22feat_499995flows.csv"  # Run D


TARGET_COL = "label"
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# LOCKED base models from Run A (top 4 by weighted performance score)
# ============================================================
# Same hyperparameters as Run A/B - strict conditions preserved
LOCKED_MODELS = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="gini", max_depth=7, splitter="best",
        min_samples_split=24, min_samples_leaf=10,
        random_state=RANDOM_STATE
    ),
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]),
    "SVM": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-4,
                     probability=True, random_state=RANDOM_STATE))
    ]),
    "MLP": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32,), activation="relu", solver="adam",
            alpha=0.0001, learning_rate_init=0.001, max_iter=300,
            random_state=RANDOM_STATE))
    ]),
}


# LOCKED weighted performance score weights from Run A (all four scored 0.9641)
LOCKED_WEIGHTS = {
    "Decision Tree": 0.9641,
    "Logistic Regression": 0.9641,
    "SVM": 0.9641,
    "MLP": 0.9641,
}


learning_families = {
    "Decision Tree": "rule-based",
    "Logistic Regression": "linear",
    "SVM": "margin-based",
    "MLP": "neural",
}



# ============================================================
# Data loading and preprocessing
# ============================================================
def load_and_prep(path):
    """Load a v3 dataset and return cleaned dataframe."""
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    df.columns = [c.strip() for c in df.columns]


    # v3 files already have src_ip/dst_ip dropped, but safe guard
    drop_cols = [c for c in ["src_ip", "dst_ip"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped IP columns: {drop_cols}")


    # Encode protocol if still string
    if "protocol" in df.columns and df["protocol"].dtype == object:
        df["protocol"] = df["protocol"].astype(str).str.upper()
        from sklearn.preprocessing import LabelEncoder as LE
        proto_le = LE()
        df["protocol"] = proto_le.fit_transform(df["protocol"])
        print("Encoded protocol column")


    return df



# ============================================================
# Helper functions
# ============================================================
def get_probabilities(clf, X_test):
    """Get probability predictions, handling pipelines."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)
    return None



def evaluate_predictions(y_true, y_pred, y_prob=None, n_classes=9):
    """Compute all metrics for a set of predictions."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_prob is not None:
        try:
            y_bin = label_binarize(y_true, classes=list(range(n_classes)))
            metrics["auc_macro"] = roc_auc_score(y_bin, y_prob, multi_class="ovr",
                                                  average="macro")
        except Exception:
            metrics["auc_macro"] = np.nan
    else:
        metrics["auc_macro"] = np.nan
    return metrics



def majority_vote(base_preds):
    """Hard majority vote across base classifier predictions."""
    n_samples = base_preds[0].shape[0]
    final = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes = [pred[i] for pred in base_preds]
        counter = Counter(votes)
        final[i] = counter.most_common(1)[0][0]
    return final



def weighted_vote(base_probs, weights):
    """Soft weighted vote using probability outputs."""
    weighted_sum = np.zeros_like(base_probs[0])
    total_weight = sum(weights)
    for prob, w in zip(base_probs, weights):
        weighted_sum += (w / total_weight) * prob
    return np.argmax(weighted_sum, axis=1), weighted_sum



class AdaptiveVoter:
    def __init__(self, n_models, n_classes):
        self.n_models = n_models
        self.n_classes = n_classes
        self.weights = np.ones(n_models) / n_models


    def update_weights(self, fold_f1_scores):
        scores = np.array(fold_f1_scores)
        if scores.sum() == 0:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = scores / scores.sum()


    def predict(self, base_probs):
        weighted_sum = np.zeros_like(base_probs[0])
        for prob, w in zip(base_probs, self.weights):
            weighted_sum += w * prob
        return np.argmax(weighted_sum, axis=1), weighted_sum



# ============================================================
# Main ensemble evaluation function (LOCKED models only)
# ============================================================
def run_ensemble_evaluation(X, y_encoded, label_encoder,
                            run_name, n_features, out_suffix):
    """Run ensemble pipeline using LOCKED base models and weights.


    Skips all baseline 8-model training - goes straight to ensemble
    voting with the 4 locked models from Run A.
    """


    n_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)


    print(f"\n{'#' * 70}")
    print(f"# RUN: {run_name} ({n_features} features)")
    print(f"{'#' * 70}")


    print(f"\n{'=' * 60}")
    print("STRICT CONDITIONS: Using LOCKED base models from Run A")
    print(f"{'=' * 60}")
    for name in LOCKED_MODELS.keys():
        print(f"  - {name} ({learning_families[name]})  "
              f"weight={LOCKED_WEIGHTS[name]:.4f}")


    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


    model_names = list(LOCKED_MODELS.keys())
    model_weights = [LOCKED_WEIGHTS[name] for name in model_names]


    ensemble_results = {
        "Majority Vote": {"accuracy": [], "precision_macro": [], "recall_macro": [],
                          "f1_macro": [], "mcc": [], "auc_macro": []},
        "Weighted Vote": {"accuracy": [], "precision_macro": [], "recall_macro": [],
                          "f1_macro": [], "mcc": [], "auc_macro": []},
        "Stacking": {"accuracy": [], "precision_macro": [], "recall_macro": [],
                     "f1_macro": [], "mcc": [], "auc_macro": []},
        "Adaptive Vote": {"accuracy": [], "precision_macro": [], "recall_macro": [],
                          "f1_macro": [], "mcc": [], "auc_macro": []},
    }


    all_y_true = []
    all_preds = {"Majority Vote": [], "Weighted Vote": [],
                 "Stacking": [], "Adaptive Vote": []}


    adaptive_voter = AdaptiveVoter(n_models=len(model_names), n_classes=n_classes)
    disagreement_per_fold = []


    # Track per-base-model F1 per fold for reporting
    base_model_fold_f1 = {name: [] for name in model_names}


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")


        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]


        all_y_true.extend(y_test.tolist())


        base_preds = []
        base_probs = []
        fold_f1_per_model = []


        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        stacking_meta_train = np.zeros((len(train_idx), n_classes * len(model_names)))
        stacking_meta_test = np.zeros((len(test_idx), n_classes * len(model_names)))


        for m_idx, (name, model) in enumerate(LOCKED_MODELS.items()):
            clf = clone(model)
            clf.fit(X_train, y_train)


            pred = clf.predict(X_test)
            prob = get_probabilities(clf, X_test)


            base_preds.append(pred)
            base_probs.append(prob)


            model_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            fold_f1_per_model.append(model_f1)
            base_model_fold_f1[name].append(model_f1)
            print(f"  {name}: F1={model_f1:.4f}")


            col_start = m_idx * n_classes
            col_end = col_start + n_classes


            for inner_train_idx, inner_val_idx in inner_skf.split(X_train, y_train):
                X_inner_train = X_train.iloc[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train.iloc[inner_val_idx]


                inner_clf = clone(model)
                inner_clf.fit(X_inner_train, y_inner_train)
                inner_prob = get_probabilities(inner_clf, X_inner_val)
                if inner_prob is not None:
                    stacking_meta_train[inner_val_idx, col_start:col_end] = inner_prob


            if prob is not None:
                stacking_meta_test[:, col_start:col_end] = prob


        # Disagreement
        pairwise_disagree = []
        for i in range(len(base_preds)):
            for j in range(i+1, len(base_preds)):
                pairwise_disagree.append(np.mean(base_preds[i] != base_preds[j]))
        avg_disagree = np.mean(pairwise_disagree)
        disagreement_per_fold.append(avg_disagree)
        print(f"  Avg pairwise disagreement: {avg_disagree:.4f} "
              f"({avg_disagree*100:.1f}%)")


        # Strategy 1: Majority Vote
        maj_pred = majority_vote(base_preds)
        m = evaluate_predictions(y_test, maj_pred, n_classes=n_classes)
        for k in ensemble_results["Majority Vote"]:
            ensemble_results["Majority Vote"][k].append(m.get(k, np.nan))
        all_preds["Majority Vote"].extend(maj_pred.tolist())
        print(f"  Majority Vote:  F1={m['f1_macro']:.4f}")


        # Strategy 2: Weighted Vote
        w_pred, w_prob = weighted_vote(base_probs, model_weights)
        m = evaluate_predictions(y_test, w_pred, w_prob, n_classes=n_classes)
        for k in ensemble_results["Weighted Vote"]:
            ensemble_results["Weighted Vote"][k].append(m.get(k, np.nan))
        all_preds["Weighted Vote"].extend(w_pred.tolist())
        print(f"  Weighted Vote:  F1={m['f1_macro']:.4f}")


        # Strategy 3: Stacking
        meta_clf = LR_Meta(max_iter=1000, random_state=RANDOM_STATE,
                           multi_class="multinomial", solver="lbfgs")
        meta_clf.fit(stacking_meta_train, y_train)
        s_pred = meta_clf.predict(stacking_meta_test)
        s_prob = meta_clf.predict_proba(stacking_meta_test)
        m = evaluate_predictions(y_test, s_pred, s_prob, n_classes=n_classes)
        for k in ensemble_results["Stacking"]:
            ensemble_results["Stacking"][k].append(m.get(k, np.nan))
        all_preds["Stacking"].extend(s_pred.tolist())
        print(f"  Stacking:       F1={m['f1_macro']:.4f}")


        # Strategy 4: Adaptive Vote
        a_pred, a_prob = adaptive_voter.predict(base_probs)
        m = evaluate_predictions(y_test, a_pred, a_prob, n_classes=n_classes)
        for k in ensemble_results["Adaptive Vote"]:
            ensemble_results["Adaptive Vote"][k].append(m.get(k, np.nan))
        all_preds["Adaptive Vote"].extend(a_pred.tolist())
        print(f"  Adaptive Vote:  F1={m['f1_macro']:.4f}  "
              f"weights={[f'{w:.3f}' for w in adaptive_voter.weights]}")


        adaptive_voter.update_weights(fold_f1_per_model)


    # ── Results summary ──
    print(f"\n{'=' * 60}")
    print(f"RESULTS SUMMARY - {run_name}")
    print(f"{'=' * 60}")


    print(f"\nAvg pairwise disagreement: {np.mean(disagreement_per_fold):.4f} "
          f"({np.mean(disagreement_per_fold)*100:.1f}%)")
    print(f"Per-fold disagreement: "
          f"{[f'{d*100:.2f}%' for d in disagreement_per_fold]}")


    print("\nBase model F1 per fold:")
    for name in model_names:
        folds_str = ", ".join([f"{v:.4f}" for v in base_model_fold_f1[name]])
        mean_f1 = np.mean(base_model_fold_f1[name])
        print(f"  {name:22s}  mean={mean_f1:.4f}  folds=[{folds_str}]")


    summary_rows = []
    for strategy_name, metrics in ensemble_results.items():
        row = {"strategy": strategy_name}
        for metric_name, values in metrics.items():
            row[f"{metric_name}_mean"] = np.mean(values)
            row[f"{metric_name}_std"] = np.std(values)
        summary_rows.append(row)


    summary_df = pd.DataFrame(summary_rows).sort_values(
        by="f1_macro_mean", ascending=False)


    print("\nEnsemble Strategy Comparison:")
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(f"  {row['strategy']:20s}  "
              f"F1={row['f1_macro_mean']:.6f}+/-{row['f1_macro_std']:.6f}  "
              f"MCC={row['mcc_mean']:.6f}  "
              f"AUC={row['auc_macro_mean']:.6f}  "
              f"Acc={row['accuracy_mean']:.6f}")


    # ── Per-class classification reports ──
    print(f"\n{'=' * 60}")
    print("PER-CLASS CLASSIFICATION REPORTS")
    print(f"{'=' * 60}")


    all_y_true_arr = np.array(all_y_true)


    for strategy_name in ["Majority Vote", "Weighted Vote", "Stacking", "Adaptive Vote"]:
        preds_arr = np.array(all_preds[strategy_name])


        print(f"\n--- {strategy_name} ---")
        print(classification_report(
            all_y_true_arr, preds_arr,
            target_names=class_names, digits=6, zero_division=0
        ))


        cm = confusion_matrix(all_y_true_arr, preds_arr)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print("Confusion Matrix:")
        print(cm_df.to_string())
        print()


        report_dict = classification_report(
            all_y_true_arr, preds_arr,
            target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(
            f"ensemble_{strategy_name.lower().replace(' ', '_')}_perclass_{out_suffix}.csv"
        )


        cm_df.to_csv(
            f"ensemble_{strategy_name.lower().replace(' ', '_')}_confusion_{out_suffix}.csv"
        )


    # ── Save summary CSVs ──
    suffix = f"_{out_suffix}"
    summary_df.to_csv(f"ensemble_strategy_results{suffix}.csv", index=False)


    selection_info = pd.DataFrame([
        {"model": name, "learning_family": learning_families[name],
         "composite_score": LOCKED_WEIGHTS[name]}
        for name in model_names
    ])
    selection_info.to_csv(f"ensemble_selected_models{suffix}.csv", index=False)


    # Save disagreement per fold
    disagree_df = pd.DataFrame({
        "fold": list(range(1, N_SPLITS + 1)),
        "pairwise_disagreement": disagreement_per_fold,
    })
    disagree_df.to_csv(f"ensemble_disagreement{suffix}.csv", index=False)


    print(f"\nSaved CSVs with suffix '{suffix}':")
    print(f"  - ensemble_strategy_results{suffix}.csv")
    print(f"  - ensemble_selected_models{suffix}.csv")
    print(f"  - ensemble_disagreement{suffix}.csv")
    print(f"  - Per-class reports and confusion matrices for each strategy")


    return summary_df



# ============================================================
# RUN C: 27 features, 5x scale
# ============================================================
print("\n" + "#" * 70)
print("# PREPARING RUN C: 27 FEATURES (5x SCALE)")
print("#" * 70)


df_C = load_and_prep(DATA_PATH_27)
X_C = df_C.drop(columns=[TARGET_COL]).copy()
y_C = df_C[TARGET_COL].copy()


for col in X_C.columns:
    X_C[col] = pd.to_numeric(X_C[col], errors="coerce")


label_encoder = LabelEncoder()
y_C_encoded = label_encoder.fit_transform(y_C)
n_classes = len(label_encoder.classes_)


print(f"Class labels: {list(label_encoder.classes_)}")
print(f"Feature count: {X_C.shape[1]}")
print(f"Total samples: {len(X_C)}")


_run_C_start = time.time()
summary_C = run_ensemble_evaluation(
    X_C, y_C_encoded, label_encoder,
    run_name="Run C: 27 Features (5x scale)",
    n_features=27,
    out_suffix="v3_27feat"
)
_run_C_seconds = time.time() - _run_C_start
print(f"\nRun C wall-clock time: {_run_C_seconds:.1f} seconds "
      f"({_run_C_seconds/60:.1f} minutes, {_run_C_seconds/3600:.2f} hours)")



# ============================================================
# RUN D: 22 features, 5x scale (TCP flag features removed)
# ============================================================
print("\n\n" + "#" * 70)
print("# PREPARING RUN D: 22 FEATURES (5x SCALE, FLAGS REMOVED)")
print("#" * 70)


df_D = load_and_prep(DATA_PATH_22)
X_D = df_D.drop(columns=[TARGET_COL]).copy()
y_D = df_D[TARGET_COL].copy()


for col in X_D.columns:
    X_D[col] = pd.to_numeric(X_D[col], errors="coerce")


y_D_encoded = label_encoder.transform(y_D)


print(f"Feature count: {X_D.shape[1]}")
print(f"Total samples: {len(X_D)}")


_run_D_start = time.time()
summary_D = run_ensemble_evaluation(
    X_D, y_D_encoded, label_encoder,
    run_name="Run D: 22 Features (5x scale, flags removed)",
    n_features=22,
    out_suffix="v3_22feat"
)
_run_D_seconds = time.time() - _run_D_start
print(f"\nRun D wall-clock time: {_run_D_seconds:.1f} seconds "
      f"({_run_D_seconds/60:.1f} minutes, {_run_D_seconds/3600:.2f} hours)")



# ============================================================
# Final comparison: Run C vs Run D
# ============================================================
print("\n\n" + "=" * 70)
print("FINAL COMPARISON: RUN C (27 feat) vs RUN D (22 feat) at 5x scale")
print("=" * 70)


print(f"\nLocked base models: {list(LOCKED_MODELS.keys())}")
print(f"Locked weights:     {list(LOCKED_WEIGHTS.values())}")


print("\nRun C (27 features):")
for _, row in summary_C.iterrows():
    print(f"  {row['strategy']:20s}  F1={row['f1_macro_mean']:.6f}")


print("\nRun D (22 features):")
for _, row in summary_D.iterrows():
    print(f"  {row['strategy']:20s}  F1={row['f1_macro_mean']:.6f}")



# ============================================================
# Cross-run delta table: Run C -> Run D
# ============================================================
print("\n" + "=" * 70)
print("CROSS-RUN DELTA TABLE: Run C (27 feat) -> Run D (22 feat)")
print("=" * 70)


merged = summary_C.merge(summary_D, on="strategy", suffixes=("_C", "_D"))


delta_rows = []
for _, row in merged.iterrows():
    delta_rows.append({
        "Strategy": row["strategy"],
        "F1_C": row["f1_macro_mean_C"],
        "F1_D": row["f1_macro_mean_D"],
        "F1_Delta": row["f1_macro_mean_D"] - row["f1_macro_mean_C"],
        "MCC_C": row["mcc_mean_C"],
        "MCC_D": row["mcc_mean_D"],
        "MCC_Delta": row["mcc_mean_D"] - row["mcc_mean_C"],
        "Acc_C": row["accuracy_mean_C"],
        "Acc_D": row["accuracy_mean_D"],
        "Acc_Delta": row["accuracy_mean_D"] - row["accuracy_mean_C"],
    })


delta_df = pd.DataFrame(delta_rows)


print(f"\n{'Strategy':20s}  {'F1(C)':>10s}  {'F1(D)':>10s}  {'dF1':>10s}  "
      f"{'MCC(C)':>10s}  {'MCC(D)':>10s}  {'dMCC':>10s}")
print("-" * 100)
for _, row in delta_df.iterrows():
    print(f"  {row['Strategy']:18s}  {row['F1_C']:>10.6f}  {row['F1_D']:>10.6f}  {row['F1_Delta']:>+10.6f}  "
          f"{row['MCC_C']:>10.6f}  {row['MCC_D']:>10.6f}  {row['MCC_Delta']:>+10.6f}")


delta_df["F1_Drop"] = abs(delta_df["F1_Delta"])
most_robust = delta_df.loc[delta_df["F1_Drop"].idxmin(), "Strategy"]
least_robust = delta_df.loc[delta_df["F1_Drop"].idxmax(), "Strategy"]


print(f"\nMost robust to feature removal at scale:  {most_robust} "
      f"(dF1 = {delta_df.loc[delta_df['F1_Drop'].idxmin(), 'F1_Delta']:+.6f})")
print(f"Least robust to feature removal at scale: {least_robust} "
      f"(dF1 = {delta_df.loc[delta_df['F1_Drop'].idxmax(), 'F1_Delta']:+.6f})")


delta_df.to_csv("ensemble_cross_run_delta_CD.csv", index=False)
print("\nSaved: ensemble_cross_run_delta_CD.csv")


# ============================================================
# Wall-clock timing summary (scalability cost evidence)
# ============================================================
print("\n" + "=" * 70)
print("SCALABILITY TIMING SUMMARY")
print("=" * 70)
print(f"  Run C (27 feat, 5x scale):  {_run_C_seconds/60:.1f} min "
      f"({_run_C_seconds/3600:.2f} hours)")
print(f"  Run D (22 feat, 5x scale):  {_run_D_seconds/60:.1f} min "
      f"({_run_D_seconds/3600:.2f} hours)")
print(f"  Total wall-clock:           {(_run_C_seconds + _run_D_seconds)/60:.1f} min "
      f"({(_run_C_seconds + _run_D_seconds)/3600:.2f} hours)")


# Save timing CSV
timing_df = pd.DataFrame([
    {"run": "Run C", "features": 27, "samples": 499995,
     "seconds": _run_C_seconds, "minutes": _run_C_seconds/60,
     "hours": _run_C_seconds/3600},
    {"run": "Run D", "features": 22, "samples": 499995,
     "seconds": _run_D_seconds, "minutes": _run_D_seconds/60,
     "hours": _run_D_seconds/3600},
])
timing_df.to_csv("ensemble_wallclock_timing_CD.csv", index=False)
print("\nSaved: ensemble_wallclock_timing_CD.csv")


print("\n" + "=" * 70)
print("DONE - All Run C and Run D results saved")
print("=" * 70)
