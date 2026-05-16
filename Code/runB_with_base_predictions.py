"""
DDoS Ensemble Voting Strategies - Run B RE-RUN (with base prediction save)
==========================================================================
Re-runs Run B (22 features, 99,999 flows) under STRICT LOCKED conditions
identical to the original Run B execution.

PURPOSE OF RE-RUN
    The original Run B saved only the four ENSEMBLE STRATEGY predictions
    (Majority/Weighted/Stacking/Adaptive). It did not save the four BASE
    CLASSIFIER predictions (DT/LR/SVM/MLP). The Run D script computes
    pairwise disagreement on the BASE classifiers, so we cannot compute
    apples-to-apples Run B disagreement from the existing predictions
    file. This re-run saves base predictions per fold so the disagreement
    metric in the dissertation discussion section can be reported using
    Run D's exact formula on Run B's data.

LOCKED from Run A/B (DO NOT CHANGE):
    - Top 4 base models: Decision Tree, Logistic Regression, SVM, MLP
    - Static weights:    0.9641 for all four
    - CV splits:         5-fold StratifiedKFold
    - Random state:      42
    - Meta-learner:      Logistic Regression (Stacking)
    - Same hyperparameters as original Run A/B

OUTPUT
    - ensemble_strategy_results_v2_22feat.csv (4x4 metrics table)
    - ensemble_disagreement_v2_22feat.csv (per-fold pairwise disagreement)
    - ensemble_selected_models_v2_22feat.csv
    - ensemble_<strategy>_perclass_v2_22feat.csv (4 files)
    - ensemble_<strategy>_confusion_v2_22feat.csv (4 files)
    - predictions_full_v2_22feat.csv  <-- NEW (base + ensemble preds)

Usage:
    python runB_with_base_predictions.py
"""


import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------
# Force unbuffered stdout/stderr
# ---------------------------------------------------------------------
import sys
import os
os.environ["PYTHONUNBUFFERED"] = "1"
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:
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
# Configuration — RUN B ONLY
# ============================================================
DATA_PATH_22 = "runB_22feat_99999flows.csv"  # Run B dataset (99,999 flows)


TARGET_COL = "label"
N_SPLITS = 5
RANDOM_STATE = 42
OUT_SUFFIX = "v2_22feat"  # Run B re-run output suffix


# ============================================================
# LOCKED base models from Run A (same as Run D)
# ============================================================
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
    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    df.columns = [c.strip() for c in df.columns]


    drop_cols = [c for c in ["src_ip", "dst_ip"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped IP columns: {drop_cols}")


    if "protocol" in df.columns and df["protocol"].dtype == object:
        df["protocol"] = df["protocol"].astype(str).str.upper()
        from sklearn.preprocessing import LabelEncoder as LE
        proto_le = LE()
        df["protocol"] = proto_le.fit_transform(df["protocol"])
        print("Encoded protocol column")


    return df


# ============================================================
# Helper functions (identical to Run D script)
# ============================================================
def get_probabilities(clf, X_test):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X_test)
    return None


def evaluate_predictions(y_true, y_pred, y_prob=None, n_classes=9):
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
    n_samples = base_preds[0].shape[0]
    final = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        votes = [pred[i] for pred in base_preds]
        counter = Counter(votes)
        final[i] = counter.most_common(1)[0][0]
    return final


def weighted_vote(base_probs, weights):
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
# Main ensemble evaluation function
# ============================================================
def run_ensemble_evaluation(X, y_encoded, label_encoder,
                            run_name, n_features, out_suffix):
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


    # Track base classifier predictions across all folds
    all_base_preds = {name: [] for name in model_names}
    all_fold_ids = []


    adaptive_voter = AdaptiveVoter(n_models=len(model_names), n_classes=n_classes)
    disagreement_per_fold = []


    base_model_fold_f1 = {name: [] for name in model_names}


    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
        print(f"\n--- Fold {fold}/{N_SPLITS} ---")


        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]


        all_y_true.extend(y_test.tolist())
        # Track fold ID per sample
        all_fold_ids.extend([fold] * len(y_test))


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
            # Save base classifier predictions across folds
            all_base_preds[name].extend(pred.tolist())


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


        # Disagreement (same formula as Run D)
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


    disagree_df = pd.DataFrame({
        "fold": list(range(1, N_SPLITS + 1)),
        "pairwise_disagreement": disagreement_per_fold,
    })
    disagree_df.to_csv(f"ensemble_disagreement{suffix}.csv", index=False)


    # Save full predictions CSV with base classifier columns
    predictions_df = pd.DataFrame({
        "fold": all_fold_ids,
        "y_true": all_y_true,
        "pred_majority_vote": all_preds["Majority Vote"],
        "pred_weighted_vote": all_preds["Weighted Vote"],
        "pred_stacking": all_preds["Stacking"],
        "pred_adaptive_vote": all_preds["Adaptive Vote"],
    })
    for name in model_names:
        col_name = "pred_" + name.lower().replace(" ", "_")
        predictions_df[col_name] = all_base_preds[name]
    predictions_df.to_csv(f"predictions_full{suffix}.csv", index=False)


    print(f"\nSaved CSVs with suffix '{suffix}':")
    print(f"  - ensemble_strategy_results{suffix}.csv")
    print(f"  - ensemble_selected_models{suffix}.csv")
    print(f"  - ensemble_disagreement{suffix}.csv")
    print(f"  - predictions_full{suffix}.csv  (NEW: base + ensemble preds)")
    print(f"  - Per-class reports and confusion matrices for each strategy")


    return summary_df


# ============================================================
# MAIN — run B only
# ============================================================
print("\n" + "#" * 70)
print("# RUN B RE-RUN (22 features, 99,999 flows)")
print("# Adding base classifier predictions for disagreement audit")
print("#" * 70)


df_B = load_and_prep(DATA_PATH_22)
y_B = df_B[TARGET_COL]
X_B = df_B.drop(columns=[TARGET_COL])


label_encoder = LabelEncoder()
y_B_encoded = label_encoder.fit_transform(y_B)


print(f"Feature count: {X_B.shape[1]}")
print(f"Total samples: {len(X_B)}")
print(f"Class labels: {list(label_encoder.classes_)}")


_run_B_start = time.time()


summary_B = run_ensemble_evaluation(
    X=X_B, y_encoded=y_B_encoded, label_encoder=label_encoder,
    run_name="Run B RE-RUN: 22 Features (99,999 flows)",
    n_features=X_B.shape[1],
    out_suffix=OUT_SUFFIX
)


_run_B_seconds = time.time() - _run_B_start
print(f"\nRun B wall-clock time: {_run_B_seconds:.1f} seconds "
      f"({_run_B_seconds/60:.1f} min, {_run_B_seconds/3600:.2f} hours)")


# Save timing
timing_df = pd.DataFrame([
    {"run": "Run B (re-run)", "features": 22, "samples": 99999,
     "seconds": _run_B_seconds, "minutes": _run_B_seconds/60,
     "hours": _run_B_seconds/3600},
])
timing_df.to_csv(f"ensemble_wallclock_timing_{OUT_SUFFIX}.csv", index=False)


print("\n" + "=" * 70)
print("DONE — Run B re-run complete")
print("=" * 70)
print("\nNext step: run analyse_runB_postrun.py to compute disagreement")
print("from predictions_full_v2_22feat.csv")
