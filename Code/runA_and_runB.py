"""
DDoS Ensemble Voting Strategies
================================
Reads baseline results, selects top 4 base classifiers using a
weighted performance score, then evaluates 4 ensemble voting
strategies under identical experimental conditions.

Two feature configurations are evaluated:
    Run A — 27 features (full feature set after dropping IPs)
    Run B — 22 features (TCP flag features removed to increase
             task difficulty and ensemble diversity)

IMPORTANT: Run B uses the same top 4 base models selected from
Run A to ensure strict experimental conditions. The only variable
that changes between runs is the feature set.

Weighted performance score (justified by multi-class classification
literature — Chicco & Jurman, 2020; Sokolova & Lapalme, 2009):
    F1 macro  = 0.35  (balanced precision-recall per class)
    MCC       = 0.30  (robust to class imbalance artefacts)
    Recall    = 0.20  (attack detection coverage is critical)
    Precision = 0.10  (false positive control)
    Accuracy  = 0.05  (overall correctness, minor weight)

Voting strategies:
    1. Majority (Hard) Voting  — equal votes, hard labels
    2. Weighted Voting         — static weights from weighted performance scores
    3. Stacking                — meta-learner combines base predictions
    4. Adaptive Voting         — dynamic weights updated per CV fold

Usage:
    python runA_and_runB.py
"""

import warnings
warnings.filterwarnings("ignore")

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
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression as LR_Meta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


# ============================================================
# Configuration
# ============================================================
DATA_PATH = "runA_27feat_99999flows.csv"
TARGET_COL = "label"
N_SPLITS = 5
RANDOM_STATE = 42
TOP_K = 4  # Number of base models to select for ensembles

# Weighted performance score weights
W_F1 = 0.35
W_MCC = 0.30
W_RECALL = 0.20
W_PRECISION = 0.10
W_ACCURACY = 0.05


# ============================================================
# Data loading and preprocessing
# ============================================================
print(f"Loading dataset: {DATA_PATH}")
df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset shape: {df_raw.shape}")

df_raw.columns = [c.strip() for c in df_raw.columns]

drop_cols = [c for c in ["src_ip", "dst_ip"] if c in df_raw.columns]
if drop_cols:
    df_raw = df_raw.drop(columns=drop_cols)
    print(f"Dropped IP columns: {drop_cols}")

if "protocol" in df_raw.columns:
    df_raw["protocol"] = df_raw["protocol"].astype(str).str.upper()
    from sklearn.preprocessing import LabelEncoder as LE
    proto_le = LE()
    df_raw["protocol"] = proto_le.fit_transform(df_raw["protocol"])
    print("Encoded protocol column")


# ============================================================
# Define all 8 base models (same as baseline script)
# ============================================================
def build_all_models(X_full, y_full, n_splits, random_state):
    """Build model dict, including GridSearchCV for Naive Bayes."""
    models = {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini", max_depth=7, splitter="best",
            min_samples_split=24, min_samples_leaf=10,
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=30, criterion="entropy",
            min_samples_split=5, min_samples_leaf=2, max_features="sqrt",
            random_state=random_state, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=random_state
        ),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "KNN": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", tol=1e-4,
                         probability=True, random_state=random_state))
        ]),
        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(32,), activation="relu", solver="adam",
                alpha=0.0001, learning_rate_init=0.001, max_iter=300,
                random_state=random_state))
        ]),
    }

    # Tune Naive Bayes
    # GaussianNB assumes normally distributed features, which raw network
    # flow data violates (heavy skew in packet counts, byte rates, IATs).
    # PowerTransformer (Yeo-Johnson) maps each feature closer to a Gaussian
    # distribution, satisfying the classifier's core assumption and improving
    # probability calibration (Yeo & Johnson, 2000).
    print("  Tuning Naive Bayes var_smoothing via GridSearchCV...")
    nb_pipe = Pipeline([
        # Step 1: Impute missing values with median (robust to outliers)
        ("imputer", SimpleImputer(strategy="median")),
        # Step 2: Yeo-Johnson power transform to approximate Gaussian
        # distributions — required because GaussianNB estimates class-
        # conditional densities assuming normality
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        # Step 3: GaussianNB with var_smoothing to be tuned
        ("clf", GaussianNB())
    ])
    # GridSearchCV over var_smoothing: controls the portion of the largest
    # variance added to all variances for numerical stability. Searching
    # 100 values on a log scale from 1e0 to 1e-9 ensures both heavily
    # smoothed and near-exact density estimates are evaluated.
    nb_grid = GridSearchCV(
        nb_pipe,
        {"clf__var_smoothing": np.logspace(0, -9, num=100)},
        cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="f1_macro",  # Macro F1 ensures tuning optimises across all classes equally
        n_jobs=-1, verbose=0
    )
    nb_grid.fit(X_full, y_full)
    best_vs = nb_grid.best_params_["clf__var_smoothing"]
    print(f"  Best var_smoothing: {best_vs:.6e}")
    print(f"  Best CV F1 macro:   {nb_grid.best_score_:.4f}")

    # Final Naive Bayes model uses the best var_smoothing found above,
    # with the same preprocessing pipeline for consistency
    models["Naive Bayes"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power", PowerTransformer(method="yeo-johnson", standardize=True)),
        ("clf", GaussianNB(var_smoothing=best_vs))
    ])

    return models


# Learning family classification for diversity check
learning_families = {
    "Decision Tree": "rule-based",
    "Random Forest": "ensemble-tree",
    "Gradient Boosting": "ensemble-boosting",
    "Logistic Regression": "linear",
    "KNN": "instance-based",
    "SVM": "margin-based",
    "MLP": "neural",
    "Naive Bayes": "probabilistic"
}


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
# Main ensemble evaluation function
# ============================================================
def run_ensemble_evaluation(X, y_encoded, label_encoder, all_models,
                            run_name, n_features, forced_selection=None,
                            forced_weights=None):
    """Run full ensemble pipeline: baseline → select → vote → report.

    Parameters
    ----------
    forced_selection : list of str or None
        If provided, skip weighted-performance-score selection and use these
        model names for the ensemble (ensures strict conditions
        across feature configurations).
    forced_weights : dict or None
        If provided, use these weighted performance scores as static weights
        for weighted voting (from the primary run).
    """

    n_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)

    print(f"\n{'#' * 70}")
    print(f"# RUN: {run_name} ({n_features} features)")
    print(f"{'#' * 70}")

    # ── STAGE 1: Baseline evaluation ──
    print(f"\n{'=' * 60}")
    print("STAGE 1: Baseline evaluation + weighted performance scoring")
    print(f"{'=' * 60}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    baseline_results = []

    for model_name, model in all_models.items():
        print(f"\nRunning: {model_name}")
        fold_metrics = {"accuracy": [], "precision_macro": [], "recall_macro": [],
                        "f1_macro": [], "mcc": []}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            clf = clone(model)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            fold_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            fold_metrics["precision_macro"].append(
                precision_score(y_test, y_pred, average="macro", zero_division=0))
            fold_metrics["recall_macro"].append(
                recall_score(y_test, y_pred, average="macro", zero_division=0))
            fold_metrics["f1_macro"].append(
                f1_score(y_test, y_pred, average="macro", zero_division=0))
            fold_metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))

            print(f"  Fold {fold}: Acc={fold_metrics['accuracy'][-1]:.4f}, "
                  f"F1={fold_metrics['f1_macro'][-1]:.4f}")

        row = {
            "model": model_name,
            "accuracy_mean": np.mean(fold_metrics["accuracy"]),
            "precision_macro_mean": np.mean(fold_metrics["precision_macro"]),
            "recall_macro_mean": np.mean(fold_metrics["recall_macro"]),
            "f1_macro_mean": np.mean(fold_metrics["f1_macro"]),
            "mcc_mean": np.mean(fold_metrics["mcc"]),
        }
        row["composite_score"] = (
            W_F1 * row["f1_macro_mean"] +
            W_MCC * row["mcc_mean"] +
            W_RECALL * row["recall_macro_mean"] +
            W_PRECISION * row["precision_macro_mean"] +
            W_ACCURACY * row["accuracy_mean"]
        )
        baseline_results.append(row)

    baseline_df = pd.DataFrame(baseline_results).sort_values(
        by="composite_score", ascending=False)
    baseline_df["learning_family"] = baseline_df["model"].map(learning_families)

    print("\n\nWeighted Performance Score Ranking:")
    print("-" * 80)
    print(f"  Weights: F1={W_F1}, MCC={W_MCC}, Recall={W_RECALL}, "
          f"Precision={W_PRECISION}, Accuracy={W_ACCURACY}")
    print("-" * 80)
    for _, row in baseline_df.iterrows():
        print(f"  {row['model']:25s}  wps={row['composite_score']:.4f}  "
              f"F1={row['f1_macro_mean']:.4f}  MCC={row['mcc_mean']:.4f}  "
              f"family={row['learning_family']}")

    # ── Select top 4 (or use forced selection) ──
    if forced_selection is not None:
        selected = forced_selection
        print(f"\n  Using LOCKED base models from 27-feature run (strict conditions):")
    else:
        selected = []
        families_used = set()
        for _, row in baseline_df.iterrows():
            if len(selected) >= TOP_K:
                break
            selected.append(row["model"])
            families_used.add(row["learning_family"])

        if len(families_used) < 3:
            print("\n  WARNING: Fewer than 3 learning families.")
        else:
            print(f"\n  Diversity check passed: {len(families_used)} learning families")

        print(f"\n  Selected top {TOP_K} base models:")

    for name in selected:
        print(f"    - {name} ({learning_families[name]})")

    ensemble_models = {name: all_models[name] for name in selected}

    # Use forced weights from 27-feature run, or derive from this run
    if forced_weights is not None:
        composite_scores = forced_weights
        print(f"\n  Using LOCKED weighted performance score weights from 27-feature run")
    else:
        composite_scores = {row["model"]: row["composite_score"]
                            for _, row in baseline_df.iterrows()}

    # ── STAGE 2: Ensemble voting ──
    print(f"\n{'=' * 60}")
    print("STAGE 2: Ensemble voting strategies")
    print(f"{'=' * 60}")

    model_names = list(ensemble_models.keys())
    model_weights = [composite_scores[name] for name in model_names]

    print(f"\nBase models: {model_names}")
    print(f"Static weights: {[f'{w:.4f}' for w in model_weights]}")

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

    # Collect ALL predictions across folds for per-class report
    all_y_true = []
    all_preds = {"Majority Vote": [], "Weighted Vote": [],
                 "Stacking": [], "Adaptive Vote": []}
    all_probs = {"Weighted Vote": [], "Stacking": [], "Adaptive Vote": []}

    adaptive_voter = AdaptiveVoter(n_models=len(model_names), n_classes=n_classes)
    disagreement_per_fold = []

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

        for m_idx, (name, model) in enumerate(ensemble_models.items()):
            clf = clone(model)
            clf.fit(X_train, y_train)

            pred = clf.predict(X_test)
            prob = get_probabilities(clf, X_test)

            base_preds.append(pred)
            base_probs.append(prob)

            model_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
            fold_f1_per_model.append(model_f1)
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
    print(f"RESULTS SUMMARY — {run_name}")
    print(f"{'=' * 60}")

    print(f"\nAvg pairwise disagreement: {np.mean(disagreement_per_fold):.4f} "
          f"({np.mean(disagreement_per_fold)*100:.1f}%)")

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
              f"F1={row['f1_macro_mean']:.6f}±{row['f1_macro_std']:.6f}  "
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

        # Confusion matrix
        cm = confusion_matrix(all_y_true_arr, preds_arr)
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        print("Confusion Matrix:")
        print(cm_df.to_string())
        print()

        # Save per-class report to CSV
        report_dict = classification_report(
            all_y_true_arr, preds_arr,
            target_names=class_names, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(
            f"ensemble_{strategy_name.lower().replace(' ', '_')}_perclass_{n_features}feat.csv"
        )

        # Save confusion matrix
        cm_df.to_csv(
            f"ensemble_{strategy_name.lower().replace(' ', '_')}_confusion_{n_features}feat.csv"
        )

    # ── Save summary CSVs ──
    suffix = f"_{n_features}feat"
    baseline_df.to_csv(f"ensemble_baseline_composite{suffix}.csv", index=False)
    summary_df.to_csv(f"ensemble_strategy_results{suffix}.csv", index=False)

    selection_info = pd.DataFrame([
        {"model": name, "learning_family": learning_families[name],
         "composite_score": composite_scores[name]}
        for name in model_names
    ])
    selection_info.to_csv(f"ensemble_selected_models{suffix}.csv", index=False)

    print(f"\nSaved CSVs with suffix '{suffix}':")
    print(f"  - ensemble_baseline_composite{suffix}.csv")
    print(f"  - ensemble_selected_models{suffix}.csv")
    print(f"  - ensemble_strategy_results{suffix}.csv")
    print(f"  - Per-class reports and confusion matrices for each strategy")

    return summary_df, baseline_df


# ============================================================
# RUN A: 27 features (full feature set)
# ============================================================
print("\n" + "#" * 70)
print("# PREPARING RUN A: 27 FEATURES (FULL)")
print("#" * 70)

df_27 = df_raw.copy()
X_27 = df_27.drop(columns=[TARGET_COL]).copy()
y_27 = df_27[TARGET_COL].copy()

for col in X_27.columns:
    X_27[col] = pd.to_numeric(X_27[col], errors="coerce")

label_encoder = LabelEncoder()
y_27_encoded = label_encoder.fit_transform(y_27)
n_classes = len(label_encoder.classes_)

print(f"Class labels: {list(label_encoder.classes_)}")
print(f"Feature count: {X_27.shape[1]}")

all_models_27 = build_all_models(X_27, y_27_encoded, N_SPLITS, RANDOM_STATE)
summary_27, baseline_27 = run_ensemble_evaluation(
    X_27, y_27_encoded, label_encoder, all_models_27,
    run_name="27 Features (Full)", n_features=27
)

# ── Extract top 4 selection and weights from Run A ──
top4_from_27 = baseline_27.sort_values("composite_score", ascending=False).head(TOP_K)
locked_selection = top4_from_27["model"].tolist()
locked_weights = dict(zip(top4_from_27["model"], top4_from_27["composite_score"]))

print(f"\n{'=' * 60}")
print("LOCKING top 4 models from 27-feature run for 22-feature run:")
for name in locked_selection:
    print(f"  - {name} ({learning_families[name]})  weight={locked_weights[name]:.4f}")
print(f"{'=' * 60}")


# ============================================================
# RUN B: 22 features (TCP flag features removed)
# ============================================================
print("\n\n" + "#" * 70)
print("# PREPARING RUN B: 22 FEATURES (FLAGS REMOVED)")
print("#" * 70)

# TCP flag counts trivially separate attack classes since each
# attack type sets a single distinctive flag. Removing them forces
# models to rely on flow-level statistical patterns (packet rates,
# byte ratios, inter-arrival times), which better reflects real-
# world detection conditions where attackers may manipulate
# protocol headers.
flag_cols = [c for c in ["syn_count", "ack_count", "fin_count",
                          "rst_count", "psh_count"] if c in df_raw.columns]

df_22 = df_raw.drop(columns=flag_cols).copy()
X_22 = df_22.drop(columns=[TARGET_COL]).copy()
y_22 = df_22[TARGET_COL].copy()

for col in X_22.columns:
    X_22[col] = pd.to_numeric(X_22[col], errors="coerce")

y_22_encoded = label_encoder.transform(y_22)

print(f"Dropped flag features: {flag_cols}")
print(f"Feature count: {X_22.shape[1]}")

# Build all 8 models for 22-feature baseline scoring,
# but force the same top 4 and weights from the 27-feature run
all_models_22 = build_all_models(X_22, y_22_encoded, N_SPLITS, RANDOM_STATE)
summary_22, baseline_22 = run_ensemble_evaluation(
    X_22, y_22_encoded, label_encoder, all_models_22,
    run_name="22 Features (Flags Removed)", n_features=22,
    forced_selection=locked_selection,
    forced_weights=locked_weights
)


# ============================================================
# Final comparison
# ============================================================
print("\n\n" + "=" * 70)
print("FINAL COMPARISON: 27 vs 22 FEATURES")
print("=" * 70)

print(f"\nLocked base models: {locked_selection}")

print("\n27 Features:")
for _, row in summary_27.iterrows():
    print(f"  {row['strategy']:20s}  F1={row['f1_macro_mean']:.6f}")

print("\n22 Features:")
for _, row in summary_22.iterrows():
    print(f"  {row['strategy']:20s}  F1={row['f1_macro_mean']:.6f}")


# ============================================================
# Cross-run delta table
# ============================================================
print("\n" + "=" * 70)
print("CROSS-RUN DELTA TABLE: 27-Feature → 22-Feature")
print("=" * 70)

# Merge the two summaries on strategy name
merged = summary_27.merge(summary_22, on="strategy", suffixes=("_27", "_22"))

delta_rows = []
for _, row in merged.iterrows():
    delta_rows.append({
        "Strategy": row["strategy"],
        "F1_27": row["f1_macro_mean_27"],
        "F1_22": row["f1_macro_mean_22"],
        "F1_Delta": row["f1_macro_mean_22"] - row["f1_macro_mean_27"],
        "MCC_27": row["mcc_mean_27"],
        "MCC_22": row["mcc_mean_22"],
        "MCC_Delta": row["mcc_mean_22"] - row["mcc_mean_27"],
        "AUC_27": row["auc_macro_mean_27"],
        "AUC_22": row["auc_macro_mean_22"],
        "AUC_Delta": row["auc_macro_mean_22"] - row["auc_macro_mean_27"],
        "Acc_27": row["accuracy_mean_27"],
        "Acc_22": row["accuracy_mean_22"],
        "Acc_Delta": row["accuracy_mean_22"] - row["accuracy_mean_27"],
    })

delta_df = pd.DataFrame(delta_rows)

# Print formatted table
print(f"\n{'Strategy':20s}  {'F1(27)':>10s}  {'F1(22)':>10s}  {'ΔF1':>10s}  "
      f"{'MCC(27)':>10s}  {'MCC(22)':>10s}  {'ΔMCC':>10s}  "
      f"{'AUC(27)':>10s}  {'AUC(22)':>10s}  {'ΔAUC':>10s}")
print("-" * 130)
for _, row in delta_df.iterrows():
    print(f"  {row['Strategy']:18s}  {row['F1_27']:>10.6f}  {row['F1_22']:>10.6f}  {row['F1_Delta']:>+10.6f}  "
          f"{row['MCC_27']:>10.6f}  {row['MCC_22']:>10.6f}  {row['MCC_Delta']:>+10.6f}  "
          f"{row['AUC_27']:>10.6f}  {row['AUC_22']:>10.6f}  {row['AUC_Delta']:>+10.6f}")

# Identify most robust strategy (smallest F1 drop)
delta_df["F1_Drop"] = abs(delta_df["F1_Delta"])
most_robust = delta_df.loc[delta_df["F1_Drop"].idxmin(), "Strategy"]
least_robust = delta_df.loc[delta_df["F1_Drop"].idxmax(), "Strategy"]

print(f"\nMost robust to feature removal:  {most_robust} "
      f"(ΔF1 = {delta_df.loc[delta_df['F1_Drop'].idxmin(), 'F1_Delta']:+.6f})")
print(f"Least robust to feature removal: {least_robust} "
      f"(ΔF1 = {delta_df.loc[delta_df['F1_Drop'].idxmax(), 'F1_Delta']:+.6f})")

# Save delta table
delta_df.to_csv("ensemble_cross_run_delta.csv", index=False)
print("\nSaved: ensemble_cross_run_delta.csv")

print("\n" + "=" * 70)
print("DONE — All results saved")
print("=" * 70)
