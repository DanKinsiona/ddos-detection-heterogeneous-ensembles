# A Comparative Evaluation of Heterogeneous Ensemble Voting and Stacking Strategies for Multi-Class Flow-Based DDoS Detection

**Author:** Dan Kinsiona  (Student ID: C2027282)
**Supervisor:** Dr John Haggerty
**Department of Computing, Sheffield Hallam University**

This repository contains the code, datasets and reproducibility scripts for the final-year dissertation submitted to Sheffield Hallam University.

---

## Research Question

> Which heterogeneous ensemble strategy is most effective for flow-based multi-class DDoS detection?

The hypothesis is that no single strategy will dominate across all metrics and that **stability over victory** will determine which strategy is most appropriate for production environments handling nonlinear traffic.

---

## Method Summary

- **Nine balanced traffic classes** generated locally — hping3-generated attacks for eight DDoS types (SYN, UDP, ICMP, ACK, SYN-ACK, RST, FIN, PSH-ACK) and iperf for benign traffic.
- **Scapy-based flow extractor** produces bidirectional flow records at 11,111 flows per class.
- **Eight base classifiers** were evaluated under five-fold cross-validation: Decision Tree, Random Forest, Gradient Boosting, Logistic Regression, KNN, SVM, MLP, Gaussian Naive Bayes.
- **Top four** were selected by a **weighted performance score** across F1, MCC, Recall, Precision and Accuracy → Decision Tree, Logistic Regression, SVM, MLP (one per learning family for maximum diversity).
- **Four ensemble strategies** compared head-to-head: Majority Voting, Weighted Voting, Adaptive Voting and Stacking (logistic regression meta-learner).
- **Validation:** five-fold stratified cross-validation, McNemar paired test, Azure ML compute (UK South).

---

## Experimental Runs

| Run | Features | Flows  | Notes                          |
| --- | -------- | ------ | ------------------------------ |
| A   | 27       | 99,999  | Full feature set               |
| B   | 22       | 99,999  | TCP flags removed              |
| C   | 27       | 499,995 | 5× scale of Run A              |
| D   | 22       | 499,995 | 5× scale of Run B (flags removed) |

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── runA_27feat_99999flows.csv       # Run A — 27 features, 99,999 flows
│   ├── runB_22feat_99999flows.csv       # Run B — 22 features, 99,999 flows
│   ├── runC_27feat_499995flows.csv      # Run C — 27 features, 499,995 flows
│   └── runD_22feat_499995flows.csv      # Run D — 22 features, 499,995 flows
└── scripts/
    ├── runA_and_runB.py                  # Reproduces Run A + Run B
    ├── runB_with_base_predictions.py     # Run B with base-classifier predictions saved for disagreement analysis
    └── runC_and_runD.py                  # Reproduces Run C + Run D
```

---

## How to Reproduce

### 1. Set up environment

The experiments were run on **Azure Machine Learning compute (UK South region)** for reproducibility and resource consistency.

**Option A — Reproduce on Azure ML (original setup):**
1. Create or open an Azure Machine Learning workspace in the UK South region.
2. Create a compute instance (CPU-based; experiments do not require GPU).
3. Upload the four CSV files and three Python scripts to the workspace File Share.
4. Open a terminal on the compute instance and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

**Option B — Reproduce locally:**
```bash
python -m venv .venv
source .venv/bin/activate     # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Results will match Azure within numerical precision because every random source (`StratifiedKFold`, model `random_state`, meta-learner `random_state`) is fixed to `42`.

### 2. Place CSV files

All four CSVs must sit in the same working directory as the scripts (or update the `DATA_PATH` constants at the top of each script).

### 3. Run the experiments

| To reproduce         | Run                                            |
| -------------------- | ---------------------------------------------- |
| Run A + Run B        | `python runA_and_runB.py`                      |
| Run B (with base predictions for disagreement analysis) | `python runB_with_base_predictions.py` |
| Run C + Run D        | `python runC_and_runD.py`                      |

### 4. What each script does

- **`runA_and_runB.py`** — Evaluates **all eight baseline models**, ranks them by weighted performance score, selects the top four, then runs the four ensemble strategies on Run A (27 features) followed by Run B (22 features after dropping the five TCP flag count columns).
- **`runB_with_base_predictions.py`** — Re-runs Run B under the same locked top-four base classifiers but additionally saves per-fold base-classifier predictions. These feed the pairwise disagreement metric in the dissertation discussion section.
- **`runC_and_runD.py`** — Stress-tests the locked top-four base classifiers at 5× scale (499,995 flows). Run C mirrors Run A's feature set; Run D mirrors Run B's feature set.

Each script writes its results CSVs to the working directory:
- `ensemble_strategy_results_*.csv`  (per-strategy F1, MCC, Recall, Precision, Accuracy)
- `ensemble_disagreement_*.csv`      (per-fold pairwise disagreement)
- `ensemble_selected_models_*.csv`   (top-four selection)
- `ensemble_<strategy>_perclass_*.csv` and `ensemble_<strategy>_confusion_*.csv`
- `predictions_full_*.csv` (Run B re-run only — base + ensemble predictions for downstream McNemar / disagreement)

### 5. Locked experimental conditions

To preserve a fair head-to-head comparison, every run uses:
- **Top four base models** (DT, LR, SVM, MLP) — fixed from the Run A weighted performance score ranking
- **Static weights** 0.9641 for all four (from Run A's weighted performance score)
- **Weighted performance score formula:** 0.35 F1 + 0.30 MCC + 0.20 Recall + 0.10 Precision + 0.05 Accuracy
- **5-fold StratifiedKFold** (shuffled, `random_state = 42`)
- **`random_state = 42`** everywhere (model training, CV splits, meta-learner)
- **Meta-learner:** Logistic Regression (Stacking)
- **Identical hyperparameters** across Runs A/B/C/D

---

## Hyperparameters

All hyperparameters are locked across Runs A, B, C and D. Only Naive Bayes is tuned (during the baseline selection stage in `runA_and_runB.py`) via `GridSearchCV` on `var_smoothing`.

### Base classifiers (all 8 evaluated in the baseline ranking)

| Model | Hyperparameters |
| --- | --- |
| **Decision Tree** | `criterion="gini"`, `max_depth=7`, `splitter="best"`, `min_samples_split=24`, `min_samples_leaf=10`, `random_state=42` |
| **Random Forest** | `n_estimators=300`, `max_depth=30`, `criterion="entropy"`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features="sqrt"`, `random_state=42`, `n_jobs=-1` |
| **Gradient Boosting** | scikit-learn defaults, `random_state=42` |
| **Logistic Regression** | `max_iter=1000`, `random_state=42`, preceded by `SimpleImputer(strategy="median")` + `StandardScaler` |
| **KNN** | scikit-learn defaults (k=5, Minkowski metric), preceded by `SimpleImputer(strategy="median")` + `StandardScaler` |
| **SVM** | `kernel="rbf"`, `C=1.0`, `gamma="scale"`, `tol=1e-4`, `probability=True`, `random_state=42`, preceded by `SimpleImputer(strategy="median")` + `StandardScaler` |
| **MLP** | `hidden_layer_sizes=(32,)`, `activation="relu"`, `solver="adam"`, `alpha=0.0001`, `learning_rate_init=0.001`, `max_iter=300`, `random_state=42`, preceded by `SimpleImputer(strategy="median")` + `StandardScaler` |
| **Naive Bayes** | `GaussianNB(var_smoothing=<tuned>)`, preceded by `SimpleImputer(strategy="median")` + `PowerTransformer(method="yeo-johnson", standardize=True)`. `var_smoothing` is tuned via `GridSearchCV` over `np.logspace(0, -9, num=100)` with `scoring="f1_macro"` on 5-fold stratified CV |

### Top four locked for ensembles

From the Run A weighted performance score ranking, the top four (one per learning family) are locked across all runs:

- Decision Tree (rule-based family)
- Logistic Regression (linear family)
- SVM (margin-based family)
- MLP (neural family)

All four scored **0.9641** on the weighted performance score in Run A and therefore receive equal static weights of 0.9641 in Weighted Voting and Stacking.

### Ensemble strategies

| Strategy | Mechanism |
| --- | --- |
| **Majority (Hard) Voting** | Equal votes per base classifier, hard labels, tie-broken by class index |
| **Weighted Voting** | Static weights (0.9641 for each) applied to predicted class probabilities |
| **Stacking** | Out-of-fold predicted probabilities from the four base classifiers feed a Logistic Regression meta-learner (`max_iter=1000`, `multi_class="multinomial"`, `solver="lbfgs"`, `random_state=42`) |
| **Adaptive Voting** | Per-fold dynamic weights set from each base classifier's macro F1 on the previous fold; initialised uniformly |

### Cross-validation and statistical testing

- **CV:** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for all runs
- **Stacking inner CV:** out-of-fold predictions on the training fold use a nested 5-fold split to avoid leakage into the meta-learner training set
- **Significance test:** McNemar paired test (with continuity correction) on pairs of strategy predictions per fold, then aggregated
- **Metrics:** F1 macro, MCC, Recall macro, Precision macro, Accuracy (per-fold mean and std reported)

---

## Key Findings

- **Run A — heterogeneity collapse at 0.0% pairwise disagreement.** With the full feature set, the four base classifiers all agreed; ensembles cannot outperform a single model when they have nothing to combine.
- **Run B — disagreement forced to 39.01%** by removing five TCP-flag features. Stacking led the disagreement test (macro-F1 = 0.4219) and maintained 0.9634 on the agreement runs.
- **Run C — scale validation.** 499,995 flows did not change rankings. Stacking F1 std = 0.000235 vs voters 0.000307 (24% less variance under heavier workloads).
- **Run D — disagreement confirmed at 37.78%** at scale, confirming feature reduction (not data volume) drives heterogeneity.
- **Stacking is most effective on stability** — smallest cross-run |ΔF1| of 0.5415, confirming **stability over victory** as a deployable principle for production environments handling nonlinear traffic.

McNemar paired test confirmed differences are statistically significant (χ² ≈ 53,000, p < 0.001) for all four ensemble strategies.

---

## License

This code is released for academic and reproducibility purposes accompanying the dissertation submission.
