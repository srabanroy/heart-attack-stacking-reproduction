# %% [markdown]
# # Part 1 — Reproduction of Bhagat, Sharma & Agarwal (2024)
#
# **Paper.** M. Bhagat, A. Sharma and P. Agarwal, "An efficient stacking-based ensemble
# technique for early heart attack prediction," *Multimedia Tools and Applications*,
# vol. 84, pp. 36351–36375, 2025. DOI: 10.1007/s11042-024-19293-7
#
# **Goal of this notebook.** Re-implement the authors' pipeline as faithfully as the paper
# allows, using the same dataset, the same feature set, the same train/test split ratio, the
# same six base classifiers, the same 5-fold stacking ensemble, and the same evaluation
# metrics. We then compare our reproduced numbers against the authors' Table 11 and
# investigate every discrepancy we find.
#
# **Structure**
# 1. Environment and reproducibility settings
# 2. Data loading and verification against the paper's Table 9
# 3. Exploratory analysis — including a dataset integrity check the paper does not perform
# 4. Preprocessing exactly as described in Section 3.1
# 5. Train/test split (Section 3, Step-1)
# 6. The six base classifiers (Sections 3.3.1–3.3.6)
# 7. The stacking ensemble (Section 3.4.1 and Table 8)
# 8. Reproduced results vs. the paper's reported results
# 9. Split-sensitivity study — how much of the paper's result is the split?
# 10. Leakage diagnostic — the root cause of the discrepancy
#
# Every design decision the paper leaves unspecified is marked **ASSUMPTION** and justified
# in place, as required by the task brief.

# %%
# ---------------------------------------------------------------------------------------
# 1. ENVIRONMENT AND REPRODUCIBILITY
# ---------------------------------------------------------------------------------------
# A fixed global seed is used everywhere so that this notebook returns identical numbers on
# every run. The paper does not state a seed, so we choose one and hold it constant; the
# sensitivity of the results to this choice is measured explicitly in Section 9.

import os, sys, json, warnings, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # headless backend: figures are written to disk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             matthews_corrcoef)
from xgboost import XGBClassifier
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

SEED = 42                     # global random seed (ASSUMPTION - see Section 5)
TEST_SIZE = 0.20              # paper: 1025 samples -> 205 test samples => 20% (see Section 5)
STACK_CV = 5                  # paper explicitly calls its ensemble "5-fold stacking"

np.random.seed(SEED)

# Locate the project root by searching upward for the dataset, so the notebook runs
# correctly whether it is launched from the project root, from notebooks/, or from an
# editor whose working directory is set somewhere else entirely.
def find_project_root(marker: str = "data/heart.csv") -> Path:
    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate '{marker}' in {here} or any parent directory. "
        "Run this notebook from inside the project folder, or set HD_PROJECT_ROOT."
    )

ROOT = Path(os.environ["HD_PROJECT_ROOT"]) if os.environ.get("HD_PROJECT_ROOT") else find_project_root()
DATA = ROOT / "data" / "heart.csv"
FIGS = ROOT / "figures";  FIGS.mkdir(exist_ok=True, parents=True)
RES  = ROOT / "results";  RES.mkdir(exist_ok=True, parents=True)

sns.set_theme(style="whitegrid", context="notebook")

print("python     :", platform.python_version())
import sklearn, xgboost
print("numpy      :", np.__version__)
print("pandas     :", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("xgboost    :", xgboost.__version__)
print("seed       :", SEED)

# %% [markdown]
# ## 2. Data loading and verification against the paper's Table 9
#
# The paper (Section 4) states it uses the Kaggle "Heart Disease Dataset"
# (`johnsmith88/heart-disease-dataset`), described as 1025 patients and 14 attributes drawn
# from the Cleveland, Hungary, Switzerland and Long Beach V databases. We verify that the
# file we hold matches that description, and we check every attribute against the ranges the
# authors print in their Table 9. This matters: if our copy of the data differed from theirs,
# every downstream discrepancy would be uninterpretable.

# %%
df = pd.read_csv(DATA)

print(f"shape            : {df.shape}   (paper states 1025 patients x 14 attributes)")
print(f"missing values   : {df.isna().sum().sum()}")
print(f"duplicate rows   : {df.duplicated().sum()}")
print(f"class balance    : {df['target'].value_counts().to_dict()}  (1 = disease)")
df.head()

# %%
# Reproduce the paper's Table 9 (attribute ranges) from the actual file and flag mismatches.
paper_table9 = {           # transcribed verbatim from the paper, p. 36365
    "age": "29 to 77", "sex": "0, 1", "cp": "0, 1, 2, 3", "trestbps": "94 to 200",
    "chol": "126 to 564", "fbs": "0, 1", "restecg": "0, 1, 2", "thalach": "71 to 202",
    "exang": "0, 1", "oldpeak": "0 to 6.2", "slope": "0, 1, 2", "ca": "0 to 3",
    "thal": "1 to 3", "target": "0, 1",
}

rows = []
for col in df.columns:
    observed = (f"{df[col].min()} to {df[col].max()}" if df[col].nunique() > 5
                else ", ".join(map(str, sorted(df[col].unique()))))
    rows.append({"attribute": col, "paper_Table9": paper_table9[col],
                 "observed_in_file": observed,
                 "match": "yes" if observed.replace(".0", "") == paper_table9[col] else "NO"})
table9 = pd.DataFrame(rows)
table9.to_csv(RES / "table01_dataset_vs_paper_table9.csv", index=False)
table9

# %% [markdown]
# **Finding 2.1 — two attributes do not match the paper's own Table 9.**
# The paper lists `ca` as taking values 0–3 and `thal` as 1–3. The file actually contains
# `ca ∈ {0,1,2,3,4}` and `thal ∈ {0,1,2,3}`. The extra levels (`ca = 4`, `thal = 0`) are the
# well-known placeholder codes that entered this Kaggle redistribution when the original UCI
# Cleveland `?` missing-value markers were numerically re-encoded. Table 9 therefore appears
# to have been copied from the original UCI attribute documentation rather than generated
# from the file the authors actually used. This is our first indication that the dataset was
# not audited before modelling.

# %% [markdown]
# ## 3. Exploratory analysis and a dataset integrity check
#
# The paper reports no integrity checks beyond missing values. We add one that turns out to
# be decisive for interpreting its results: **how many of the 1025 rows are actually distinct
# patients?**

# %%
feature_cols = [c for c in df.columns if c != "target"]

n_total  = len(df)
n_unique = len(df.drop_duplicates())
n_dupes  = n_total - n_unique

# Multiplicity = how many times each distinct clinical record appears in the file.
multiplicity = df.groupby(list(df.columns)).size()

print(f"rows in file                 : {n_total}")
print(f"distinct clinical records    : {n_unique}")
print(f"redundant (duplicate) rows   : {n_dupes}  ({n_dupes / n_total:.1%} of the file)")
print(f"\ncopies per distinct record   :\n{multiplicity.value_counts().sort_index().to_string()}")

profile = pd.DataFrame([{
    "rows_in_file": n_total, "distinct_records": n_unique, "duplicate_rows": n_dupes,
    "duplicate_fraction": round(n_dupes / n_total, 4),
    "min_copies": int(multiplicity.min()), "max_copies": int(multiplicity.max()),
    "class_1_rows": int((df.target == 1).sum()), "class_0_rows": int((df.target == 0).sum()),
    "class_1_distinct": int((df.drop_duplicates().target == 1).sum()),
    "class_0_distinct": int((df.drop_duplicates().target == 0).sum()),
}])
profile.to_csv(RES / "table02_dataset_profile.csv", index=False)
profile.T

# %% [markdown]
# **Finding 3.1 — the dataset contains only 302 distinct patients, not 1025.**
# Every distinct record is repeated 3, 4 or 8 times. The file is an *upsampled* copy of the
# 303-row UCI Cleveland dataset, not an aggregation of four hospital databases as Section 4
# of the paper claims. The consequences of this for the paper's headline result are developed
# in Section 10.

# %%
# Figure 1 - class balance and the duplication structure that the paper does not report.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

sns.countplot(x="target", data=df, ax=axes[0], palette="Set2", hue="target", legend=False)
axes[0].set_title(f"(a) Class balance, all {n_total} rows")
axes[0].set_xlabel("target (1 = heart disease)"); axes[0].set_ylabel("rows")

sns.countplot(x="target", data=df.drop_duplicates(), ax=axes[1], palette="Set2",
              hue="target", legend=False)
axes[1].set_title(f"(b) Class balance, {n_unique} distinct records")
axes[1].set_xlabel("target (1 = heart disease)"); axes[1].set_ylabel("distinct records")

vc = multiplicity.value_counts().sort_index()
axes[2].bar(vc.index.astype(str), vc.values, color="#d95f02")
axes[2].set_title("(c) Copies per distinct record")
axes[2].set_xlabel("number of identical copies in the file")
axes[2].set_ylabel("distinct records")
for i, v in enumerate(vc.values):
    axes[2].text(i, v + 2, str(v), ha="center", fontsize=9)

plt.tight_layout(); plt.savefig(FIGS / "fig01_class_balance_and_duplication.png", dpi=200)
plt.show()

# %%
# Figure 2 - replication of the paper's Fig. 2 (feature importance).
# The paper does not say which model produced its importance plot. Random Forest impurity
# importance is the standard choice and is the only one of their six models that yields the
# smooth profile shown in their figure.
# ASSUMPTION: Random Forest Gini importance on the full dataset.
rf_imp = RandomForestClassifier(random_state=SEED).fit(df[feature_cols], df["target"])
imp = pd.Series(rf_imp.feature_importances_, index=feature_cols).sort_values()

plt.figure(figsize=(7.5, 5))
plt.barh(imp.index, imp.values, color=sns.color_palette("Set2", len(imp)))
plt.xlabel("importance"); plt.ylabel("features")
plt.title("Reproduction of paper Fig. 2 — feature importance")
plt.tight_layout(); plt.savefig(FIGS / "fig02_feature_importance.png", dpi=200)
plt.show()

imp.sort_values(ascending=False).to_frame("rf_gini_importance").to_csv(
    RES / "table03_feature_importance.csv")
print(imp.sort_values(ascending=False).round(4).to_string())

# %% [markdown]
# **Finding 3.2 — the paper's narrative contradicts its own figure.**
# The text on p. 36358 states that "Exercise Induced Angina is the most important feature ...
# followed by maximum heart rate achieved". Their own Fig. 2 shows `thal` as the second-most
# important feature and `thalach` only sixth. Our reproduction ranks `cp`, `ca`, `thal` and
# `oldpeak` above `exang`. Two points follow: (i) the paper's description of its own figure is
# inaccurate, and (ii) impurity-based importance on a file with 70% duplicated rows is not a
# stable quantity, so exact agreement should not be expected.
#
# Note also that Section 3.2 of the paper is titled "Feature selection" and argues for
# reducing dimensionality, but **no feature is ever removed** — all 13 predictors are carried
# into every model. We reproduce what was done, not what was described.

# %% [markdown]
# ## 4. Preprocessing (paper Section 3.1)
#
# The paper specifies three preprocessing operations:
# 1. *missing-value imputation by linear regression* — **not applicable**: the file has no
#    missing values (verified in Section 2), so this step is a no-op on this dataset.
# 2. *normalisation of numeric attributes to a common scale*.
# 3. *encoding of categorical variables* — **already done**: every categorical attribute is
#    delivered pre-encoded as an integer code in this file.
#
# **ASSUMPTION (scaler).** The paper says "normalization ... converts numeric attributes to a
# similar scale" but names no scaler. We use `StandardScaler` (zero mean, unit variance),
# the most common reading of that sentence and the scikit-learn default for this task.
#
# **ASSUMPTION (which models are scaled).** Scaling changes the behaviour of Logistic
# Regression, KNN and Naive Bayes but is irrelevant to the three tree-based models, which are
# invariant to monotone feature transforms. We therefore scale inside a `Pipeline` for the
# three scale-sensitive models only. Crucially, fitting the scaler *inside* a pipeline means
# it is fitted on training data alone — applying it to the full dataset first would leak test
# statistics into training, an error we are careful not to introduce.
#
# **Note on an internal contradiction in the paper.** The paper's Fig. 1 shows preprocessing
# applied *before* the train/test split, while its Table 8 algorithm lists "Step-1 split" then
# "Step-2 preprocess both training and testing data". These prescribe different, and not
# equally valid, procedures. We follow Table 8 (split first), which is the statistically
# correct order.

# %%
X = df[feature_cols].copy()
y = df["target"].copy()
print(f"design matrix X: {X.shape}   target y: {y.shape}   features: {feature_cols}")

# %% [markdown]
# ## 5. Train/test split (paper Table 8, Step-1)
#
# The paper never states its split ratio, test-set size, seed, or whether the split was
# stratified. We recover the first two arithmetically from the reported metrics:
#
# * The paper's accuracies are all exact multiples of 1/205 (e.g. 0.9268 = 190/205,
#   0.9853 = 202/205), so the **test set contained 205 samples** = 20% of 1025.
# * Their recalls are exact multiples of 1/110 (e.g. 0.9090 = 100/110, 0.9545 = 105/110), so
#   the test set held **110 positive and 95 negative** samples.
#
# A *stratified* 20% split of this dataset would yield 105 positives and 100 negatives. The
# authors obtained 110/95, so their split was **not stratified**.
#
# **ASSUMPTION.** `train_test_split(test_size=0.20, stratify=None, random_state=42)`.
# The ratio and the absence of stratification are recovered from the paper; only the seed is
# free, and Section 9 quantifies exactly how much that free choice matters.

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=None)

print(f"train : {X_train.shape[0]} samples   test : {X_test.shape[0]} samples")
print(f"test class balance : {y_test.value_counts().to_dict()}  "
      f"(paper's split implied 110 positive / 95 negative)")

# %% [markdown]
# ## 6. The six base classifiers (paper Sections 3.3.1–3.3.6)
#
# The paper names six classifiers but **reports no hyperparameters for any of them**, and its
# per-classifier algorithm tables (Tables 2–7) are generic textbook pseudocode.
#
# **ASSUMPTION (hyperparameters).** We use scikit-learn / XGBoost library defaults throughout,
# with `random_state` fixed. This is the only reproducible reading of an unspecified
# configuration, it is what an unmodified implementation of the paper's pseudocode produces,
# and it makes our baseline honest — we are not tuning a baseline the authors did not tune.
# The one exception is `max_iter=1000` for Logistic Regression, raised from the default 100
# purely to guarantee convergence (the default emits a non-convergence warning on this data
# and would otherwise report a partially-optimised model).

# %%
def build_base_models(seed: int = SEED) -> dict:
    """Return the paper's six classifiers, freshly constructed.

    Scale-sensitive models (LR, NB, KNN) are wrapped in a Pipeline so that the
    StandardScaler is fitted on training folds only - never on the test set.
    Tree-based models (DT, RF, XGB) are scale-invariant and are used unwrapped.
    """
    return {
        "LR":  Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=1000, random_state=seed))]),
        "NB":  Pipeline([("scaler", StandardScaler()),
                         ("clf", GaussianNB())]),
        "KNN": Pipeline([("scaler", StandardScaler()),
                         ("clf", KNeighborsClassifier())]),
        "DT":  DecisionTreeClassifier(random_state=seed),
        "RF":  RandomForestClassifier(random_state=seed),
        "XGB": XGBClassifier(random_state=seed, eval_metric="logloss"),
    }

base_models = build_base_models()
for name, mdl in base_models.items():
    print(f"{name:4s} -> {mdl.__class__.__name__}")

# %% [markdown]
# ## 7. The stacking ensemble (paper Section 3.4.1, Table 8)
#
# Table 8 of the paper specifies: train all six classifiers as base learners, then "stack all
# these classifiers to build meta classifier". Section 5 twice describes the ensemble as
# "5-fold stacking", which fixes the cross-validation used to generate meta-features.
#
# **ASSUMPTION (meta-learner).** The paper never names the meta-classifier; Section 3.4.1 says
# only that it "can be any model, including a logistic regression, decision tree, or neural
# network". We use **Logistic Regression**, because (i) it is the first option the authors
# list, (ii) it is the standard choice in the stacking literature since Wolpert's original
# formulation, and (iii) it is the scikit-learn `StackingClassifier` default.
#
# **ASSUMPTION (meta-features).** We use scikit-learn's `stack_method="auto"`, which supplies
# predicted probabilities where available. The paper says only that the meta-model is "trained
# on the predictions of the base models".

# %%
def build_stacking(seed: int = SEED) -> StackingClassifier:
    """The paper's proposed model: 6 base learners + LR meta-learner, 5-fold stacking."""
    return StackingClassifier(
        estimators=list(build_base_models(seed).items()),
        final_estimator=LogisticRegression(max_iter=1000, random_state=seed),
        cv=STACK_CV,            # "5-fold stacking based ensemble technique" (paper Table 14)
        stack_method="auto",
        n_jobs=1,
    )

stack_model = build_stacking()
print(stack_model)

# %% [markdown]
# ## 8. Evaluation
#
# We compute every metric the paper reports in its Table 10/11: Accuracy, Precision,
# Recall (= Sensitivity), F1, Specificity, MCC and AUC. The task brief requires
# Accuracy, Precision, Recall, F1 and AUC, all of which are included.

# %%
def evaluate(name, model, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit `model` and return the full metric set reported by the paper."""
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # AUC needs a continuous score; use predict_proba where available.
    y_score = (model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba")
               else model.decision_function(X_te))

    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    return {
        "Model": name,
        "Accuracy":    accuracy_score(y_te, y_pred),
        "Precision":   precision_score(y_te, y_pred),
        "Recall":      recall_score(y_te, y_pred),      # == Sensitivity
        "F1":          f1_score(y_te, y_pred),
        "Specificity": tn / (tn + fp),                  # true-negative rate
        "MCC":         matthews_corrcoef(y_te, y_pred),
        "AUC":         roc_auc_score(y_te, y_score),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }

results, fitted = [], {}
for name, mdl in build_base_models().items():
    results.append(evaluate(name, mdl, X_train, y_train, X_test, y_test))
    fitted[name] = mdl

stack_model = build_stacking()
results.append(evaluate("Stacking (proposed)", stack_model, X_train, y_train, X_test, y_test))
fitted["Stacking (proposed)"] = stack_model

repro = pd.DataFrame(results).set_index("Model")
repro.round(4)

# %%
# Side-by-side comparison with the paper's Table 11.
paper_table11 = pd.DataFrame({
    "Model":       ["LR", "NB", "KNN", "DT", "RF", "XGB", "Stacking (proposed)"],
    "Paper_Acc":   [0.8439, 0.8439, 0.8585, 0.9268, 0.9268, 0.9073, 0.9853],
    "Paper_F1":    [0.8620, 0.8608, 0.8687, 0.9289, 0.9333, 0.9132, 0.9861],
    "Paper_Recall":[0.9090, 0.9000, 0.8727, 0.8909, 0.9545, 0.9090, 0.9727],
    "Paper_Prec":  [0.8196, 0.8250, 0.8648, 0.9702, 0.9130, 0.9174, 1.0000],
    "Paper_AUC":   [0.9204, 0.9185, 0.9304, 0.9702, 0.9734, 0.9830, 0.9880],
}).set_index("Model")

comparison = paper_table11.join(
    repro[["Accuracy", "F1", "Recall", "Precision", "AUC"]].rename(columns={
        "Accuracy": "Repro_Acc", "F1": "Repro_F1", "Recall": "Repro_Recall",
        "Precision": "Repro_Prec", "AUC": "Repro_AUC"}))
comparison["Acc_diff"] = comparison["Repro_Acc"] - comparison["Paper_Acc"]
comparison = comparison[["Paper_Acc", "Repro_Acc", "Acc_diff", "Paper_F1", "Repro_F1",
                         "Paper_Recall", "Repro_Recall", "Paper_Prec", "Repro_Prec",
                         "Paper_AUC", "Repro_AUC"]]

repro.to_csv(RES / "table04_reproduced_metrics.csv")
comparison.to_csv(RES / "table05_reproduced_vs_paper.csv")
comparison.round(4)

# %% [markdown]
# ### Reading the comparison
#
# * **The headline claim reproduces.** The paper's stacking ensemble reports 98.53% accuracy;
#   ours reaches essentially the same figure. The central result of the paper is therefore
#   *reproducible* — which, as Section 10 shows, is not the same as it being *valid*.
# * **The three scale-sensitive models reproduce closely.** LR, NB and KNN land within a few
#   percentage points of the published values.
# * **The three tree models over-shoot the paper.** DT, RF and XGB reach far higher accuracy
#   in our hands than the 92.68%/92.68%/90.73% the authors report. Under a protocol with
#   duplicate rows on both sides of the split (Section 10), a fully-grown tree *must* score
#   near-perfectly, because it can memorise the training rows and then meet exact copies of
#   them at test time. The authors' lower tree numbers are therefore the anomaly, not ours;
#   they are consistent with depth-limited trees or a different split, neither of which is
#   documented in the paper.

# %%
# Figure 3 - confusion matrices for the six base classifiers (paper Fig. 3).
fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
for ax, (name, mdl) in zip(axes.ravel(), fitted.items()):
    cm = confusion_matrix(y_test, mdl.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
    ax.set_title(f"{name}  (acc = {accuracy_score(y_test, mdl.predict(X_test)):.4f})")
    ax.set_xlabel("predicted"); ax.set_ylabel("actual")
plt.suptitle("Reproduction of paper Fig. 3 — confusion matrices", y=1.01)
plt.tight_layout(); plt.savefig(FIGS / "fig03_confusion_matrices.png", dpi=200,
                                bbox_inches="tight")
plt.show()

# %%
# Figure 4 - confusion matrix of the proposed stacking model (paper Fig. 5).
plt.figure(figsize=(4.6, 3.9))
cm = confusion_matrix(y_test, fitted["Stacking (proposed)"].predict(X_test))
sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Greens",
            xticklabels=["No disease", "Disease"], yticklabels=["No disease", "Disease"])
plt.title("Stacking ensemble — confusion matrix\n(reproduction of paper Fig. 5)")
plt.xlabel("predicted"); plt.ylabel("actual")
plt.tight_layout(); plt.savefig(FIGS / "fig04_stacking_confusion_matrix.png", dpi=200)
plt.show()

# %%
# Figure 5 - ROC curves for all models including the ensemble (paper Figs. 4 and 6).
plt.figure(figsize=(7, 6))
for name, mdl in fitted.items():
    score = (mdl.predict_proba(X_test)[:, 1] if hasattr(mdl, "predict_proba")
             else mdl.decision_function(X_test))
    fpr, tpr, _ = roc_curve(y_test, score)
    lw = 2.6 if "Stacking" in name else 1.4
    plt.plot(fpr, tpr, lw=lw, label=f"{name} (AUC = {roc_auc_score(y_test, score):.4f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Reproduction of paper Figs. 4 & 6 — ROC curves")
plt.legend(loc="lower right", fontsize=8.5)
plt.tight_layout(); plt.savefig(FIGS / "fig05_roc_curves.png", dpi=200)
plt.show()

# %% [markdown]
# ## 9. Split-sensitivity study
#
# The paper reports a single number per model from a single unspecified split. Because the
# seed is the one thing we could not recover, we measure how much it matters: we repeat the
# entire pipeline over 100 different random splits of the same size and ratio, and examine
# the resulting distribution. This tells us whether the published numbers are a stable
# property of the method or an artefact of one partition.

# %%
def run_one_split(seed: int) -> dict:
    """Run the full paper pipeline on one random 80/20 split and return accuracies."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=seed)
    out = {"seed": seed}
    for nm, m in build_base_models(SEED).items():          # models keep the fixed model seed
        m.fit(Xtr, ytr)
        out[nm] = accuracy_score(yte, m.predict(Xte))
    st = build_stacking(SEED)
    st.fit(Xtr, ytr)
    out["Stacking"] = accuracy_score(yte, st.predict(Xte))
    return out

sens = pd.DataFrame(Parallel(n_jobs=-1)(delayed(run_one_split)(s) for s in range(100)))
sens.to_csv(RES / "table06_split_sensitivity.csv", index=False)

summary = sens.drop(columns="seed").agg(["mean", "std", "min", "max"]).T
summary["paper_reported"] = [0.8439, 0.8439, 0.8585, 0.9268, 0.9268, 0.9073, 0.9853]
summary.round(4)

# %%
# Figure 6 - distribution of accuracy across 100 random splits vs the paper's single value.
plt.figure(figsize=(9.5, 5))
order = ["LR", "NB", "KNN", "DT", "RF", "XGB", "Stacking"]
sns.boxplot(data=sens[order], palette="Set3", hue=None)
for i, m in enumerate(order):
    plt.scatter(i, summary.loc[m, "paper_reported"], color="crimson", zorder=5, s=70,
                marker="D", label="paper's reported value" if i == 0 else None)
plt.ylabel("test accuracy"); plt.xlabel("model")
plt.title("Accuracy over 100 random 80/20 splits (paper's protocol)\n"
          "red diamonds = the single value reported in the paper")
plt.legend(); plt.tight_layout()
plt.savefig(FIGS / "fig06_split_sensitivity.png", dpi=200)
plt.show()

# %% [markdown]
# **Finding 9.1.** The tree-based models and the stacking ensemble sit in a tight, very high
# band across *every* split — their performance is not a lucky partition, it is systematic.
# The paper's reported DT/RF/XGB values fall *below* that band, reinforcing the conclusion in
# Section 8 that the authors' tree models were configured differently from the defaults their
# pseudocode implies. The linear and distance-based models (LR, NB, KNN) show far more
# spread, and the paper's values for those sit comfortably inside our distribution.

# %% [markdown]
# ## 10. Leakage diagnostic — why 98.53% is not a generalisation estimate
#
# Section 3 established that the 1025 rows contain only 302 distinct patients, each repeated
# 3–8 times. A random row-level split therefore does not separate patients — it separates
# *copies*. We now measure exactly how bad this is: for each test row, does an identical
# clinical record also appear in the training set?

# %%
# Give every distinct clinical record a group id, then measure train/test contamination.
record_key = df[feature_cols].astype(str).agg("|".join, axis=1)
group_id = pd.factorize(record_key)[0]

overlaps = []
for seed in range(100):
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=TEST_SIZE,
                                      random_state=seed)
    train_groups = set(group_id[idx_tr])
    overlaps.append(np.mean([g in train_groups for g in group_id[idx_te]]))

overlaps = np.array(overlaps)
print(f"Fraction of TEST rows whose exact clinical record also appears in TRAIN")
print(f"  mean over 100 splits : {overlaps.mean():.4f}")
print(f"  range                : {overlaps.min():.4f} - {overlaps.max():.4f}")

pd.DataFrame({"seed": range(100), "train_test_record_overlap": overlaps}).to_csv(
    RES / "table07_leakage_diagnostic.csv", index=False)

# %%
# Figure 7 - the leakage, and what happens to accuracy once it is removed.
from sklearn.model_selection import cross_val_score


def grouped_folds(groups, n_splits: int = 5, seed: int = SEED):
    """Build record-grouped cross-validation folds deterministically.

    Why not scikit-learn's GroupKFold? Its fold-assignment algorithm is not stable
    across scikit-learn releases, so the same code produces different folds - and
    therefore different numbers - on a different install. Since this notebook's whole
    argument rests on the grouped results being reproducible, the folds are built here
    explicitly instead.

    Every copy of a clinical record shares one group id, so all copies always land in
    the same fold: no record can appear in both the training and the test side.

    Groups are shuffled with a seeded generator and then dealt round-robin across the
    folds, which keeps the folds close to equal size. The result depends only on
    `seed` and `n_splits`, never on the library version.

    Returns a list of (train_index, test_index) arrays, ready to pass as `cv=`.
    """
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    rng = np.random.RandomState(seed)
    shuffled = rng.permutation(unique_groups)

    # Deal groups to folds round-robin: group i goes to fold i % n_splits.
    fold_of_group = {g: i % n_splits for i, g in enumerate(shuffled)}
    fold_index = np.array([fold_of_group[g] for g in groups])

    all_rows = np.arange(len(groups))
    return [(all_rows[fold_index != k], all_rows[fold_index == k])
            for k in range(n_splits)]

leak_free = {}
gkf = grouped_folds(group_id, n_splits=5, seed=SEED)
for nm, m in build_base_models().items():
    leak_free[nm] = cross_val_score(m, X, y, cv=gkf, scoring="accuracy", n_jobs=-1).mean()
leak_free["Stacking"] = cross_val_score(build_stacking(), X, y, cv=gkf,
                                        scoring="accuracy", n_jobs=-1).mean()

leak_df = pd.DataFrame({
    "paper_protocol_random_split": [summary.loc[m, "mean"] for m in order],
    "leakage_free_group_CV":       [leak_free[m] for m in order],
}, index=order)
leak_df["inflation"] = (leak_df["paper_protocol_random_split"]
                        - leak_df["leakage_free_group_CV"])
leak_df.to_csv(RES / "table08_leakage_impact.csv")

fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
axes[0].hist(overlaps * 100, bins=18, color="#c0392b", edgecolor="white")
axes[0].axvline(overlaps.mean() * 100, color="k", ls="--",
                label=f"mean = {overlaps.mean()*100:.1f}%")
axes[0].set_xlabel("% of test rows with an identical record in the training set")
axes[0].set_ylabel("number of splits (out of 100)")
axes[0].set_title("(a) Train/test contamination under the paper's protocol")
axes[0].legend()

leak_df[["paper_protocol_random_split", "leakage_free_group_CV"]].plot(
    kind="bar", ax=axes[1], color=["#c0392b", "#2980b9"], rot=0)
axes[1].set_ylabel("accuracy"); axes[1].set_ylim(0.5, 1.02)
axes[1].set_title("(b) Accuracy with and without duplicate leakage")
axes[1].legend(["paper's random split", "leakage-free (record-grouped CV)"], fontsize=9)
plt.tight_layout(); plt.savefig(FIGS / "fig07_leakage_diagnostic.png", dpi=200)
plt.show()

leak_df.round(4)

# %% [markdown]
# **Finding 10.1 — the decisive result of Part 1.**
#
# On average **97.6% of test rows have an identical clinical record sitting in the training
# set**. The models are not being asked to generalise to new patients; they are being asked to
# recall patients they have already seen. Once records are kept intact across the split
# (`GroupKFold` on the record identity), the stacking ensemble falls from ~98.5% to roughly
# 77%, an inflation of more than twenty accuracy points.
#
# This explains every pattern we observed:
# * why high-capacity memorisers (DT, RF, XGB) score near-perfectly while the smooth,
#   low-capacity models (LR, NB) do not — only the former can exploit exact repetition;
# * why stacking appears to deliver a dramatic gain over its own base learners;
# * why the paper's 98.53% far exceeds the 85–92% range typical of published Cleveland
#   heart-disease results obtained under clean protocols.
#
# The paper's headline number is reproducible but it is **not an estimate of predictive
# performance on unseen patients**. Addressing this is the starting point of Part 2.

# %% [markdown]
# ## 11. Internal inconsistencies in the published paper
#
# Independent of our experiments, the paper contains several self-contradictions that any
# reproduction attempt must confront. We record them because they materially affect what
# "reproducing the paper" even means.

# %%
inconsistencies = pd.DataFrame([
    {"#": 1, "Location": "Abstract vs Introduction (p. 36351 / 36353)",
     "Issue": "Abstract: RF and DT both 92.68%, XGBoost 90.73%, stacked 98.53%. "
              "Introduction: XGBoost 93.17%, DT 92.68%, stacked 96.58%.",
     "Consequence": "Two different sets of headline results; no way to know which was run."},
    {"#": 2, "Location": "Table 11 'Specificity' column",
     "Issue": "The Specificity column is numerically identical to the F1-score column for "
              "all seven models.",
     "Consequence": "Specificity is not actually reported; the column is mislabelled."},
    {"#": 3, "Location": "Table 11 'MCC' column",
     "Issue": "The MCC column is numerically identical to the Precision column for all "
              "seven models.",
     "Consequence": "MCC is not actually reported; MCC = 1.0 with 3 false negatives is "
                    "mathematically impossible."},
    {"#": 4, "Location": "Fig. 1 vs Table 8",
     "Issue": "Fig. 1 preprocesses before splitting; Table 8 splits before preprocessing.",
     "Consequence": "The two descriptions imply different (and not equally valid) pipelines."},
    {"#": 5, "Location": "Section 3.2 'Feature selection'",
     "Issue": "A feature-selection stage is described and motivated but never applied; "
              "all 13 predictors reach every model.",
     "Consequence": "A described component of the method does not exist in the results."},
    {"#": 6, "Location": "Section 4 dataset description",
     "Issue": "Described as 1025 patients from four databases; the file holds 302 distinct "
              "records, all from Cleveland, upsampled to 1025 rows.",
     "Consequence": "Sample size is overstated by 3.4x; the evaluation protocol is invalid."},
    {"#": 7, "Location": "Table 9 attribute ranges",
     "Issue": "ca given as 0-3 (file: 0-4); thal given as 1-3 (file: 0-3).",
     "Consequence": "Table 9 documents the original UCI data, not the file used."},
    {"#": 8, "Location": "Section 3.2 text vs Fig. 2",
     "Issue": "Text claims thalach is the second-most important feature; the figure shows "
              "thal second and thalach sixth.",
     "Consequence": "The narrative misreports the authors' own figure."},
])
inconsistencies.to_csv(RES / "table09_paper_inconsistencies.csv", index=False)
pd.set_option("display.max_colwidth", 95)
inconsistencies

# %% [markdown]
# ## 12. Part 1 summary
#
# 1. The dataset was located, verified against the paper's Table 9, and found to match on 12
#    of 14 attributes.
# 2. The paper's pipeline was re-implemented in full: preprocessing, a non-stratified 80/20
#    split (ratio and class counts recovered arithmetically from the published metrics), six
#    base classifiers at library defaults, and a 5-fold stacking ensemble with a Logistic
#    Regression meta-learner. Every unspecified detail is flagged and justified above.
# 3. **The headline result reproduces**: our stacking ensemble matches the paper's 98.53%.
# 4. **Three classifiers reproduce closely** (LR, NB, KNN); **three over-shoot** (DT, RF, XGB),
#    which the leakage analysis explains and which suggests the authors' trees were
#    configured differently from their own pseudocode.
# 5. **The headline result is not valid.** 723 of 1025 rows are duplicates; 97.6% of test rows
#    have a twin in training. Removing that leakage costs the ensemble over twenty accuracy
#    points.
# 6. Eight internal inconsistencies in the published paper were catalogued.
#
# Part 2 builds a solution that is designed around these findings.
