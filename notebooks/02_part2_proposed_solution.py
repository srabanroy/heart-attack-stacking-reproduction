# %% [markdown]
# # Part 2 — A leakage-aware, representation-matched stacking pipeline
#
# Part 1 reproduced Bhagat *et al.* (2024) and established that their headline result —
# 98.53% accuracy from a six-model stacking ensemble — is an artefact of duplicate-induced
# data leakage rather than a measure of predictive performance. This notebook designs,
# implements and evaluates a replacement pipeline built around that finding.
#
# ## Motivation: the limitations we address
#
# | # | Limitation of the original paper | Evidence from Part 1 | How we address it |
# |---|---|---|---|
# | L1 | **Duplicate leakage.** 723 of 1025 rows are duplicates; 97.6% of test rows have an identical record in training. | §10, Fig. 7 | Deduplicate to 302 distinct records (primary protocol); record-grouped CV on the full file (sensitivity check). |
# | L2 | **Single unrepeated hold-out split**, no variance estimate, no confidence interval. | §9, Fig. 6 | Repeated stratified 5-fold CV (5 repeats = 25 estimates) with paired significance testing. |
# | L3 | **No hyperparameter optimisation.** No parameters reported for any of the six models. | §6 | Nested cross-validation protects final hyperparameter tuning inside the training folds. Earlier representation and ensemble-composition choices remain preselected and exploratory. |
# | L4 | **Nominal attributes fed as integer codes** to linear and distance-based learners, imposing a false ordering on `cp`, `restecg`, `slope`, `thal`, `ca`. | §4 | Representation matching: one-hot for linear/kernel learners, native codes for tree learners. |
# | L5 | **Ensemble size mistaken for ensemble value.** Six correlated learners stacked without diversity analysis. | §8 | Explicit composition study; keep only complementary learners. |
# | L6 | **Fixed 0.5 decision threshold** despite an asymmetric clinical cost (a missed heart attack is far costlier than a false alarm). | §8 | Out-of-fold probability calibration assessment plus an explicit, clinically-motivated operating point. |
# | L7 | **Feature selection described but never performed.** | §3 | We do not claim a component we do not run. |
#
# ## Our proposed method in one sentence
#
# **RM-Stack**: a *representation-matched*, *leakage-aware* stacking ensemble in which each
# base learner receives the feature encoding appropriate to its inductive bias, the ensemble's
# membership is chosen by a documented diversity study rather than by including every model
# available, and the whole pipeline is assessed under record-grouped, repeated cross-validation
# with probabilities assessed for calibration and an explicitly chosen clinical operating point.
#
# This is a methodological contribution, not a change of classifier: the same family of
# algorithms is used, but the data handling, the representation, the ensemble design and the
# evaluation protocol are all rebuilt.

# %%
# ---------------------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------------------
import os, warnings, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              StackingClassifier)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_validate, cross_val_predict, GridSearchCV)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             fbeta_score, roc_auc_score, roc_curve, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve, brier_score_loss)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
SEED = 42
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
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True, parents=True)
RES  = ROOT / "results"; RES.mkdir(exist_ok=True, parents=True)
sns.set_theme(style="whitegrid", context="notebook")

print("python:", platform.python_version(), "| seed:", SEED)

# %% [markdown]
# ## 1. Leakage-aware data preparation (addresses L1)
#
# The paper treats the file's 1025 rows as 1025 patients. The file contains 302 distinct
# clinical records, each recorded between three and eight times, but no patient identifier.
# Two defensible protocols follow, and we run both:
#
# * **Primary — deduplication.** Collapse the file to its 302 distinct clinical records and
#   evaluate with ordinary stratified cross-validation. This is the honest sample: duplicated
#   rows carry no additional clinical information, they only reweight the loss and corrupt the
#   split.
# * **Secondary — record-grouped CV.** Keep all 1025 rows but constrain every fold so that all
#   copies of a record stay on the same side. This preserves the paper's dataset exactly while
#   removing the leakage, and serves as a sensitivity check that our conclusions do not depend
#   on the choice to deduplicate.

# %%
df_full = pd.read_csv(DATA)
FEATURES = [c for c in df_full.columns if c != "target"]

# Group id = identity of the distinct clinical record (used by the secondary protocol).
record_key = df_full[FEATURES].astype(str).agg("|".join, axis=1)
GROUPS_FULL = pd.factorize(record_key)[0]

# Primary protocol: the deduplicated sample.
df = df_full.drop_duplicates().reset_index(drop=True)
X, y = df[FEATURES], df["target"].values

print(f"file as published        : {df_full.shape[0]} rows")
print(f"distinct records (used)  : {X.shape[0]} rows x {X.shape[1]} features")
print(f"class balance            : {dict(pd.Series(y).value_counts())}  "
      f"(positive rate {y.mean():.3f})")

# %% [markdown]
# ## 2. Representation matching (addresses L4)
#
# The Cleveland attribute set mixes three kinds of variable, and the paper treats them all
# identically — as integers, optionally standardised:
#
# * **Continuous**: `age`, `trestbps`, `chol`, `thalach`, `oldpeak`
# * **Nominal** (unordered categories that happen to be coded as integers): `cp` (chest-pain
#   type), `restecg`, `slope`, `thal`, `ca`
# * **Binary**: `sex`, `fbs`, `exang`
#
# Handing `cp ∈ {0,1,2,3}` to a logistic regression as a single numeric column asserts that
# asymptomatic chest pain is "three times" typical angina and that the effect is linear in
# that code. It is not: these are unordered diagnostic categories. The same error affects KNN
# (Euclidean distance over arbitrary codes) and Gaussian Naive Bayes (a Gaussian density over
# a nominal code).
#
# Tree ensembles are less directly constrained by a single linear coefficient, but their
# threshold splits still depend on the arbitrary order of integer category codes. Whether
# one-hot expansion helps or harms them is therefore empirical and sample-dependent.
#
# Rather than assume, we measure. The study below runs every candidate learner under both
# encodings.

# %%
CONTINUOUS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
NOMINAL    = ["cp", "restecg", "slope", "thal", "ca"]
BINARY     = ["sex", "fbs", "exang"]

def onehot_prep() -> ColumnTransformer:
    """Representation for linear / kernel / distance learners.

    Continuous columns standardised; nominal columns one-hot expanded (first level
    dropped to avoid collinearity); binary columns passed through unchanged.
    Fitted inside a Pipeline, so it only ever sees training folds.
    """
    return ColumnTransformer([
        ("cont", StandardScaler(), CONTINUOUS),
        ("nom",  OneHotEncoder(handle_unknown="ignore", drop="first"), NOMINAL),
        ("bin",  "passthrough", BINARY),
    ])

def scaled_prep() -> StandardScaler:
    """The paper's representation: every column treated as a scaled number."""
    return StandardScaler()

# %%
# --- Encoding x learner study -----------------------------------------------------------
# A moderate screening protocol (5-fold x 3 repeats) with a DIFFERENT seed from the final
# evaluation, so that design decisions made here are not tuned to the reporting split.
SCREEN_CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=7)

candidates = {
    "LR (C=0.05)":  LogisticRegression(C=0.05, max_iter=5000, random_state=SEED),
    "LR (C=0.3)":   LogisticRegression(C=0.3,  max_iter=5000, random_state=SEED),
    "LR (C=1)":     LogisticRegression(C=1,    max_iter=5000, random_state=SEED),
    "SVM-RBF (C=1)":SVC(C=1, probability=True, random_state=SEED),
    "SVM-RBF (C=3)":SVC(C=3, probability=True, random_state=SEED),
    "KNN (k=9)":    KNeighborsClassifier(n_neighbors=9,  weights="distance"),
    "KNN (k=21)":   KNeighborsClassifier(n_neighbors=21, weights="distance"),
    "GaussianNB":   GaussianNB(),
    "RF (leaf=1)":  RandomForestClassifier(n_estimators=400, min_samples_leaf=1,
                                           random_state=SEED, n_jobs=1),
    "RF (leaf=3)":  RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                           random_state=SEED, n_jobs=1),
    "ExtraTrees (leaf=3)": ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                                random_state=SEED, n_jobs=1),
    "XGBoost (d=2)": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=2,
                                   subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                                   random_state=SEED, eval_metric="logloss", n_jobs=1),
}

rows = []
for name, est in candidates.items():
    for enc_name, prep in [("integer-coded (paper)", scaled_prep), ("one-hot (ours)", onehot_prep)]:
        pipe = Pipeline([("prep", prep()), ("clf", clone(est))])
        cvres = cross_validate(pipe, X, y, cv=SCREEN_CV,
                               scoring=["accuracy", "roc_auc"], n_jobs=-1)
        rows.append({"Learner": name, "Encoding": enc_name,
                     "Accuracy": cvres["test_accuracy"].mean(),
                     "AUC": cvres["test_roc_auc"].mean()})

enc_study = pd.DataFrame(rows)
enc_pivot = enc_study.pivot(index="Learner", columns="Encoding", values="AUC")
enc_pivot["gain_from_one_hot"] = enc_pivot["one-hot (ours)"] - enc_pivot["integer-coded (paper)"]
enc_study.to_csv(RES / "table10_encoding_study.csv", index=False)
enc_pivot.round(4).sort_values("gain_from_one_hot", ascending=False)

# %%
# Figure 8 - does one-hot encoding help, and for which learner families?
fig, ax = plt.subplots(figsize=(9.5, 5.2))
plot_df = enc_pivot.sort_values("gain_from_one_hot")
colors = ["#2980b9" if g > 0 else "#c0392b" for g in plot_df["gain_from_one_hot"]]
ax.barh(plot_df.index, plot_df["gain_from_one_hot"], color=colors)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("change in ROC-AUC when nominal attributes are one-hot encoded")
ax.set_title("Encoding is not a global choice — it depends on the learner's inductive bias\n"
             "blue = one-hot helps, red = one-hot hurts")
plt.tight_layout(); plt.savefig(FIGS / "fig08_encoding_study.png", dpi=200)
plt.show()

# %% [markdown]
# **Finding 2.1 — encoding must be matched to the learner, not chosen globally.**
# The effect of one-hot encoding depends on both learner family and hyperparameters. Its
# largest gain is for Logistic Regression (C=1, roughly two AUC points); some other linear,
# kernel and distance-based candidates improve only slightly or become worse. The tested
# tree ensembles are mostly worse under one-hot encoding, with XGBoost effectively unchanged.
# This empirical variation is why the paper's uniform treatment of all 13 columns as scaled
# integers is not a defensible global preprocessing rule.
#
# Our pipeline therefore gives **each base learner the representation suited to it** — a
# design choice that is invisible in the paper's flat "preprocess then classify" workflow.

# %% [markdown]
# ## 3. Ensemble composition study (addresses L5)
#
# The paper stacks all six classifiers it happens to have trained. Stacking theory says the
# meta-learner benefits from base learners whose *errors* are decorrelated, not from having
# many of them — and with only 302 samples, every additional base learner adds meta-features
# that the meta-learner must fit from the same small sample. We test this directly.

# %%
def make_member(kind: str):
    """Construct a base learner with its matched representation (per Finding 2.1)."""
    if kind == "lr_oh":                                    # linear: one-hot
        return Pipeline([("prep", onehot_prep()),
                         ("clf", LogisticRegression(C=1, max_iter=5000, random_state=SEED))])
    if kind == "svc_oh":                                   # kernel: one-hot
        return Pipeline([("prep", onehot_prep()),
                         ("clf", SVC(C=1, probability=True, random_state=SEED))])
    if kind == "knn":                                      # distance: scaled integer codes
        return Pipeline([("prep", scaled_prep()),
                         ("clf", KNeighborsClassifier(n_neighbors=21, weights="distance"))])
    if kind == "rf":                                       # trees: native codes, unscaled
        return RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                      random_state=SEED, n_jobs=1)
    if kind == "et":
        return ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                    random_state=SEED, n_jobs=1)
    if kind == "xgb":
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=2,
                             subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEED, eval_metric="logloss", n_jobs=1)
    raise ValueError(kind)

def make_stack(members, meta_C: float = 1.0) -> StackingClassifier:
    """Stack `members` with out-of-fold PROBABILITY meta-features and an L2 LR meta-learner.

    stack_method='predict_proba' passes continuous class probabilities to the meta-learner.
    Our reconstructed paper baseline also uses probabilities where available, so this is an
    implementation detail rather than a claimed difference from the baseline.
    """
    return StackingClassifier(
        estimators=[(m, make_member(m)) for m in members],
        final_estimator=LogisticRegression(C=meta_C, max_iter=5000, random_state=SEED),
        cv=5, stack_method="predict_proba", n_jobs=1,
    )

# %%
compositions = {
    "ExtraTrees alone (no stack)":        make_member("et"),
    "lr+et (2 members)":                  make_stack(["lr_oh", "et"]),
    "lr+et+xgb (3)":                      make_stack(["lr_oh", "et", "xgb"]),
    "lr+knn+et+xgb (4)":                  make_stack(["lr_oh", "knn", "et", "xgb"]),
    "lr+rf+et+xgb (4)":                   make_stack(["lr_oh", "rf", "et", "xgb"]),
    "lr+svc+knn+rf+et+xgb (6)":           make_stack(["lr_oh","svc_oh","knn","rf","et","xgb"]),
}

rows = []
for name, mdl in compositions.items():
    cvres = cross_validate(mdl, X, y, cv=SCREEN_CV, scoring=["accuracy", "roc_auc"], n_jobs=-1)
    rows.append({"Composition": name,
                 "n_members": 1 if "alone" in name else len(mdl.estimators),
                 "Accuracy": cvres["test_accuracy"].mean(),
                 "AUC": cvres["test_roc_auc"].mean()})
comp_study = pd.DataFrame(rows)
comp_study.to_csv(RES / "table11_composition_study.csv", index=False)
comp_study.round(4)

# %%
# Figure 9 - ensemble size vs performance.
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.plot(comp_study["n_members"], comp_study["AUC"], "o-", color="#2980b9", label="ROC-AUC")
ax.plot(comp_study["n_members"], comp_study["Accuracy"], "s--", color="#d95f02",
        label="Accuracy")
for _, r in comp_study.iterrows():
    ax.annotate(r["Composition"].split(" (")[0], (r["n_members"], r["AUC"]),
                textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
ax.set_xlabel("number of base learners in the stack")
ax.set_ylabel("cross-validated score")
ax.set_title("More base learners do not mean a better ensemble (n = 302)")
ax.legend(); plt.tight_layout()
plt.savefig(FIGS / "fig09_composition_study.png", dpi=200)
plt.show()

# %% [markdown]
# **Finding 3.1 — a compact, diverse stack beats the paper's six-model stack.**
# Every larger tested stack performs worse than the two-member stack. The best
# configuration pairs a regularised **logistic regression on one-hot features** (a smooth,
# high-bias, low-variance decision surface) with **extremely randomised trees on native codes**
# (a non-linear, low-bias, high-variance learner). The observed results are consistent with
# complementary error patterns, although this experiment does not directly estimate error
# correlation. The larger tested stacks add meta-features without improving held-out AUC.
#
# This is an evidence-based design decision, and it directly contradicts the paper's implicit
# assumption that stacking more models yields a better ensemble.

# %% [markdown]
# ## 4. The proposed model: RM-Stack
#
# Pulling the findings together:

# %%
def build_rm_stack() -> StackingClassifier:
    """RM-Stack - the proposed model.

    Members (each with its matched representation):
      * lr_oh : LogisticRegression(C=1) on [standardised continuous | one-hot nominal | binary]
      * et    : ExtraTreesClassifier(400 trees, min_samples_leaf=3) on native integer codes
    Meta-learner:
      * LogisticRegression(C=1) trained on 5-fold OUT-OF-FOLD predicted probabilities.

    Every transformation lives inside the pipeline, so nothing is fitted on held-out data.
    """
    return make_stack(["lr_oh", "et"], meta_C=1.0)

rm_stack = build_rm_stack()
print(rm_stack)

# %% [markdown]
# ### Comparator and interpretation boundary
#
# Both pipelines are evaluated on identical leakage-free folds, which controls the evaluation
# samples. However, this is not a fully symmetric development comparison: RM-Stack's
# representation, composition and hyperparameters were deliberately selected, whereas the
# reconstructed paper ensemble retains library-default settings where the paper was silent.
# The observed difference therefore compares the complete pipelines and cannot isolate the
# contribution of representation matching, ensemble composition or tuning.

# %%
def build_paper_stack() -> StackingClassifier:
    """The original paper's ensemble (Part 1), for head-to-head comparison."""
    base = {
        "LR":  Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "DT":  DecisionTreeClassifier(random_state=SEED),
        "RF":  RandomForestClassifier(random_state=SEED, n_jobs=1),
        "XGB": XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=1),
        "NB":  Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
        "KNN": Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
    }
    return StackingClassifier(list(base.items()),
                              LogisticRegression(max_iter=1000, random_state=SEED),
                              cv=5, n_jobs=1)

# %% [markdown]
# ## 5. Evaluation protocol (addresses L2)
#
# * **Repeated stratified 5-fold cross-validation**, 5 repeats → 25 held-out estimates per
#   model, with class proportions preserved in every fold.
# * **Identical partitions for both models** (same `random_state`), so fold-level differences
#   are paired. Folds are not independent because their training sets overlap.
# * The ordinary paired t-test and Wilcoxon signed-rank test are retained as descriptive
#   sensitivity checks. The Nadeau-Bengio corrected resampled t-test is the primary
#   inferential result, and Cohen's *d_z* describes effect size.

# %%
FINAL_SPLITS = 5
FINAL_REPEATS = 5
FINAL_CV = RepeatedStratifiedKFold(n_splits=FINAL_SPLITS, n_repeats=FINAL_REPEATS,
                                   random_state=SEED)
SCORING  = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = {}
for name, mdl in [("Paper stacking ensemble", build_paper_stack()),
                  ("RM-Stack (proposed)",     build_rm_stack())]:
    cv_results[name] = cross_validate(mdl, X, y, cv=FINAL_CV, scoring=SCORING, n_jobs=-1)
    print(f"{name} evaluated over {len(cv_results[name]['test_accuracy'])} folds")

summary = pd.DataFrame({
    name: {m: f"{r['test_'+m].mean():.4f} ± {r['test_'+m].std():.4f}" for m in SCORING}
    for name, r in cv_results.items()
}).T
summary.to_csv(RES / "table12_cv_summary.csv")
summary

# %%
# --- Paired and dependence-aware significance testing ------------------------------------
sig_rows = []
for metric in SCORING:
    a = cv_results["Paper stacking ensemble"]["test_" + metric]   # baseline
    b = cv_results["RM-Stack (proposed)"]["test_" + metric]       # proposed
    diff = b - a
    t_stat, p_t = stats.ttest_rel(b, a)
    w_stat, p_w = stats.wilcoxon(b, a)
    cohen_dz = diff.mean() / diff.std(ddof=1)

    # Nadeau-Bengio corrected resampled t-test. Repeated k-fold scores are correlated because
    # their training sets overlap. For 5-fold CV, n_test / n_train = 1/4.
    correction = (1 / len(diff)) + (1 / (FINAL_SPLITS - 1))
    corrected_se = np.sqrt(correction * diff.var(ddof=1))
    corrected_t = diff.mean() / corrected_se
    corrected_p = 2 * stats.t.sf(abs(corrected_t), df=len(diff) - 1)
    sig_rows.append({
        "Metric": metric, "Paper": a.mean(), "RM-Stack": b.mean(),
        "Difference": diff.mean(), "t": t_stat, "p (paired t)": p_t,
        "p (Wilcoxon)": p_w, "corrected t": corrected_t,
        "p (corrected t)": corrected_p, "Cohen dz": cohen_dz,
        "Significant (corrected a=0.05)": "yes" if corrected_p < 0.05 else "no",
    })
sig = pd.DataFrame(sig_rows)
sig.to_csv(RES / "table13_significance_tests.csv", index=False)
sig.round(4)

# %%
# Figure 10 - per-fold distributions and paired differences.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

long = pd.concat([
    pd.DataFrame({"score": cv_results[n]["test_accuracy"], "model": n} ) for n in cv_results
] + [
    pd.DataFrame({"score": cv_results[n]["test_roc_auc"], "model": n + "\n(AUC)"})
    for n in cv_results
])
acc_long = pd.DataFrame({
    "Accuracy": np.r_[cv_results["Paper stacking ensemble"]["test_accuracy"],
                      cv_results["RM-Stack (proposed)"]["test_accuracy"]],
    "AUC": np.r_[cv_results["Paper stacking ensemble"]["test_roc_auc"],
                 cv_results["RM-Stack (proposed)"]["test_roc_auc"]],
    "Model": ["Paper stack"] * 25 + ["RM-Stack"] * 25,
})
sns.boxplot(data=acc_long, x="Model", y="Accuracy", ax=axes[0], palette="Set2", hue="Model",
            legend=False)
sns.stripplot(data=acc_long, x="Model", y="Accuracy", ax=axes[0], color="k", alpha=.5, size=4)
axes[0].set_title("(a) Accuracy over 25 held-out folds")

d_acc = (cv_results["RM-Stack (proposed)"]["test_accuracy"]
         - cv_results["Paper stacking ensemble"]["test_accuracy"])
d_auc = (cv_results["RM-Stack (proposed)"]["test_roc_auc"]
         - cv_results["Paper stacking ensemble"]["test_roc_auc"])
axes[1].axhline(0, color="k", lw=1)
axes[1].plot(range(25), d_acc, "o-", label=f"Accuracy (mean {d_acc.mean():+.4f})", alpha=.8)
axes[1].plot(range(25), d_auc, "s-", label=f"ROC-AUC (mean {d_auc.mean():+.4f})", alpha=.8)
axes[1].set_xlabel("fold index"); axes[1].set_ylabel("RM-Stack minus Paper stack")
axes[1].set_title("(b) Paired per-fold differences (positive = RM-Stack better)")
axes[1].legend()
plt.tight_layout(); plt.savefig(FIGS / "fig10_proposed_vs_baseline.png", dpi=200)
plt.show()

# %% [markdown]
# ## 6. Nested hyperparameter tuning check (addresses L3)
#
# Sections 2 and 3 used the full 302-record development dataset to examine representation and
# ensemble structure, so the main comparison contains design-selection optimism. The nested
# analysis below protects the tuning of three RM-Stack hyperparameters: an inner loop selects
# them using training data only, and an outer loop scores the selected settings on unseen
# held-out data. It does not repeat representation or composition selection inside each outer
# training fold. In addition, RM-Stack is tuned while the reproduced baseline retains
# library-default settings. Consequently, this analysis estimates the performance of the
# selected RM-Stack pipeline but does not establish that representation matching or ensemble
# composition caused its advantage.

# %%
param_grid = {
    "lr_oh__clf__C":         [0.3, 1.0, 3.0],   # regularisation of the linear member
    "et__min_samples_leaf":  [2, 3],            # capacity of the tree member
    "final_estimator__C":    [0.5, 1.0],        # regularisation of the meta-learner
}

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

search = GridSearchCV(build_rm_stack(), param_grid, scoring="roc_auc",
                      cv=inner_cv, n_jobs=-1, refit=True)

nested = cross_validate(search, X, y, cv=outer_cv,
                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                        n_jobs=1, return_estimator=True)

nested_summary = pd.DataFrame({
    m: [nested["test_" + m].mean(), nested["test_" + m].std()]
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]
}, index=["mean", "std"]).T
nested_summary.to_csv(RES / "table14_nested_cv.csv")

print("Hyperparameters selected in each outer fold:")
for i, est in enumerate(nested["estimator"], 1):
    print(f"  fold {i}: {est.best_params_}")
print()
nested_summary.round(4)

# %% [markdown]
# **Finding 6.1.** The nested hyperparameter-tuning estimate is close to the repeated-CV
# estimate, and the inner loop chooses broadly similar settings. This supports stability of
# the tuned parameters, but it does not eliminate bias from choosing the representation and
# ensemble composition on the full dataset.

# %% [markdown]
# ## 7. Calibration and the clinical operating point (addresses L6)
#
# Accuracy at a 0.5 threshold is the wrong target for a screening test. Missing a patient who
# is about to have a heart attack (a false negative) is far costlier than referring a healthy
# patient for a further test (a false positive). The paper never examines this: it reports a
# single confusion matrix at the default threshold.
#
# We therefore (i) check whether our model's probabilities are *calibrated* — that a predicted
# 0.8 really means 80% — and (ii) choose an operating point explicitly, by maximising the
# F2-score, which weights recall four times as heavily as precision.

# %%
# Out-of-fold probabilities: every prediction comes from a model that never saw that patient.
oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
proba_rm    = cross_val_predict(build_rm_stack(),    X, y, cv=oof_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
proba_paper = cross_val_predict(build_paper_stack(), X, y, cv=oof_cv,
                                method="predict_proba", n_jobs=-1)[:, 1]

print(f"Brier score (lower is better)   RM-Stack: {brier_score_loss(y, proba_rm):.4f}   "
      f"Paper stack: {brier_score_loss(y, proba_paper):.4f}")

# %%
# Sweep the decision threshold and record the clinical trade-off.
thresholds = np.linspace(0.05, 0.95, 181)
sweep = pd.DataFrame([{
    "threshold": t,
    "accuracy":  accuracy_score(y, (proba_rm >= t).astype(int)),
    "precision": precision_score(y, (proba_rm >= t).astype(int), zero_division=0),
    "recall":    recall_score(y, (proba_rm >= t).astype(int)),
    "f1":        f1_score(y, (proba_rm >= t).astype(int)),
    "f2":        fbeta_score(y, (proba_rm >= t).astype(int), beta=2),
} for t in thresholds])

best_f2 = sweep.loc[sweep["f2"].idxmax()]
default = sweep.iloc[(sweep["threshold"] - 0.5).abs().idxmin()]

operating = pd.DataFrame([default, best_f2],
                         index=["default (0.50)", f"F2-optimal ({best_f2.threshold:.2f})"])
operating.to_csv(RES / "table15_operating_points.csv")
operating.round(4)

# %%
# Figure 11 - calibration, threshold sweep and the resulting confusion matrices.
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# (a) reliability diagram
for nm, p, c in [("RM-Stack", proba_rm, "#2980b9"), ("Paper stack", proba_paper, "#c0392b")]:
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
    axes[0].plot(mean_pred, frac_pos, "o-", color=c,
                 label=f"{nm} (Brier {brier_score_loss(y, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="perfectly calibrated")
axes[0].set_xlabel("mean predicted probability"); axes[0].set_ylabel("observed frequency")
axes[0].set_title("(a) Calibration (out-of-fold)"); axes[0].legend(fontsize=8)

# (b) threshold sweep
axes[1].plot(sweep.threshold, sweep.recall,    label="Recall (sensitivity)")
axes[1].plot(sweep.threshold, sweep.precision, label="Precision")
axes[1].plot(sweep.threshold, sweep.f1,        label="F1")
axes[1].plot(sweep.threshold, sweep.f2,        label="F2 (recall-weighted)", lw=2.5)
axes[1].axvline(0.5, color="grey", ls=":", label="paper's default 0.50")
axes[1].axvline(best_f2.threshold, color="crimson", ls="--",
                label=f"F2-optimal {best_f2.threshold:.2f}")
axes[1].set_xlabel("decision threshold"); axes[1].set_ylabel("score")
axes[1].set_title("(b) Choosing the operating point"); axes[1].legend(fontsize=8)

# (c) what that means in classified cases
cm_def  = confusion_matrix(y, (proba_rm >= 0.5).astype(int))
cm_best = confusion_matrix(y, (proba_rm >= best_f2.threshold).astype(int))
labels = np.array([[f"TN\n{cm_def[0,0]} -> {cm_best[0,0]}", f"FP\n{cm_def[0,1]} -> {cm_best[0,1]}"],
                   [f"FN\n{cm_def[1,0]} -> {cm_best[1,0]}", f"TP\n{cm_def[1,1]} -> {cm_best[1,1]}"]])
sns.heatmap(cm_best, annot=labels, fmt="", cmap="Greens", cbar=False, ax=axes[2],
            xticklabels=["pred. no disease", "pred. disease"],
            yticklabels=["no disease", "disease"])
axes[2].set_title("(c) Patients at 0.50 -> F2-optimal threshold")

plt.tight_layout(); plt.savefig(FIGS / "fig11_calibration_and_threshold.png", dpi=200)
plt.show()

print(f"Missed cases (false negatives) at threshold 0.50 : {cm_def[1,0]}")
print(f"Missed cases (false negatives) at F2-optimal     : {cm_best[1,0]}")

# %% [markdown]
# **Finding 7.1 — the operating point matters more than the last accuracy point.**
# Moving from the default threshold to the F2-optimal threshold cuts the number of missed
# disease cases dramatically, at the cost of additional false alarms. For a *screening*
# instrument whose output triggers a confirmatory test rather than a treatment decision, that
# is the correct trade — and it is a lever the original paper never touches. Note that overall
# accuracy *falls* at this operating point: an accuracy-only evaluation, as in the paper,
# would reject the clinically preferable model.

# %% [markdown]
# ## 8. Sensitivity check: the secondary (record-grouped) protocol
#
# Everything above used the deduplicated 302-record sample. We now confirm the conclusion
# holds on the paper's file exactly as published, with all 1025 rows retained and leakage
# blocked by grouping instead of by deduplication.

# %%

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

gkf = grouped_folds(GROUPS_FULL, n_splits=5, seed=SEED)
Xf, yf = df_full[FEATURES], df_full["target"].values

group_rows = []
for name, mdl in [("Paper stacking ensemble", build_paper_stack()),
                  ("RM-Stack (proposed)",     build_rm_stack())]:
    r = cross_validate(mdl, Xf, yf, cv=gkf, scoring=["accuracy", "roc_auc", "f1"], n_jobs=-1)
    group_rows.append({"Model": name,
                       "Accuracy": r["test_accuracy"].mean(),
                       "AUC": r["test_roc_auc"].mean(),
                       "F1": r["test_f1"].mean()})
group_df = pd.DataFrame(group_rows).set_index("Model")
group_df.to_csv(RES / "table16_grouped_protocol.csv")
group_df.round(4)

# %% [markdown]
# **Finding 8.1.** The ranking is unchanged under the secondary protocol, so our conclusion
# does not depend on the decision to deduplicate.

# %% [markdown]
# ## 9. Final comparative analysis
#
# The table below is the central result of this study. It places four numbers side by side:
# what the paper claimed, what we reproduced under its protocol, what that same pipeline is
# actually worth once leakage is removed, and what our proposed pipeline achieves.

# %%
final = pd.DataFrame([
    {"Study": "Bhagat et al. (2024), as published",
     "Protocol": "single random 80/20 split, duplicates on both sides",
     "Accuracy": 0.9853, "AUC": 0.9880,
     "Valid estimate of generalisation?": "No - 97.6% test/train record overlap"},
    {"Study": "Part 1 reproduction (this work)",
     "Protocol": "same protocol as the paper",
     "Accuracy": float(pd.read_csv(RES / "table04_reproduced_metrics.csv")
                       .set_index("Model").loc["Stacking (proposed)", "Accuracy"]),
     "AUC": float(pd.read_csv(RES / "table04_reproduced_metrics.csv")
                  .set_index("Model").loc["Stacking (proposed)", "AUC"]),
     "Valid estimate of generalisation?": "No - reproduces the same flaw"},
    {"Study": "Paper's ensemble, leakage-free evaluation",
     "Protocol": "deduplicated, repeated stratified 5-fold CV",
     "Accuracy": cv_results["Paper stacking ensemble"]["test_accuracy"].mean(),
     "AUC": cv_results["Paper stacking ensemble"]["test_roc_auc"].mean(),
     "Valid estimate of generalisation?": "Yes"},
    {"Study": "RM-Stack (proposed, this work)",
     "Protocol": "deduplicated, repeated stratified 5-fold CV",
     "Accuracy": cv_results["RM-Stack (proposed)"]["test_accuracy"].mean(),
     "AUC": cv_results["RM-Stack (proposed)"]["test_roc_auc"].mean(),
     "Valid estimate of generalisation?": "Yes"},
])
final.to_csv(RES / "table17_final_comparison.csv", index=False)
pd.set_option("display.max_colwidth", 60)
final.round(4)

# %%
# Figure 12 - the headline figure of this study.
fig, ax = plt.subplots(figsize=(10.5, 5.4))
labels = ["Paper\n(as published)", "Part 1\nreproduction",
          "Paper's ensemble\n(leakage-free)", "RM-Stack\n(proposed)"]
vals   = final["Accuracy"].values
cols   = ["#c0392b", "#e67e22", "#7f8c8d", "#27ae60"]
bars = ax.bar(labels, vals, color=cols, width=.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + .012, f"{v:.4f}", ha="center", fontweight="bold")
ax.axhspan(0, 0.90, color="grey", alpha=.06)
ax.set_ylim(0.6, 1.06); ax.set_ylabel("accuracy")
ax.set_title("The 98.5% headline is leakage, not performance\n"
             "(bars 1–2 measure memorisation of duplicated rows; bars 3–4 measure "
             "generalisation to unseen clinical records)")
ax.annotate("", xy=(1, 0.985), xytext=(2, 0.830),
            arrowprops=dict(arrowstyle="<->", color="crimson", lw=2))
ax.text(1.5, 0.895, f"{(vals[1]-vals[2])*100:.1f} pt\ninflation", ha="center",
        color="crimson", fontweight="bold")
plt.tight_layout(); plt.savefig(FIGS / "fig12_final_comparison.png", dpi=200)
plt.show()

# %% [markdown]
# ## 10. Discussion
#
# **Strengths.**
# * The evaluation is leakage-free under two independent protocols (deduplication and
#   record-grouped CV), repeated 25 times, and supported by paired significance tests rather
#   than a single number.
# * Every preprocessing step is fitted inside the cross-validation loop, so no test statistic
#   ever reaches a training fold.
# * The encoding and composition studies provide exploratory evidence for the selected design,
#   although they do not isolate the causal effect of either choice.
# * Probability outputs are assessed with a reliability diagram and Brier score, and the
#   operating-point trade-off is made explicit.
#
# **Limitations.**
# * After deduplication only 302 distinct records remain, all from the Cleveland cohort. Confidence
#   intervals are correspondingly wide, and the improvement in accuracy, while consistent
#   across folds, is modest in absolute terms.
# * Representation and ensemble composition were selected using the full 302-record development
#   dataset rather than repeated inside every outer training fold. The resulting performance
#   estimate may therefore contain design-selection optimism.
# * RM-Stack's members and hyperparameters were deliberately selected, whereas the reconstructed
#   baseline uses library defaults where the paper reports no settings. Part of the mean
#   difference may reflect tuning or model selection, so it cannot be attributed solely to
#   representation matching or ensemble composition.
# * The dataset carries no external validation cohort; the Hungarian, Swiss and Long Beach
#   databases named in the paper's Section 4 are not actually present in this file, so a
#   cross-site generalisation study is not possible with these data.
# * `ca = 4` and `thal = 0` are retained as legitimate levels, since their original meaning
#   (missing values in the UCI source) cannot be recovered from this redistribution.
# * The F2 operating point is chosen on out-of-fold predictions from the same 302 records; a
#   prospective deployment would need to re-select it on a separate calibration cohort.
#
# **Practical implication.** The most important finding of this study is negative and it is
# about method rather than about hearts: a widely-redistributed benchmark file contained 70%
# duplicated rows, and a peer-reviewed paper built a state-of-the-art claim on top of them
# without checking. Any clinical ML result on a redistributed benchmark should be accompanied
# by a record-level duplication audit — a check that costs one line of code and, in this case,
# accounts for more than twenty accuracy points.

# %% [markdown]
# ## 11. Part 2 summary
#
# 1. Seven concrete limitations of the original paper were identified from the Part 1 evidence.
# 2. **RM-Stack** was designed to address them: leakage-aware data handling, representation
#    matched to each learner's inductive bias, a compact diverse ensemble chosen by a
#    documented composition study, out-of-fold probability meta-features, calibration
#    assessment, and an explicit clinical operating point.
# 3. Under identical leakage-free folds, RM-Stack achieved higher mean accuracy and ROC-AUC
#    than the reproduced paper ensemble. None of the five metric differences was significant
#    under the dependence-aware corrected test.
# 4. The comparison estimates the difference between the complete pipelines. Representation
#    and composition were selected using the full development dataset, and RM-Stack received
#    more tuning than the default-based reproduced baseline. The results therefore do not
#    establish that any individual design choice caused the observed mean gain.
# 5. Threshold selection reduces missed disease cases substantially — a clinically meaningful
#    gain invisible to the accuracy-only evaluation used in the paper.

# %% [markdown]
# ## Generative AI Acknowledgement
#
# I used generative AI tools to help plan the experiments, draft and comment portions of this
# code, assist with debugging. I ran the code myself, checked the outputs against the saved result
# files. I take full responsibility for the accuracy and final content of this work.
