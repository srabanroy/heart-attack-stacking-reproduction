# Reproducing and Improving a Stacking-Based Heart Attack Prediction Model

Code package for the HD task **"Machine Learning Mini Research"**.

**Paper reproduced (Part 1)**
M. Bhagat, A. Sharma and P. Agarwal, "An efficient stacking-based ensemble technique for
early heart attack prediction," *Multimedia Tools and Applications*, vol. 84,
pp. 36351–36375, 2025. DOI: [10.1007/s11042-024-19293-7](https://doi.org/10.1007/s11042-024-19293-7)

---

## 1. What this package contains

```
.
├── data/
│   └── heart.csv                         # the dataset used by the paper (1025 x 14)
├── notebooks/
│   ├── 01_part1_reproduction.ipynb       # Part 1 - faithful reproduction + critical analysis
│   ├── 01_part1_reproduction.py          # same notebook in jupytext 'percent' script form
│   ├── 02_part2_proposed_solution.ipynb  # Part 2 - proposed RM-Stack solution
│   └── 02_part2_proposed_solution.py     # same notebook in script form
├── src/
│   └── make_architecture_figure.py       # draws fig13, the RM-Stack architecture diagram
├── results/                              # 17 result tables, written as CSV by the notebooks
├── figures/                              # 13 figures, written as PNG by the notebooks
├── requirements.txt
└── README.md
```

Both notebooks are committed **with their outputs already executed**, so the results can be
read without running anything. They are also fully re-runnable from a clean environment
(Section 3). Use the reproduction requirements file when exact agreement with the submitted
tables is required.

## 2. Dataset

`data/heart.csv` is a public mirror of the Kaggle *Heart Disease Dataset*
(`johnsmith88/heart-disease-dataset`) named in Section 4 of the paper: 1025 rows, 13
predictors plus a binary `target`. The submitted file was obtained from the Hugging Face
dataset mirror `vishal323/heart`, because direct Kaggle access was unavailable. Its schema,
row count and initial records match the widely redistributed Kaggle file; the checksum below
identifies the exact file used for every reported result.

| property | value |
|---|---|
| rows as published | 1025 |
| **distinct clinical records** | **302** |
| duplicate rows | 723 (70.5% of the file) |
| copies per distinct record | 3, 4 or 8 |
| class balance (all rows) | 526 positive / 499 negative |
| missing values | 0 |
| SHA-256 | `ea77cdebac756ab984173c29107ccf4848577fffb77b5848a65e0426d473589b` |

Verify the file is unmodified:

```bash
shasum -a 256 data/heart.csv
```

The duplication figure above is the central finding of Part 1 and the starting point of
Part 2.

## 3. How to install, run and reproduce

### Install

Requires **Python 3.9 or newer**.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For a convenient installation using compatible minimum versions:

```bash
pip install -r requirements.txt
```

For numerical reproduction of the submitted tables, use the tested core environment:

```bash
pip install -r requirements-reproduction.txt
```

The first cell of each notebook prints the versions actually in use. Later library versions
can change fitted probabilities and tree importance values slightly even when the seed and
fold assignments are unchanged; model rankings and the main conclusions should remain stable.

### Run

```bash
jupyter lab                        # then run notebooks/01_... followed by notebooks/02_...
```

Run them in order: notebook 02 reads one table that notebook 01 writes
(`results/table04_reproduced_metrics.csv`).

### Reproduce everything non-interactively

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_part1_reproduction.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_part2_proposed_solution.ipynb
```

Measured runtime: **~2 min** for notebook 01 and **~12 min** for notebook 02 (the
composition study and the nested cross-validation dominate). Notebook 01 also completed in
33 s on the Python 3.10 machine described below, so these are indicative rather than fixed.

### Determinism

Every stochastic component is seeded with `SEED = 42`, and all cross-validation splitters
take an explicit `random_state`. Exact numerical agreement is expected under the tested core
versions in `requirements-reproduction.txt`. Compatible newer versions may produce small
floating-point or estimator-level differences.

The notebooks locate the project root by searching upward for `data/heart.csv`, so they run
correctly whether they are launched from the project root, from `notebooks/`, or from an
editor whose working directory is set elsewhere. To override, set the environment variable
`HD_PROJECT_ROOT` to the project folder.

### Environment used for the reported results

| Component | Version |
|---|---|
| Python | 3.12.13 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| scipy | 1.17.1 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |
| joblib | 1.6.0 |

The code also runs on newer compatible libraries, but fitted probabilities and tree
importance values can differ slightly. Use the environment above for the submitted tables.

## 4. Headline results

### Part 1 — reproduction

| Model | Paper accuracy | Reproduced accuracy |
|---|---|---|
| Logistic Regression | 0.8439 | 0.7951 |
| Naive Bayes | 0.8439 | 0.8000 |
| KNN | 0.8585 | 0.8341 |
| Decision Tree | 0.9268 | 0.9854 |
| Random Forest | 0.9268 | 0.9854 |
| XGBoost | 0.9073 | 0.9854 |
| **Stacking ensemble** | **0.9853** | **0.9854** |

The paper's headline claim reproduces almost exactly. The diagnostic in notebook 01 then
shows why that number is not a measure of predictive performance: under the paper's random
row-level split, **97.6% of test rows have an identical clinical record in the training
set**. Without patient identifiers, the defensible conclusion is that the models are recalling
exact clinical profiles already present in training rather than generalising to unseen records.

### Part 2 — proposed solution (RM-Stack)

Both models evaluated under identical leakage-free repeated stratified 5-fold
cross-validation (25 folds) on the 302 distinct records:

| Metric | Paper's ensemble | RM-Stack (proposed) | Δ | corrected *t* p-value |
|---|---|---|---|---|
| Accuracy | 0.8298 | **0.8436** | +0.0138 | 0.423 |
| Precision | 0.8253 | **0.8465** | +0.0212 | 0.215 |
| Recall | 0.8769 | 0.8767 | −0.0002 | 0.994 |
| F1 | 0.8478 | **0.8583** | +0.0105 | 0.515 |
| ROC-AUC | 0.9038 | **0.9188** | +0.0150 | 0.167 |

The proposed model has higher mean accuracy, precision, F1 and ROC-AUC, but none of these
differences is statistically significant under the dependence-aware corrected resampled
t-test. The ordinary paired and Wilcoxon tests are retained in the results table as
descriptive sensitivity checks; they are anti-conservative when cross-validation training
sets overlap. Nested hyperparameter tuning gives accuracy 0.8543 and AUC 0.9219. Under the
record-grouped protocol on the unmodified 1025-row file, accuracy changes from 0.7720 to
0.8162 and AUC from 0.8768 to 0.9079.

## 5. What RM-Stack is

A representation-matched, leakage-aware stacking ensemble:

1. **Leakage-aware data handling** — deduplication to 302 distinct records (primary), with
   record-grouped cross-validation on the full file as a sensitivity check.
2. **Representation matching** — the five nominal attributes (`cp`, `restecg`, `slope`,
   `thal`, `ca`) are one-hot encoded for the selected linear member and left as native integer
   codes for the selected tree member. Notebook 02 shows that the effect of encoding depends
   on both learner family and hyperparameters, so preprocessing is selected per learner rather
   than imposed globally.
3. **Compact ensemble** — a regularised logistic regression plus extremely randomised trees,
   selected by a documented composition study in which every larger tested stack performed
   worse than the two-member stack at n = 302.
4. **Out-of-fold probability meta-features** with a regularised logistic-regression
   meta-learner.
5. **Calibration assessment and an explicit clinical operating point** chosen by maximising
   F2, which weights recall four times as heavily as precision. The notebook evaluates
   calibration with a reliability diagram and Brier score; it does not fit a separate
   probability-calibration model.

## 6. Results index

| File | Contents |
|---|---|
| `table01_dataset_vs_paper_table9.csv` | Attribute ranges: file vs the paper's Table 9 |
| `table02_dataset_profile.csv` | Duplication and class-balance profile |
| `table03_feature_importance.csv` | Reproduction of the paper's Fig. 2 |
| `table04_reproduced_metrics.csv` | Full reproduced metric set (Part 1) |
| `table05_reproduced_vs_paper.csv` | Reproduced vs the paper's Table 11 |
| `table06_split_sensitivity.csv` | Accuracy over 100 random splits |
| `table07_leakage_diagnostic.csv` | Train/test record overlap per split |
| `table08_leakage_impact.csv` | Accuracy with and without leakage |
| `table09_paper_inconsistencies.csv` | Eight internal contradictions in the paper |
| `table10_encoding_study.csv` | Encoding × learner study |
| `table11_composition_study.csv` | Ensemble composition study |
| `table12_cv_summary.csv` | Repeated-CV summary, both models |
| `table13_significance_tests.csv` | Paired t-test, Wilcoxon, Cohen's dz |
| `table14_nested_cv.csv` | Nested cross-validation estimate |
| `table15_operating_points.csv` | Default vs F2-optimal threshold |
| `table16_grouped_protocol.csv` | Record-grouped sensitivity check |
| `table17_final_comparison.csv` | Final four-way comparison |

Figures `fig01`–`fig07` belong to Part 1; `fig08`–`fig12` to Part 2; `fig13` is the
RM-Stack architecture diagram. Filenames state their contents.

## 7. Assumptions made where the paper is silent

The paper omits several implementation details. Each is flagged inline in notebook 01 and
justified there; in summary:

| Detail | Paper | Assumption made | Justification |
|---|---|---|---|
| Split ratio | not stated | 80/20 | The paper's accuracies are exact multiples of 1/205, so the test set held 205 = 20% of 1025 rows |
| Stratification | not stated | none | Their recalls are multiples of 1/110, implying 110 positive / 95 negative in test; a stratified split gives 105/100 |
| Random seed | not stated | 42 | Arbitrary but fixed; sensitivity to this choice is measured over 100 splits in notebook 01 §9 |
| Scaler | "normalization" | `StandardScaler` | Standard reading of the sentence; the scikit-learn default |
| Hyperparameters | none given | library defaults | The only reproducible reading of an unspecified configuration, and what the paper's own pseudocode implies |
| Meta-learner | "can be any model" | Logistic Regression | First option the authors list; standard in the stacking literature; the scikit-learn default |
| Feature importance model | not stated | Random Forest Gini | The only one of their six models producing the profile shown in their Fig. 2 |

## 8. Generative AI acknowledgement

Generative AI tools were used to help plan the experimental design, draft and comment parts
of the implementation. All code was run locally on my machine.

## 9. Attribution

The dataset is redistributed from the public Hugging Face mirror identified in Section 2 so
that the results can be reproduced without an external account. The paper identifies the
corresponding Kaggle dataset as its source. The dataset remains the property of its original
publishers.
