"""Figure 13 - RM-Stack architecture, for Section 8.3 of the report.

Stage labels live in a reserved left gutter so that no connector ever crosses text.
Colours match the figures produced by the two notebooks.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figures"; FIGS.mkdir(exist_ok=True)

C_DATA, C_LINEAR, C_TREE = "#d95f02", "#2980b9", "#27ae60"
C_META, C_OUT, C_EDGE, C_MUTED = "#8e44ad", "#16a085", "#2c3e50", "#7f8c8d"

fig, ax = plt.subplots(figsize=(13.2, 9.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x, y, w, h, text, fc, fs=9.0, bold=False, alpha=0.16):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55,rounding_size=1.3",
                                linewidth=1.6, edgecolor=fc, facecolor=fc, alpha=alpha,
                                zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=C_EDGE, zorder=3,
            linespacing=1.5)


def arrow(x1, y1, x2, y2, color=C_EDGE, lw=1.8):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                                 linewidth=lw, color=color, zorder=4))


def stage(y, label, color):
    """Rotated stage label in the reserved left gutter, plus a faint rule."""
    ax.text(3.4, y, label, rotation=90, ha="center", va="center", fontsize=9.4,
            fontweight="bold", color=color)
    ax.plot([7.2, 7.2], [y - 7.5, y + 7.5], color=color, lw=2.4, alpha=0.55,
            solid_capstyle="round")


ax.text(53, 97.5, "RM-Stack: representation-matched, leakage-aware stacking",
        ha="center", fontsize=15, fontweight="bold", color=C_EDGE)

# ------------------------------------------------------- STAGE 1: data handling
stage(85.5, "DATA", C_DATA)
box(13, 81, 23, 8.6, "heart.csv\n1025 rows\n(as published)", C_DATA, bold=True)
box(41, 81, 25, 8.6, "record-level audit\n302 distinct records\n723 duplicates (70.5%)", C_DATA)
box(71, 81, 26, 8.6, "PRIMARY: deduplicate\n→ 302 records\n"
                     "SECONDARY: group all copies\ninto the same fold", C_DATA, fs=8.3)
arrow(36.2, 85.3, 40.6, 85.3, C_DATA)
arrow(66.2, 85.3, 70.6, 85.3, C_DATA)

# ------------------------------------------------------- evaluation protocol bar
box(25, 71.5, 60, 5.8, "repeated stratified 5-fold cross-validation   (5 repeats = 25 folds)\n"
                       "every transform below is fitted on training folds only",
    C_MUTED, fs=8.7, alpha=0.13)
arrow(84, 80.6, 84, 77.7, C_DATA)

# ------------------------------------ STAGE 2: representation matched to learner
stage(54, "REPRESENTATION", C_META)
box(13, 55, 36, 10.2,
    "one-hot representation\n"
    "continuous (age, trestbps, chol, thalach, oldpeak) → standardised\n"
    "nominal (cp, restecg, slope, thal, ca) → one-hot, drop first\n"
    "binary (sex, fbs, exang) → passthrough", C_LINEAR, fs=8.1)
box(59, 55, 36, 10.2,
    "native representation\n"
    "all 13 predictors kept as original integer codes\n"
    "no scaling, no one-hot expansion\n"
    "splits isolate individual categories directly", C_TREE, fs=8.1)
arrow(31, 71.1, 31, 65.6, C_LINEAR)
arrow(77, 71.1, 77, 65.6, C_TREE)

box(17, 43, 28, 8.2, "Logistic Regression (C = 1)\nsmooth · high bias · low variance",
    C_LINEAR, bold=True, fs=8.9)
box(63, 43, 28, 8.2, "Extra Trees (400, leaf = 3)\nnon-linear · low bias · high variance",
    C_TREE, bold=True, fs=8.9)
arrow(31, 54.6, 31, 51.6, C_LINEAR)
arrow(77, 54.6, 77, 51.6, C_TREE)

ax.text(54, 47.2, "complementary\nerrors", ha="center", va="center", fontsize=8.5,
        style="italic", color=C_MUTED, linespacing=1.5)

# ---------------------------------------------- STAGE 3: meta-level combination
stage(30.5, "META-LEVEL", C_META)
box(25, 32, 60, 5.6, "out-of-fold predicted PROBABILITIES from each base learner "
                     "(5-fold internal CV)", C_META, fs=8.7)
arrow(31, 42.6, 34, 37.9, C_LINEAR)
arrow(77, 42.6, 74, 37.9, C_TREE)

box(31, 24.2, 48, 5.6, "meta-learner:  Logistic Regression (C = 1)", C_META, bold=True,
    fs=9.2)
arrow(55, 31.6, 55, 30.1, C_META)

# ------------------------------- STAGE 4: calibrated output and operating point
stage(14.5, "OUTPUT", C_OUT)
box(13, 10.5, 24, 7.6, "calibrated probability\nof heart disease", C_OUT, fs=8.9)
box(42, 10.5, 24, 7.6, "threshold selection\nmaximise F2\n(recall weighted 4×)", C_OUT, fs=8.6)
box(71, 10.5, 26, 7.6, "referral decision\nrecall 0.88 → 0.98", C_OUT, bold=True, fs=8.9)
arrow(46, 23.8, 27, 18.5, C_OUT)
arrow(37.2, 14.3, 41.6, 14.3, C_OUT)
arrow(66.2, 14.3, 70.6, 14.3, C_OUT)

# ------------------------------------------------------------------- footnote
ax.plot([9, 97], [5.6, 5.6], color=C_MUTED, lw=0.9, alpha=0.5)
ax.text(53, 3.9,
        "Differences from Bhagat et al. (2024):   duplicates removed rather than split "
        "across train and test  ·  encoding matched per learner rather than uniform  ·  "
        "2 complementary members rather than 6 correlated ones",
        ha="center", va="center", fontsize=7.9, style="italic", color=C_MUTED)
ax.text(53, 1.9,
        "probability meta-features rather than hard labels  ·  25 folds rather than a "
        "single split  ·  decision threshold chosen rather than assumed",
        ha="center", va="center", fontsize=7.9, style="italic", color=C_MUTED)

plt.savefig(FIGS / "fig13_rmstack_architecture.png", dpi=210, bbox_inches="tight",
            facecolor="white")
print("written")
