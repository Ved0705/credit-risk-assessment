"""Generate paper figures from the project's saved model outputs (backend/models/metadata.json)."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(sys.argv[1])            # project root
OUT = Path(sys.argv[2])             # figures dir
FAIR = json.load(open(sys.argv[3])) # fairness report json
meta = json.load(open(ROOT / "backend/models/metadata.json"))
labels = json.load(open(ROOT / "backend/models/feature_schema.json"))["labels"]
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10})
INK, BLUE, GREEN, AMBER, RED, GREY = "#1f2937", "#3d5af1", "#1f9d6b", "#d99a06", "#c8423b", "#6b7280"

def box(ax, x, y, w, h, text, fc="#eef2ff", ec=BLUE, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=INK, weight="bold" if bold else "normal")

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))

# ---- Fig 1: system architecture ----
fig, ax = plt.subplots(figsize=(7.2, 5.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
box(ax, 0.6, 6.3, 8.8, 0.85, "React + Vite web dashboard\nDashboard  |  Assessment  |  Explainability  |  Fairness  |  Performance", "#f3f4f6", GREY, 8.5)
arrow(ax, 5, 6.3, 5, 5.75); ax.text(5.12, 5.98, "REST / JSON", fontsize=8, color=GREY)
box(ax, 2.0, 4.9, 6.0, 0.8, "FastAPI backend\nPydantic validation  |  error handling  |  CORS", "#eef2ff", BLUE, 8.5, True)
xs = [0.1, 2.5, 4.9, 7.3]
names = ["Prediction service\n(Pipeline: encoding,\nscaling, Random\nForest)", "Explanation service\n(SHAP TreeExplainer,\nlocal + global)", "Fairness service\n(Fairlearn\nMetricFrame,\ntest set)", "Stability service\n(perturbation +\nre-prediction)"]
for x, n in zip(xs, names):
    arrow(ax, 5, 4.9, x + 1.1, 4.15); box(ax, x, 2.7, 2.2, 1.45, n, "#ecfdf5", GREEN, 7.8)
    arrow(ax, x + 1.1, 2.7, 4.5, 1.65)
box(ax, 0.1, 0.35, 2.2, 1.25, "Offline training\n(train.py,\nGridSearchCV)", "#fef2f2", RED, 7.8)
arrow(ax, 2.3, 0.97, 2.55, 0.97)
box(ax, 2.6, 0.35, 3.9, 1.25, "Model artefacts (joblib/JSON)\nmodel, metadata, feature schema,\nSHAP background, test set", "#fffbeb", AMBER, 7.8)
box(ax, 6.8, 0.35, 3.1, 1.25, "In-memory assessment\nstore (decision, top\nfactors, input hash)", "#fffbeb", AMBER, 7.8)
ax.annotate("", xy=(9.8, 1.6), xytext=(8.0, 4.9), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2, connectionstyle="angle,angleA=0,angleB=90"))
fig.tight_layout(); fig.savefig(OUT / "system_architecture.png", dpi=300); plt.close(fig)

# ---- Fig 2: methodology pipeline ----
steps = ["German Credit dataset\n(1,000 applicants, 20 attributes)", "Decoding + cleaning\n(UCI code book)", "Separate sensitive attributes\n(sex/marital status, foreign worker)",
         "Stratified 80/20 split\n(800 train / 200 test)", "Pipeline: one-hot encoding +\nstandard scaling + classifier", "5-fold CV grid search\n(LR, RF, XGBoost; ROC-AUC)",
         "Model selection\n(simplicity margin 0.01)", "Cost-based threshold\n(5:1 cost, out-of-fold)", "Held-out evaluation"]
post = ["SHAP explanations", "Fairness audit", "Stability analysis"]
fig, ax = plt.subplots(figsize=(7.2, 6.4)); ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
y = 10.1
for i, s in enumerate(steps):
    box(ax, 2.6, y, 4.8, 0.78, s, "#eef2ff", BLUE, 8)
    if i < len(steps) - 1: arrow(ax, 5, y, 5, y - 0.33)
    y -= 1.11
for i, s in enumerate(post):
    x = 0.4 + i * 3.2
    box(ax, x, 0.05, 2.9, 0.7, s, "#ecfdf5", GREEN, 8.5, True); arrow(ax, 5, y + 1.11, x + 1.45, 0.75)
fig.tight_layout(); fig.savefig(OUT / "methodology.png", dpi=300); plt.close(fig)

# ---- Fig 3: model comparison ----
mets = ["accuracy", "precision", "recall", "f1", "roc_auc"]; names = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
comp = meta["comparison"]; colors = [GREY, BLUE, AMBER]
fig, ax = plt.subplots(figsize=(7.2, 3.6)); w = 0.26
for j, c in enumerate(comp):
    vals = [c[m] for m in mets]
    bars = ax.bar([i + (j - 1) * w for i in range(5)], vals, w, label=c["model"], color=colors[j])
    for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(range(5)); ax.set_xticklabels(names); ax.set_ylim(0, 1); ax.set_ylabel("Score (test set, threshold 0.5)")
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=300); plt.close(fig)

# ---- Fig 4: global SHAP importance ----
gi = meta["global_importance"][::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
ax.barh([labels[g["feature"]] for g in gi], [g["mean_abs_shap"] for g in gi], color=BLUE)
ax.set_xlabel("Mean |SHAP value| (probability units, 200 test applicants)"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "shap_global_importance.png", dpi=300); plt.close(fig)

# ---- Fig 5: fairness ----
titles = {"sex": "Sex", "age_group": "Age group", "foreign_worker_group": "Foreign worker"}
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
for ax, a in zip(axes, FAIR["attributes"]):
    g = a["groups"]; x = range(len(g))
    ax.bar([i - 0.2 for i in x], [v["approval_rate"] for v in g], 0.4, color=BLUE, label="Approval rate")
    ax.bar([i + 0.2 for i in x], [v["false_positive_rate"] for v in g], 0.4, color=AMBER, label="Good payers rejected (FPR)")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{v['group']}\n(n={v['count']})" for v in g], fontsize=7)
    ax.set_title(titles[a["attribute"]], fontsize=9); ax.set_ylim(0, 1); ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Rate"); axes[0].legend(frameon=False, fontsize=6.5, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "fairness_groups.png", dpi=300); plt.close(fig)
print("figures written")
