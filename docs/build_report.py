"""Build docs/CardioLens3D_Documentation.pdf (<= 6 pages) from the exported results."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
IMG = DOCS / "img"
IMG.mkdir(parents=True, exist_ok=True)
M = json.loads((ROOT / "web" / "model" / "model.json").read_text())
VAL = json.loads((ROOT / "ml" / "results" / "shap_validation.json").read_text())
T = ["CAD", "LAD", "LCX", "RCA"]
NAMES = {"LogReg-L2": "Logistic reg. (L2)", "LogReg-L1": "Logistic reg. (L1)", "RandomForest": "Random forest",
         "HistGB": "Hist. grad. boosting", "XGBoost": "XGBoost (depth 2)", "Ensemble(LR+XGB)": "Ensemble L1-LR + XGB (deployed)"}

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
BLUE = "#2a78d6"

fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
for ax, t in zip(axes, T):
    fpr, tpr = M["targets"][t]["metrics"]["roc"]
    auc = M["targets"][t]["metrics"]["cv"]["roc_auc"]
    ax.plot([0, 1], [0, 1], ls="--", c="#bbb", lw=0.8)
    ax.plot(fpr, tpr, c=BLUE, lw=1.8)
    ax.set_title(f"{t}  (CV AUC {auc[0]:.3f} ± {auc[1]:.3f})", fontsize=8)
    ax.set_xlabel("False positive rate")
    ax.set_aspect("equal")
axes[0].set_ylabel("True positive rate")
fig.tight_layout()
fig.savefig(IMG / "roc.png", dpi=220)
plt.close(fig)

fig, axes = plt.subplots(1, 4, figsize=(9, 2.4))
for ax, t in zip(axes, T):
    cal = M["targets"][t]["metrics"]["calibration"]
    ax.plot([0, 1], [0, 1], ls="--", c="#bbb", lw=0.8)
    ax.plot([c[0] for c in cal], [c[1] for c in cal], "o-", c=BLUE, lw=1.6, ms=4)
    ax.set_title(t, fontsize=8)
    ax.set_xlabel("Mean predicted")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
axes[0].set_ylabel("Observed rate")
fig.tight_layout()
fig.savefig(IMG / "calibration.png", dpi=220)
plt.close(fig)

feats = [f["label"] for f in M["features"]]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.1))
for ax, t in zip(axes, ["CAD", "LAD"]):
    imp = M["targets"][t]["global_importance"]
    order = sorted(range(len(imp)), key=lambda j: -imp[j])[:10][::-1]
    ax.barh([feats[j] for j in order], [imp[j] for j in order], color=BLUE, height=0.6)
    ax.set_title(f"{t}: mean |SHAP| (log-odds)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
fig.tight_layout()
fig.savefig(IMG / "importance.png", dpi=220)
plt.close(fig)

# ------------------------------------------------------------------ document
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=16, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11.5, spaceBefore=8, spaceAfter=3, textColor=colors.HexColor("#8a1f1f"))
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9, leading=12.2, spaceAfter=4)
S = ParagraphStyle("S", parent=B, fontSize=7.8, leading=10, textColor=colors.HexColor("#555555"))
BUL = ParagraphStyle("BUL", parent=B, leftIndent=10, bulletIndent=0)
W = A4[0] - 30 * mm


def p(t, st=B):
    return Paragraph(t, st)


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


def table(rows, widths, head=True, zebra=True, bold_rows=()):
    t = Table(rows, colWidths=widths, repeatRows=1 if head else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), 7.6), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#999999"))]
    if head:
        st.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f4f2")))
    for r in bold_rows:
        st.append(("FONTNAME", (0, r), (-1, r), "Helvetica-Bold"))
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fbe9e9")))
    t.setStyle(TableStyle(st))
    return t


story = []
story += [p("CardioLens 3D: Coronary Risk, Mapped onto the Heart", H1),
          p("Multimodal AI Hackathon 2026 · Track A: Cardiovascular Risk Visualization &amp; Prediction · "
            "Live app: <font color='#2a78d6'>https://azlaanaliladha-ace.github.io/cardiolens-3d/</font> · "
            "Code: <font color='#2a78d6'>https://github.com/azlaanaliladha-ACE/cardiolens-3d</font>", S),
          Spacer(1, 4),
          p("<b>Clinical safety notice.</b> CardioLens 3D is an educational decision-support prototype. Its predictions "
            "are not a substitute for coronary angiography, CT angiography or any formal diagnostic imaging, and must not "
            "be used to diagnose, treat or rule out disease. The notice is shown permanently in the UI and acknowledged on first visit.", S)]

story += [p("1. Overview", H2),
          p("A risk percentage alone does not tell a clinician or a patient <i>where</i> disease is likely. CardioLens 3D "
            "predicts overall coronary artery disease (CAD) and ≥50% stenosis of each of the three major coronary arteries "
            "(LAD, LCX, RCA) from 55 routine demographic, examination, ECG, laboratory and echocardiographic features. It then "
            "paints each vessel-level probability onto the matching artery, and its perfusion territory, of an interactive 3D heart "
            "inside a translucent torso. Every prediction is explained with <b>exact</b> SHAP values computed live in the browser. "
            "Any input can be edited and all four models, their explanations and the 3D scene update in about 1 ms."),
          ]
story += bullets([
    "<b>Predictive pipeline:</b> per-target ensemble (L1 logistic regression + depth-2 XGBoost averaged in log-odds), "
    "benchmarked against five model families with 5×5 repeated stratified cross-validation; no target leakage.",
    "<b>3D mapping:</b> procedural anatomical heart with an 11-segment coronary tree (LM, LAD, D1, D2, LCX, OM1, OM2, RCA, AM, PDA, PL) "
    "routed along the real grooves; vessels and myocardial territories coloured by risk; rotate, zoom, six clinical views, click-to-inspect.",
    "<b>Clinical dashboard:</b> CAD and vessel probabilities, SHAP waterfall per target, physiological breakdown with reference ranges, "
    "cohort percentiles and signed contributions; ground-truth comparison and out-of-fold probabilities for all 303 cohort patients.",
    "<b>Zero-install deployment:</b> a static site (three.js + vanilla JS); the model ships as a 460 KB JSON file; runs on integrated graphics."])

story += [p("2. Data and preprocessing", H2),
          p("<b>Dataset.</b> Extension of the Z-Alizadeh Sani dataset (UCI #411; 303 patients; CC BY 4.0). Targets: "
            "<i>Cath</i> (CAD vs normal, prevalence 71.3%), and Stenotic vs Normal for <i>LAD</i> (58.4%), <i>LCX</i> (39.3%) "
            "and <i>RCA</i> (37.6%). <b>LAD, LCX, RCA and Cath are excluded from every model's input</b> (asserted in code)."),
          p("<b>Encoding.</b> Y/N flags → 0/1; sex → male indicator; bundle-branch block → LBBB and RBBB indicators; valvular heart disease → "
            "ordinal 0–3 (none/mild/moderate/severe). The constant column <i>Exertional CP</i> (all “N”) is dropped. Result: 55 numeric "
            "features grouped as Demographics (6), Risk factors &amp; history (11), Examination (7), Symptoms (6), ECG (8), Laboratory (14), "
            "Echocardiography (3). There are no missing values. Standardisation is fit inside each training fold through scikit-learn Pipelines, so no "
            "statistics leak from validation folds.")]

story += [p("3. Model architecture and selection", H2),
          p("For each target we compared five families under identical 5-fold stratified CV repeated 5 times (25 folds, seed 42): "
            "L2 and L1 logistic regression (standardised), random forest (500 trees), histogram gradient boosting and XGBoost (300 trees, "
            "depth 2, learning rate 0.05, row/column subsampling). Hyper-parameters were fixed to conservative values rather than searched, "
            "because a hyper-parameter search on 303 samples without nested CV would give optimistic estimates."),
          p("Linear models were best for CAD and RCA, while trees were better for LAD and LCX. We therefore deploy a per-target "
            "<b>log-odds ensemble</b>: logit(x) = ½·LR<sub>L1</sub>(x) + ½·XGB(x). It is best or near-best on every target, has the best Brier "
            "score on every target, and, because both members are explained in the same log-odds space, its SHAP values are exactly "
            "φ = ½·φ<sub>LR</sub> + ½·φ<sub>XGB</sub>.")]

rows = [["Target", "Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1", "Brier ↓"]]
bold = []
for t in T:
    for k, v in M["benchmark"][t].items():
        rows.append([t, NAMES[k]] + [f"{v[m][0]:.3f}" for m in ["roc_auc", "accuracy", "precision", "recall", "f1", "brier"]])
        if k.startswith("Ensemble"):
            bold.append(len(rows) - 1)
story += [KeepTogether([p("<b>Table 1.</b> Cross-validated performance (mean over 25 held-out folds, threshold 0.5). Deployed model highlighted.", S),
                        table(rows, [12 * mm, 48 * mm] + [19 * mm] * 6, bold_rows=bold)])]

story += [p("4. Evaluation results", H2),
          Image(str(IMG / "roc.png"), width=W, height=W * 2.4 / 9),
          p("<b>Figure 1.</b> Out-of-fold ROC curves of the deployed ensemble (single 5-fold split; titles give the 25-fold mean ± SD).", S),
          Image(str(IMG / "calibration.png"), width=W, height=W * 2.4 / 9),
          p("<b>Figure 2.</b> Reliability diagrams (5 probability bins, out-of-fold). Overall CAD and LAD are well calibrated; LCX and RCA probabilities are compressed toward the prevalence.", S),
          p("<b>Interpretation.</b> Overall CAD is predicted with high discrimination (AUC 0.929, F1 0.915). LAD is moderately well predicted "
            "(AUC 0.837). LCX and RCA are harder (AUC ≈ 0.73, recall ≈ 0.5 at the 0.5 threshold). This matches the clinical expectation that "
            "routine non-invasive data says more about <i>whether</i> CAD is present than about <i>which</i> vessel is affected. The UI shows these "
            "limits in the Model card rather than hiding them.")]

story += [p("5. Clinical interpretability", H2),
          p("<b>Exact SHAP in the browser.</b> For the standardised logistic member, φ<sub>j</sub> = w<sub>j</sub>(x<sub>j</sub> − μ<sub>j</sub>)/σ<sub>j</sub> "
            "(exact because the training mean of each standardised input is zero). For XGBoost we implemented path-dependent TreeSHAP "
            "<i>exactly</i> by enumerating all coalitions of the ≤3 features each depth-2 tree splits on. Conditional expectations use the "
            "training cover stored in each node, and splits are compared in float32 as XGBoost does. Validation results:"),
          ]
vr = [["Target", "max |φ − XGBoost pred_contribs|", "max additivity error", "max |p_browser − p_python| (303 pts)"]]
for t in T:
    vr.append([t, f"{VAL[t]['max_abs_diff_vs_xgboost_pred_contribs']:.1e}", f"{VAL[t]['max_additivity_error']:.1e}", "< 1e-6 (live check in app)"])
story += [table(vr, [16 * mm, 52 * mm, 40 * mm, 60 * mm]), Spacer(1, 4),
          Image(str(IMG / "importance.png"), width=W, height=W * 3.1 / 9),
          p("<b>Figure 3.</b> Global importance (mean |SHAP| over the cohort). Typical chest pain, age, regional wall-motion abnormality, "
            "diabetes, hypertension and ECG T-wave inversion dominate, in line with established CAD risk factors.", S),
          p("<b>Dashboard views.</b> (i) A <b>SHAP waterfall</b> for the selected target (CAD, LAD, LCX or RCA), from the cohort baseline to "
            "this patient's log-odds, with probabilities at both ends and a live additivity check. (ii) A <b>physiological breakdown</b> table "
            "listing every measurement with its value, unit, adult reference range, H/L flag, cohort percentile and signed contribution "
            "bar, sortable to “out of range only”. (iii) A <b>vessel card</b> opened by clicking an artery or heart wall, showing the "
            "vessel's probability, territory and top four drivers.")]

story += [p("6. 3D pipeline", H2),
          p("<b>Anatomy.</b> The ventricular mass is an analytic surface S(θ, φ). θ is the polar angle from base to apex and φ is the azimuth "
            "from anterior towards the patient's left. A unit sphere is tapered toward a pointed apex, widened on the left-ventricular side, "
            "and carved with Gaussian grooves for the atrioventricular sulcus and the anterior and posterior interventricular sulci. Atrial "
            "appendages, the ascending aorta and arch branches, the pulmonary trunk and arteries, the SVC and the IVC are tube meshes. The heart "
            "sits inside a Fresnel-shaded translucent torso with ribs, sternum and spine, and is oriented with the apex anterior, inferior and leftward."),
          p("<b>Coronary tree.</b> Each segment is a Catmull-Rom spline defined in (θ, φ) space along its anatomical course and lifted "
            "onto the surface, so the vessels stay attached to the myocardium. Segments: LM → LAD with D1 and D2 in the anterior "
            "interventricular groove; LCX with OM1 and OM2 in the left AV groove; RCA in the right AV groove with acute marginal, PDA "
            "(posterior interventricular groove) and posterolateral branches (right-dominant). Each mesh carries a vessel ID, which gives a "
            "fixed, explicit mapping between model outputs and the anatomy."),
          p("<b>Risk mapping.</b> Vessels use a diverging blue → grey → red scale centred on the 0.5 decision threshold; the left main takes "
            "the worse of LAD and LCX. Every ventricular vertex has soft perfusion weights w<sub>v</sub> ∝ exp(−(d<sub>v</sub>² − d<sub>min</sub>²)/σ²) "
            "from its distance to each vessel tree, and its colour is the weighted blend of the three vessels' risk colours. The wall therefore "
            "shows the typical LAD (anterior, septum, apex), LCX (lateral) and RCA (inferior, right ventricle) territories. Selecting a vessel "
            "dims the other territories. Raycasting on the wall resolves the dominant supplying artery, so clicking the myocardium explains that vessel."),
          p("<b>Real-time behaviour.</b> Input edits trigger one requestAnimationFrame-batched update: 4 models + exact SHAP (≈1 ms), "
            "a vertex-colour repaint (≈16k vertices) and label refresh. Flow particles move more slowly in vessels with higher stenosis "
            "probability, and the heart beats at the patient's entered pulse rate. Rendering uses about 60k triangles and a capped pixel "
            "ratio, which runs smoothly on integrated GPUs. To add a structure, add an entry to the <i>TREE</i> table. To add a target or "
            "feature, re-run <i>train.py</i>; the UI is generated from the exported metadata.")]

story += [p("7. System architecture and usage", H2),
          table([["Component", "File(s)", "Role"],
                 ["Preprocessing", "ml/preprocess.py", "Excel → 55 numeric features + 4 targets; feature metadata"],
                 ["Benchmark", "ml/benchmark.py", "5 families × 4 targets, 5×5 repeated stratified CV"],
                 ["Training / export", "ml/train.py", "Final ensembles, SHAP validation, ROC/calibration, model.json"],
                 ["Model weights", "ml/models/*.joblib, web/model/model.json", "Fitted sklearn/XGBoost objects; portable JSON"],
                 ["Inference + SHAP", "web/js/model.js", "Browser inference, exact SHAP, cohort percentiles"],
                 ["3D scene", "web/js/heart3d.js", "Procedural heart, coronary tree, territories, interaction"],
                 ["Dashboard", "web/js/main.js, charts.js", "Inputs, presets, waterfall, physiology table, model card"]],
                [32 * mm, 62 * mm, 86 * mm]),
          Spacer(1, 4),
          p("<b>Run.</b> <font face='Courier'>cd web &amp;&amp; python -m http.server 8000</font> (the trained model is committed). "
            "<b>Reproduce.</b> <font face='Courier'>pip install -r requirements.txt; cd ml; python benchmark.py; python train.py</font>. "
            "All randomness is seeded. <b>Workflow:</b> choose a cohort example, a template or any of the 303 patients → inspect the "
            "3D map and vessel bars → edit any input (changed fields are highlighted, with deltas against the loaded patient) → read "
            "the waterfall and physiology table → check the Model card.")]

story += [p("8. Limitations", H2)]
story += bullets([
    "Single-centre cohort of 303 patients with no external validation; calibration and accuracy may not transfer to other populations.",
    "Vessel-level labels only. The lesion position along a vessel is unknown, so whole vessels are coloured. Territories assume right-dominant anatomy.",
    "The heart is schematic, not derived from patient imaging. “Region RWMA” is a count of regions, not a localised finding.",
    "LCX/RCA predictions are only moderately informative (AUC ≈ 0.73). The 0.5 threshold favours specificity for these vessels.",
    "SHAP describes the model's behaviour, not causal physiology."])
story += [p("<b>References.</b> Alizadehsani R. et al., Extension of Z-Alizadeh Sani dataset, UCI ML Repository (2017), doi:10.24432/C5Q31T. "
            "Lundberg &amp; Lee, A Unified Approach to Interpreting Model Predictions, NeurIPS 2017. Lundberg et al., From local explanations "
            "to global understanding with explainable AI for trees, Nature Machine Intelligence 2020. Chen &amp; Guestrin, XGBoost, KDD 2016. three.js (MIT).", S)]

shot = IMG / "app.png"
if shot.exists():
    story.insert(5, Image(str(shot), width=W, height=W * 0.51))
    story.insert(6, p("<b>Figure 0.</b> The CardioLens 3D explorer: patient inputs (left), the 3D risk map (centre) and the explanation dashboard (right).", S))


def footer(c, d):
    c.saveState()
    c.setFont("Helvetica", 7)
    c.setFillColor(colors.HexColor("#777777"))
    c.drawString(15 * mm, 9 * mm, "CardioLens 3D · educational decision-support prototype, not a diagnostic device")
    c.drawRightString(A4[0] - 15 * mm, 9 * mm, f"{d.page}")
    c.restoreState()


out = DOCS / "CardioLens3D_Documentation.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=13 * mm,
                  bottomMargin=15 * mm, title="CardioLens 3D Documentation").build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", out)
