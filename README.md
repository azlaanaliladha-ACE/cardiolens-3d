# CardioLens 3D: coronary risk, mapped onto the heart

**Multimodal AI Hackathon 2026 · Track A: Cardiovascular Risk Visualization & Prediction**

CardioLens 3D predicts **overall coronary artery disease (CAD)** and **≥50% stenosis of the LAD, LCX and RCA** from routine clinical, ECG, laboratory and echocardiography data. It then paints each probability onto the matching coronary artery, and its perfusion territory, of an interactive 3D heart inside a translucent torso. Every prediction comes with **exact SHAP explanations**, computed live in the browser.

> ⚠️ **Educational decision-support prototype, not a diagnostic device.** The model was trained on one public cohort of 303 patients. It is not a substitute for coronary angiography, CT angiography or any formal diagnostic imaging or clinical judgement.

**Live demo:** _see the deployment URL in the Devpost submission_ · **Stack:** Python (scikit-learn, XGBoost) → JSON → vanilla JS + three.js. No server and no GPU required.

---

## Features

| Requirement | What CardioLens does |
|---|---|
| **Predictive modelling** | One model per target (CAD, LAD, LCX, RCA). Each is a log-odds average of an L1 logistic regression and a depth-2 XGBoost, benchmarked against 5 model families with 5×5 repeated stratified CV. LAD, LCX, RCA and Cath are **never used as inputs**. |
| **3D visual mapping** | Procedural, anatomically oriented heart (ventricles, atria, auricles, aorta and arch branches, pulmonary trunk, SVC, IVC). The coronary tree is routed along the real grooves: **LM → LAD (+D1, D2), LCX (+OM1, OM2), RCA (+AM, PDA, PL)**. Vessels and their myocardial **perfusion territories** are coloured by predicted probability. Rotate, zoom, six clinical camera views (anterior, LAO, left lateral, posterior, inferior, body). Click a vessel *or the heart wall* to inspect the supplying artery. |
| **Clinical dashboard** | Overall CAD probability and vessel bars, all with status icons and labels. A SHAP waterfall per target, a physiological breakdown table (value, reference range, H/L flag, cohort percentile, signed contribution) and what-if editing of all 55 inputs, with deltas against the loaded patient. |
| **Real-time integration** | Any input change re-runs all four models and their exact SHAP values (~1 ms), then recolours the vessels and territories. Flow particles slow down in high-risk vessels, and the heart beats at the patient's own pulse rate. |
| **Ground truth & honesty** | Load any of the 303 cohort patients to compare against their angiography labels and **out-of-fold** probabilities from a model that never saw them. The model card shows ROC, calibration, a full benchmark table and live verification of SHAP additivity. |
| **Safety** | A persistent disclaimer banner, a first-visit acknowledgement dialog, and a limitations section. |

## Results (repeated stratified 5-fold CV, 5 repeats = 25 folds, threshold 0.5)

| Target | Prevalence | ROC-AUC | Accuracy | Precision | Recall | F1 | Brier ↓ |
|---|---|---|---|---|---|---|---|
| **CAD** | 71.3% | **0.929** | 0.877 | 0.904 | 0.928 | 0.915 | 0.091 |
| **LAD** | 58.4% | **0.837** | 0.769 | 0.794 | 0.822 | 0.805 | 0.164 |
| **LCX** | 39.3% | **0.733** | 0.683 | 0.617 | 0.500 | 0.548 | 0.206 |
| **RCA** | 37.6% | **0.727** | 0.678 | 0.598 | 0.462 | 0.515 | 0.206 |

Full comparison (L2-LR, L1-LR, Random Forest, HistGB, XGBoost and the ensemble) is in [`ml/results/metrics.json`](ml/results/metrics.json) and in the app's **Model card** tab. The ensemble has the best or near-best AUC and the best Brier score on every target. Single models win on individual vessels (RF on LAD, XGBoost on LCX, L1-LR on RCA), but none is consistent across all four.

**SHAP fidelity:** the JavaScript/Python TreeSHAP implementation matches XGBoost's native `pred_contribs` to < 3×10⁻⁷, and `baseline + ΣSHAP = model log-odds` to < 2×10⁻⁶ ([`ml/results/shap_validation.json`](ml/results/shap_validation.json)). The app re-checks additivity live for every prediction.

## How it works

```
data/raw/*.xlsx ──► ml/preprocess.py ──► ml/benchmark.py (5 families × 4 targets, 5×5 CV)
                                     └─► ml/train.py ──► ml/models/*.joblib     (fitted models)
                                                      ├► ml/results/*.json     (metrics, SHAP validation)
                                                      └► web/model/model.json  (weights + trees + cohort + metrics)
web/  (static)  index.html · js/model.js (inference + exact SHAP) · js/heart3d.js (three.js scene)
                js/charts.js (SVG charts) · js/main.js (dashboard) · css/style.css
```

* **Preprocessing.** Y/N → 0/1, sex → male indicator, BBB → LBBB/RBBB one-hot, VHD → ordinal 0–3. The constant column `Exertional CP` is dropped, leaving 55 inputs in 7 clinical groups. Scaling is fit inside each CV fold through sklearn `Pipeline`s.
* **Ensemble.** `logit = 0.5·LR_L1(x) + 0.5·XGB(x)`. Averaging in log-odds keeps SHAP exactly additive: `φ = 0.5·φ_LR + 0.5·φ_XGB`.
* **Exact SHAP in the browser.** For the standardised LR, `φ_j = w_j (x_j − μ_j)/σ_j`. For XGBoost, path-dependent TreeSHAP is computed **exactly** per tree by enumerating subsets of the ≤3 features a depth-2 tree uses, with conditional expectations from the training cover. Splits are compared in float32, as XGBoost does.
* **3D mapping.** The ventricular surface is an analytic function `S(θ, φ)` with grooves carved in. Coronary centre-lines are Catmull-Rom splines in `(θ, φ)` space lifted onto the surface, so they stay attached to the myocardium. Each surface vertex gets soft territory weights from its distance to each vessel tree; the wall colour blends the three vessels' risk colours by those weights. The left main takes the worse of LAD and LCX. To add a structure, add an entry to `TREE` in `heart3d.js`. To add a target, add it to `TARGETS` in `preprocess.py`.
* **Colour.** A diverging blue → grey → red scale centred on the 0.5 decision threshold is used identically in 3D, the bars and the legend. Status is always shown with an icon and a label, never colour alone.

## Run it

**Web app only.** The trained model is committed, so no Python is needed:

```bash
cd web
python -m http.server 8000      # or any static server
# open http://localhost:8000
```

**Reproduce training** (Python 3.10+):

```bash
pip install -r requirements.txt
cd ml
python preprocess.py   # sanity check: (303, 55) features
python benchmark.py    # model comparison  -> ml/results/benchmark.json
python train.py        # final models, SHAP validation, export -> web/model/model.json
```

All randomness is seeded (`SEED = 42`). The raw dataset is included under `data/raw/` and can also be downloaded from UCI (#411).

**Deploy:** the repository root has `vercel.json` with `outputDirectory: web`. Any static host works.

## Limitations

* Single-centre cohort of 303 patients with no external validation. Performance on other populations is unknown.
* LCX/RCA prediction is only moderately informative (AUC ≈ 0.73), and recall at 0.5 is about 0.5.
* Labels are per vessel, not per segment. The app cannot localise a lesion along a vessel, so the whole vessel is coloured. Territories assume typical right-dominant anatomy.
* The heart is a schematic model, not patient imaging. "Region RWMA" is a count, not a mapped location.
* SHAP explains the model, not the disease. Contributions are not causal effects.
* Hyper-parameters were set to conservative fixed values rather than tuned with nested CV, to avoid overfitting a 303-sample dataset.

## Data & credits

* **Dataset:** Alizadehsani, R., Roshanzamir, M., & Sani, Z. (2017). *Extension of Z-Alizadeh Sani dataset* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5Q31T (CC BY 4.0).
* **SHAP:** Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*; Lundberg et al. (2020), *From local explanations to global understanding with explainable AI for trees*.
* **Rendering:** three.js (MIT). The anatomy is generated procedurally, so no third-party mesh licence applies.

Built for the Multimodal AI Hackathon 2026 (KamandPrompt, IIT Mandi).
