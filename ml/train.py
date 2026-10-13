"""Train the final per-target ensembles and export everything the web app needs.

For each target (CAD, LAD, LCX, RCA):
  logit = 0.5 * L1-LogReg(x) + 0.5 * XGBoost(x)        (log-odds averaging)

Because both members are explained in log-odds space, SHAP values add up exactly:
  phi_ens = 0.5 * phi_lr + 0.5 * phi_xgb,  base_ens + sum(phi_ens) == logit(x)

Exports
  web/model/model.json   - weights, trees, feature metadata, cohort, metrics, ROC curves
  ml/models/*.joblib     - fitted sklearn / xgboost objects
  ml/results/*.json      - CV metrics and SHAP validation report
"""
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold

from benchmark import LogitEnsemble, SEED, cv_eval, models, scores
from preprocess import TARGETS, build, feature_meta

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "model"
MODELS = ROOT / "ml" / "models"
RESULTS = ROOT / "ml" / "results"
W = 0.5


# ---------------------------------------------------------------- tree export
def export_trees(booster, feat_names):
    """Flatten XGBoost JSON dumps into compact arrays (one dict per tree)."""
    trees = []
    for dump in booster.get_dump(dump_format="json", with_stats=True):
        node = json.loads(dump)
        nodes = {}

        def walk(n):
            nodes[n["nodeid"]] = n
            for c in n.get("children", []):
                walk(c)
        walk(node)
        n_max = max(nodes) + 1
        t = {"f": [-1] * n_max, "thr": [0.0] * n_max, "yes": [-1] * n_max, "no": [-1] * n_max,
             "val": [0.0] * n_max, "cover": [0.0] * n_max}
        for i, n in nodes.items():
            t["cover"][i] = float(n["cover"])
            if "leaf" in n:
                t["val"][i] = float(n["leaf"])
            else:
                sp = n["split"]  # trained on ndarray -> names are "f<idx>"
                t["f"][i] = int(sp[1:]) if sp[0] == "f" and sp[1:].isdigit() else feat_names.index(sp)
                t["thr"][i] = float(np.float32(n["split_condition"]))
                t["yes"][i], t["no"][i] = n["yes"], n["no"]
        trees.append(t)
    return trees


def tree_predict(t, x):
    i = 0
    while t["f"][i] >= 0:
        i = t["yes"][i] if np.float32(x[t["f"][i]]) < t["thr"][i] else t["no"][i]
    return t["val"][i]


def tree_cond_exp(t, x, S, i=0):
    """E[f(x) | x_S] using training cover (path-dependent TreeSHAP definition)."""
    f = t["f"][i]
    if f < 0:
        return t["val"][i]
    if f in S:
        nxt = t["yes"][i] if np.float32(x[f]) < t["thr"][i] else t["no"][i]
        return tree_cond_exp(t, x, S, nxt)
    y, n = t["yes"][i], t["no"][i]
    return (t["cover"][y] * tree_cond_exp(t, x, S, y)
            + t["cover"][n] * tree_cond_exp(t, x, S, n)) / t["cover"][i]


def tree_shap(t, x, n_feat):
    """Exact Shapley values by subset enumeration over the (few) features a tree uses."""
    phi = np.zeros(n_feat)
    used = sorted({f for f in t["f"] if f >= 0})
    M = len(used)
    cache = {}

    def v(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = tree_cond_exp(t, x, key)
        return cache[key]
    for j in used:
        others = [u for u in used if u != j]
        for k in range(M):
            w = factorial(k) * factorial(M - k - 1) / factorial(M)
            for S in combinations(others, k):
                phi[j] += w * (v(S + (j,)) - v(S))
    return phi


# ---------------------------------------------------------------- export
def export_target(ens, X, feat_names):
    lr_scaler, lr = ens.lr_.named_steps["standardscaler"], ens.lr_.named_steps["logisticregression"]
    booster = ens.xgb_.get_booster()
    trees = export_trees(booster, feat_names)
    # XGBoost adds a global bias (base_score in margin space) to the sum of leaves.
    Xv = X.values
    margins = ens.xgb_.predict(Xv, output_margin=True)
    leaf_sums = np.array([sum(tree_predict(t, x) for t in trees) for x in Xv])
    bias = float(np.median(margins - leaf_sums))
    assert np.allclose(margins, leaf_sums + bias, atol=1e-4), "tree export mismatch"
    return {
        "w": W,
        "lr": {"mean": lr_scaler.mean_.tolist(), "scale": lr_scaler.scale_.tolist(),
               "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])},
        "xgb": {"bias": bias, "trees": trees},
    }


def ens_shap(exp, x):
    lr = exp["lr"]
    z = (x - np.array(lr["mean"])) / np.array(lr["scale"])
    phi_lr = np.array(lr["coef"]) * z           # scaler is fit on the same data, so E[z] = 0
    base_lr = lr["intercept"]
    trees = exp["xgb"]["trees"]
    phi_x = sum(tree_shap(t, x, len(x)) for t in trees)
    base_x = exp["xgb"]["bias"] + sum(tree_cond_exp(t, x, frozenset()) for t in trees)
    return W * base_lr + (1 - W) * base_x, W * phi_lr + (1 - W) * phi_x, phi_x, base_x


def oof_predictions(X, y):
    """Out-of-fold probabilities for every patient (single 5-fold split)."""
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        oof[te] = LogitEnsemble(W).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return oof


def main():
    X, Y, raw = build()
    feats = list(X.columns)
    Xv = X.values
    MODELS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    WEB.mkdir(parents=True, exist_ok=True)

    out = {"features": feature_meta(X), "targets": {}, "cohort": {}, "benchmark": {}}
    validation = {}
    rng = np.random.default_rng(SEED)
    for t in TARGETS:
        y = Y[t].values
        print(f"== {t}")
        ens = LogitEnsemble(W).fit(Xv, y)
        joblib.dump(ens, MODELS / f"{t}_ensemble.joblib")
        exp = export_target(ens, X, feats)
        booster = ens.xgb_.get_booster()

        # --- validate my TreeSHAP against the shap library + check additivity
        sample = rng.choice(len(Xv), 12, replace=False)
        # Reference: XGBoost's native path-dependent TreeSHAP (Lundberg et al. 2020).
        # (shap.TreeExplainer 0.49 cannot parse XGBoost 3.x base_score, so we use pred_contribs.)
        ref_phi = booster.predict(xgb.DMatrix(Xv[sample]), pred_contribs=True)[:, :-1]
        max_err, max_add = 0.0, 0.0
        for k, i in enumerate(sample):
            base, phi, phi_x, base_x = ens_shap(exp, Xv[i])
            max_err = max(max_err, float(np.abs(phi_x - ref_phi[k]).max()))
            max_add = max(max_add, abs(base + phi.sum() - ens.logit(Xv[i:i + 1])[0]))
        validation[t] = {"max_abs_diff_vs_xgboost_pred_contribs": max_err,
                         "max_additivity_error": max_add}
        print(f"   TreeSHAP vs xgboost native max abs diff = {max_err:.2e}, additivity err = {max_add:.2e}")
        assert max_err < 1e-4 and max_add < 1e-4

        # --- global importance: mean |SHAP| over the cohort
        all_phi = np.array([ens_shap(exp, x)[1] for x in Xv])
        exp["base"] = float(ens_shap(exp, Xv[0])[0])
        exp["global_importance"] = np.abs(all_phi).mean(0).tolist()

        # --- honest metrics: repeated CV (5x5) + OOF ROC curve
        cv = cv_eval(lambda: LogitEnsemble(W), Xv, y)
        oof = oof_predictions(Xv, y)
        fpr, tpr, _ = roc_curve(y, oof)
        idx = np.unique(np.linspace(0, len(fpr) - 1, 60).astype(int))
        bins = np.linspace(0, 1, 6)
        calib = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (oof >= lo) & (oof < hi if hi < 1 else oof <= hi)
            if m.sum():
                calib.append([float(oof[m].mean()), float(y[m].mean()), int(m.sum())])
        exp["metrics"] = {"cv": cv, "oof_single_split": scores(y, oof),
                          "roc": [fpr[idx].round(4).tolist(), tpr[idx].round(4).tolist()],
                          "calibration": calib, "prevalence": float(y.mean())}
        out["cohort"][f"oof_{t}"] = oof.round(4).tolist()
        # reference probabilities from Python, used by the web app's live parity check
        out["cohort"][f"py_{t}"] = ens.predict_proba(Xv)[:, 1].round(8).tolist()
        out["targets"][t] = exp

        bench = {}
        for name, make in models().items():
            if name == "Ensemble(LR+XGB)":
                bench[name] = cv
            else:
                bench[name] = cv_eval(make, Xv, y)
        out["benchmark"][t] = bench
        print(f"   CV AUC {cv['roc_auc'][0]:.3f}  F1 {cv['f1'][0]:.3f}")

    out["cohort"]["X"] = Xv.tolist()  # full precision: rounding can flip tree splits
    out["cohort"]["Y"] = Y.values.tolist()
    out["meta"] = {
        "dataset": "Extension of Z-Alizadeh Sani (UCI #411), 303 patients",
        "model": "Per-target log-odds average of L1 logistic regression and depth-2 XGBoost",
        "cv": "Repeated stratified 5-fold x5 (25 folds), threshold 0.5",
        "excluded_inputs": ["LAD", "LCX", "RCA", "Cath"],
    }
    (WEB / "model.json").write_text(json.dumps(out, separators=(",", ":")))
    (RESULTS / "shap_validation.json").write_text(json.dumps(validation, indent=1))
    (RESULTS / "metrics.json").write_text(json.dumps(
        {t: {"benchmark": out["benchmark"][t], "oof": out["targets"][t]["metrics"]["oof_single_split"]}
         for t in TARGETS}, indent=1))
    print("wrote", WEB / "model.json", f"{(WEB / 'model.json').stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
