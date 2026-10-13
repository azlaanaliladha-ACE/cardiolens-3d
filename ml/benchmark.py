"""Compare model families for each target under repeated stratified 5-fold CV.

Writes ml/results/benchmark.json and prints a table. Every model is wrapped in a
Pipeline so scaling / selection is fit inside each training fold only.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

from preprocess import build

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "results"
SEED = 42


def make_lr():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.3, penalty="l1", solver="liblinear", max_iter=5000))


def make_xgb(depth=2):
    return XGBClassifier(n_estimators=300, max_depth=depth, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.6, random_state=SEED, verbosity=0)


class LogitEnsemble(BaseEstimator, ClassifierMixin):
    """Average of L1-LogReg and XGBoost in log-odds space (keeps SHAP exactly additive)."""

    def __init__(self, w=0.5, depth=2):
        self.w, self.depth = w, depth

    def fit(self, X, y):
        self.lr_, self.xgb_ = make_lr().fit(X, y), make_xgb(self.depth).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def logit(self, X):
        return (self.w * self.lr_.decision_function(X)
                + (1 - self.w) * self.xgb_.predict(X, output_margin=True))

    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.logit(X)))
        return np.c_[1 - p, p]


def models():
    return {
        "LogReg-L2": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
        "LogReg-L1": lambda: make_pipeline(StandardScaler(), LogisticRegression(
            C=0.3, penalty="l1", solver="liblinear", max_iter=5000)),
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "HistGB": lambda: HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED),
        "XGBoost": lambda: XGBClassifier(
            n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.6, random_state=SEED, verbosity=0),
        "Ensemble(LR+XGB)": lambda: LogitEnsemble(),
    }


def scores(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat), "f1": f1_score(y, yhat), "brier": brier_score_loss(y, p),
    }


def cv_eval(make, X, y, n_rep=5):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_rep, random_state=SEED)
    folds = []
    for tr, te in cv.split(X, y):
        m = make().fit(X[tr], y[tr])
        folds.append(scores(y[te], m.predict_proba(X[te])[:, 1]))
    return {k: (float(np.mean([f[k] for f in folds])), float(np.std([f[k] for f in folds])))
            for k in folds[0]}


def main():
    X, Y, _ = build()
    Xv = X.values
    res = {}
    for t in Y.columns:
        y = Y[t].values
        res[t] = {}
        for name, make in models().items():
            res[t][name] = cv_eval(make, Xv, y)
            m = res[t][name]
            print(f"{t:4s} {name:13s} AUC {m['roc_auc'][0]:.3f}±{m['roc_auc'][1]:.3f} "
                  f"F1 {m['f1'][0]:.3f} Acc {m['accuracy'][0]:.3f} Brier {m['brier'][0]:.3f}")
    OUT.mkdir(exist_ok=True)
    (OUT / "benchmark.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
