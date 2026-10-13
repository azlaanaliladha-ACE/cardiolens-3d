"""Preprocessing for the Extension of Z-Alizadeh Sani dataset (UCI #411).

Turns the raw Excel sheet into a purely numeric feature matrix plus four binary
targets. LAD / LCX / RCA / Cath are *targets only* and are never used as inputs
(prevents target leakage, per the Track A rules).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "extention of Z-Alizadeh sani dataset.xlsx"

TARGETS = {
    "CAD": ("Cath", "CAD"),
    "LAD": ("LAD", "Stenotic"),
    "LCX": ("LCX", "Stenotic"),
    "RCA": ("RCA", "Stenotic"),
}
LEAKY = ["Cath", "LAD", "LCX", "RCA"]

# Constant in this dataset (all 'N') -> carries no information.
DROP = ["Exertional CP"]

YN_COLS = [
    "Obesity", "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF", "DLP",
    "Weak Peripheral Pulse", "Lung rales", "Systolic Murmur", "Diastolic Murmur",
    "Dyspnea", "Atypical", "Nonanginal", "LowTH Ang", "LVH", "Poor R Progression",
]
VHD_ORDER = {"N": 0, "mild": 1, "Moderate": 2, "Severe": 3}

# Feature metadata used by the web dashboard (group, unit, human label, kind).
GROUPS = {
    "Demographics": ["Age", "Sex", "Weight", "Height", "BMI", "Obesity"],
    "Risk factors & history": ["DM", "HTN", "Current Smoker", "EX-Smoker", "FH", "DLP",
                               "CRF", "CVA", "Airway disease", "Thyroid Disease", "CHF"],
    "Examination": ["BP", "PR", "Edema", "Weak Peripheral Pulse", "Lung rales",
                    "Systolic Murmur", "Diastolic Murmur"],
    "Symptoms": ["Typical Chest Pain", "Atypical", "Nonanginal", "Dyspnea",
                 "Function Class", "LowTH Ang"],
    "ECG": ["Q Wave", "St Elevation", "St Depression", "Tinversion", "LVH",
            "Poor R Progression", "LBBB", "RBBB"],
    "Laboratory": ["FBS", "CR", "TG", "LDL", "HDL", "BUN", "ESR", "HB", "K", "Na",
                   "WBC", "Lymph", "Neut", "PLT"],
    "Echocardiography": ["EF-TTE", "Region RWMA", "VHD"],
}
LABELS = {
    "Age": ("Age", "yr"), "Sex": ("Male sex", ""), "Weight": ("Weight", "kg"),
    "Height": ("Height", "cm"), "BMI": ("BMI", "kg/m²"), "Obesity": ("Obesity", ""),
    "DM": ("Diabetes mellitus", ""), "HTN": ("Hypertension", ""),
    "Current Smoker": ("Current smoker", ""), "EX-Smoker": ("Ex-smoker", ""),
    "FH": ("Family history of CAD", ""), "DLP": ("Dyslipidemia", ""),
    "CRF": ("Chronic renal failure", ""), "CVA": ("Prior stroke (CVA)", ""),
    "Airway disease": ("Airway disease", ""), "Thyroid Disease": ("Thyroid disease", ""),
    "CHF": ("Congestive heart failure", ""), "BP": ("Systolic BP", "mmHg"),
    "PR": ("Pulse rate", "bpm"), "Edema": ("Edema", ""),
    "Weak Peripheral Pulse": ("Weak peripheral pulse", ""), "Lung rales": ("Lung rales", ""),
    "Systolic Murmur": ("Systolic murmur", ""), "Diastolic Murmur": ("Diastolic murmur", ""),
    "Typical Chest Pain": ("Typical chest pain", ""), "Atypical": ("Atypical chest pain", ""),
    "Nonanginal": ("Non-anginal chest pain", ""), "Dyspnea": ("Dyspnea", ""),
    "Function Class": ("NYHA function class", "0–4"), "LowTH Ang": ("Low-threshold angina", ""),
    "Q Wave": ("Q wave", ""), "St Elevation": ("ST elevation", ""),
    "St Depression": ("ST depression", ""), "Tinversion": ("T-wave inversion", ""),
    "LVH": ("LV hypertrophy (ECG)", ""), "Poor R Progression": ("Poor R-wave progression", ""),
    "LBBB": ("Left bundle branch block", ""), "RBBB": ("Right bundle branch block", ""),
    "FBS": ("Fasting blood sugar", "mg/dL"), "CR": ("Creatinine", "mg/dL"),
    "TG": ("Triglycerides", "mg/dL"), "LDL": ("LDL cholesterol", "mg/dL"),
    "HDL": ("HDL cholesterol", "mg/dL"), "BUN": ("Blood urea nitrogen", "mg/dL"),
    "ESR": ("ESR", "mm/h"), "HB": ("Hemoglobin", "g/dL"), "K": ("Potassium", "mEq/L"),
    "Na": ("Sodium", "mEq/L"), "WBC": ("White cell count", "/mm³"),
    "Lymph": ("Lymphocytes", "%"), "Neut": ("Neutrophils", "%"),
    "PLT": ("Platelets", "×10³/mm³"), "EF-TTE": ("Ejection fraction (TTE)", "%"),
    "Region RWMA": ("Regions with wall-motion abnormality", "0–4"),
    "VHD": ("Valvular heart disease", "0 none – 3 severe"),
}


def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW, sheet_name=0)
    df.columns = [c.strip() for c in df.columns]
    return df


def build(df: pd.DataFrame | None = None):
    """Return (X, Y, raw) where X is numeric features and Y holds 4 binary targets."""
    df = load_raw() if df is None else df.copy()
    Y = pd.DataFrame({t: (df[col] == pos).astype(int) for t, (col, pos) in TARGETS.items()})

    X = df.drop(columns=LEAKY + DROP).rename(columns={"Length": "Height"})
    X["Sex"] = (X["Sex"] == "Male").astype(int)
    for c in YN_COLS:
        X[c] = (X[c] == "Y").astype(int)
    X["VHD"] = X["VHD"].map(VHD_ORDER).astype(int)
    X["LBBB"] = (X["BBB"] == "LBBB").astype(int)
    X["RBBB"] = (X["BBB"] == "RBBB").astype(int)
    X = X.drop(columns=["BBB"]).astype(float)

    order = [f for g in GROUPS.values() for f in g]
    assert set(order) == set(X.columns), set(order) ^ set(X.columns)
    assert not set(LEAKY) & set(X.columns)
    return X[order], Y, df


def feature_meta(X: pd.DataFrame) -> list[dict]:
    meta = []
    for group, feats in GROUPS.items():
        for f in feats:
            col = X[f]
            binary = set(np.unique(col)) <= {0.0, 1.0}
            label, unit = LABELS[f]
            meta.append({
                "key": f, "label": label, "unit": unit, "group": group,
                "kind": "binary" if binary else "numeric",
                "min": float(col.min()), "max": float(col.max()),
                "median": float(col.median()), "mean": float(col.mean()),
                "step": 1.0 if binary or np.allclose(col, col.round()) else 0.1,
            })
    return meta


if __name__ == "__main__":
    X, Y, _ = build()
    print(X.shape, "features;", Y.mean().round(3).to_dict(), "positive rate")
