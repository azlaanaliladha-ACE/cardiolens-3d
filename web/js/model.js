// In-browser inference + exact SHAP for the per-target ensembles exported by ml/train.py.
//
//   logit(x) = w * LR(x) + (1 - w) * XGB(x)
//   phi(x)   = w * phi_LR(x) + (1 - w) * phi_XGB(x)       (log-odds space)
//   base + sum(phi) == logit(x)                            (checked live in the UI)
//
// phi_LR is exact for a linear model with a standardised input (E[z] = 0).
// phi_XGB is path-dependent TreeSHAP, computed exactly per tree by enumerating
// subsets of the (at most 3) features a depth-2 tree splits on. This matches
// XGBoost's own pred_contribs to ~1e-7 (see ml/results/shap_validation.json).

export const TARGETS = ["CAD", "LAD", "LCX", "RCA"];

const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const f32 = Math.fround; // XGBoost compares features as float32

function prepTree(t) {
  const used = [...new Set(t.f.filter((f) => f >= 0))].sort((a, b) => a - b);
  const M = used.length;
  const fact = [1, 1, 2, 6, 24, 120, 720];
  // Shapley weight for a coalition of size k (excluding the feature itself)
  const w = Array.from({ length: M }, (_, k) => (fact[k] * fact[M - k - 1]) / fact[M]);
  return { ...t, used, M, w, vals: new Float64Array(1 << M) };
}

function condExp(t, x, mask, i = 0) {
  const f = t.f[i];
  if (f < 0) return t.val[i];
  const pos = t.used.indexOf(f);
  if (mask & (1 << pos)) {
    return condExp(t, x, mask, f32(x[f]) < t.thr[i] ? t.yes[i] : t.no[i]);
  }
  const y = t.yes[i], n = t.no[i];
  return (t.cover[y] * condExp(t, x, mask, y) + t.cover[n] * condExp(t, x, mask, n)) / t.cover[i];
}

function treeShapInto(t, x, phi) {
  const full = 1 << t.M;
  for (let m = 0; m < full; m++) t.vals[m] = condExp(t, x, m);
  for (let j = 0; j < t.M; j++) {
    const bit = 1 << j;
    let s = 0;
    for (let m = 0; m < full; m++) {
      if (m & bit) continue;
      let k = 0;
      for (let b = m; b; b &= b - 1) k++;
      s += t.w[k] * (t.vals[m | bit] - t.vals[m]);
    }
    phi[t.used[j]] += s;
  }
  return t.vals[full - 1]; // == tree prediction f(x)
}

export class CardioModel {
  constructor(json) {
    this.json = json;
    this.features = json.features;
    this.n = json.features.length;
    this.index = Object.fromEntries(json.features.map((f, i) => [f.key, i]));
    this.targets = {};
    for (const t of TARGETS) {
      const e = json.targets[t];
      const trees = e.xgb.trees.map(prepTree);
      // E[f] of each tree under the training distribution (empty coalition)
      const xgbBase = e.xgb.bias + trees.reduce((s, tr) => s + condExp(tr, [], 0), 0);
      this.targets[t] = { ...e, trees, xgbBase };
    }
    this.cohort = json.cohort;
    // sorted columns for percentile lookups
    this.sortedCols = Array.from({ length: this.n }, (_, j) =>
      json.cohort.X.map((r) => r[j]).sort((a, b) => a - b));
  }

  predictTarget(t, x) {
    const e = this.targets[t];
    const { w, lr } = e;
    const phiLR = new Float64Array(this.n);
    let lrLogit = lr.intercept;
    for (let j = 0; j < this.n; j++) {
      const c = (lr.coef[j] * (x[j] - lr.mean[j])) / lr.scale[j];
      phiLR[j] = c;
      lrLogit += c;
    }
    const phiX = new Float64Array(this.n);
    let xgbLogit = e.xgb.bias;
    for (const tr of e.trees) xgbLogit += treeShapInto(tr, x, phiX);

    const phi = new Float64Array(this.n);
    for (let j = 0; j < this.n; j++) phi[j] = w * phiLR[j] + (1 - w) * phiX[j];
    const base = w * lr.intercept + (1 - w) * e.xgbBase;
    const logit = w * lrLogit + (1 - w) * xgbLogit;
    const sumPhi = phi.reduce((a, b) => a + b, 0);
    return {
      target: t, p: sigmoid(logit), logit, base, pBase: sigmoid(base), phi,
      additivityError: Math.abs(base + sumPhi - logit),
    };
  }

  predict(x) {
    return Object.fromEntries(TARGETS.map((t) => [t, this.predictTarget(t, x)]));
  }

  percentile(j, v) {
    const col = this.sortedCols[j];
    let lo = 0, hi = col.length;
    while (lo < hi) { const m = (lo + hi) >> 1; if (col[m] < v) lo = m + 1; else hi = m; }
    let eq = lo;
    while (eq < col.length && col[eq] === v) eq++;
    return (100 * (lo + eq)) / 2 / col.length; // mid-rank percentile
  }

  medianPatient() {
    return this.features.map((f) => f.median);
  }
}
