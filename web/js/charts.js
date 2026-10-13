// Small dependency-free SVG charts. Every mark carries a data-tip attribute that the
// shared tooltip in main.js renders on hover.
import { SHAP_UP, SHAP_DOWN } from "./colors.js";

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const pct = (p) => `${(p * 100).toFixed(1)}%`;
const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : +v.toFixed(2));

/** SHAP waterfall (log-odds), top-k features + "other" bucket. */
export function waterfall(el, pred, features, x, k = 10) {
  const n = features.length;
  const order = [...Array(n).keys()].sort((a, b) => Math.abs(pred.phi[b]) - Math.abs(pred.phi[a]));
  const top = order.slice(0, k);
  const rest = order.slice(k).reduce((s, j) => s + pred.phi[j], 0);
  const rows = top.map((j) => ({
    label: features[j].label,
    value: features[j].kind === "binary" ? (x[j] ? "yes" : "no") : `${fmtV(x[j])}${features[j].unit ? " " + features[j].unit : ""}`,
    phi: pred.phi[j],
  }));
  rows.push({ label: `${n - k} other features`, value: "", phi: rest, other: true });

  let cum = pred.base;
  const steps = rows.map((r) => { const s = { ...r, from: cum, to: cum + r.phi }; cum += r.phi; return s; });
  const lo = Math.min(pred.base, pred.logit, ...steps.map((s) => Math.min(s.from, s.to)));
  const hi = Math.max(pred.base, pred.logit, ...steps.map((s) => Math.max(s.from, s.to)));
  const pad = (hi - lo) * 0.08 + 0.05;

  const W = 380, rowH = 24, labelW = 168, top0 = 22, plotW = W - labelW - 46;
  const H = top0 + steps.length * rowH + 34;
  const X = (v) => labelW + ((v - (lo - pad)) / (hi - lo + 2 * pad)) * plotW;

  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="SHAP waterfall">`;
  // base and final reference lines
  s += `<line x1="${X(pred.base)}" x2="${X(pred.base)}" y1="${top0 - 6}" y2="${H - 26}" class="ref"/>`;
  s += `<text x="${X(pred.base)}" y="${top0 - 10}" class="axis-label" text-anchor="middle">baseline ${pct(pred.pBase)}</text>`;
  steps.forEach((st, i) => {
    const y = top0 + i * rowH;
    const x0 = X(Math.min(st.from, st.to)), x1 = X(Math.max(st.from, st.to));
    const up = st.phi >= 0;
    const w = Math.max(2, x1 - x0);
    const tip = `${esc(st.label)}${st.value ? ` = ${esc(st.value)}` : ""}<br>${up ? "+" : "−"}${Math.abs(st.phi).toFixed(3)} log-odds (${up ? "raises" : "lowers"} risk)`;
    s += `<g class="mark" data-tip="${esc(tip)}">`;
    s += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
    s += `<text x="${labelW - 8}" y="${y + 11}" class="row-label${st.other ? " muted" : ""}" text-anchor="end">${esc(st.label)}</text>`;
    if (st.value) s += `<text x="${labelW - 8}" y="${y + 21}" class="row-sub" text-anchor="end">${esc(st.value)}</text>`;
    s += `<rect x="${x0}" y="${y + 5}" width="${w}" height="${rowH - 10}" rx="3" fill="${st.other ? "var(--muted)" : up ? SHAP_UP : SHAP_DOWN}"/>`;
    const lx = up ? x1 + 4 : x0 - 4;
    s += `<text x="${lx}" y="${y + 16}" class="val" text-anchor="${up ? "start" : "end"}">${up ? "+" : "−"}${Math.abs(st.phi).toFixed(2)}</text>`;
    if (i < steps.length - 1) s += `<line x1="${X(st.to)}" x2="${X(st.to)}" y1="${y + rowH - 5}" y2="${y + rowH + 5}" class="connector"/>`;
    s += `</g>`;
  });
  const yEnd = top0 + steps.length * rowH;
  s += `<line x1="${X(pred.logit)}" x2="${X(pred.logit)}" y1="${top0}" y2="${yEnd + 4}" class="ref strong"/>`;
  s += `<text x="${X(pred.logit)}" y="${yEnd + 18}" class="axis-label strong" text-anchor="middle">prediction ${pct(pred.p)}</text>`;
  s += `</svg>`;
  el.innerHTML = s;
}

/** Horizontal bars of global mean |SHAP|. */
export function importanceBars(el, values, features, k = 15) {
  const order = [...values.keys()].sort((a, b) => values[b] - values[a]).slice(0, k);
  const max = values[order[0]];
  const W = 640, rowH = 22, labelW = 220, plotW = W - labelW - 60, H = order.length * rowH + 8;
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Global feature importance">`;
  order.forEach((j, i) => {
    const y = 4 + i * rowH, w = Math.max(2, (values[j] / max) * plotW);
    s += `<g class="mark" data-tip="${esc(features[j].label)}<br>mean |SHAP| = ${values[j].toFixed(3)} log-odds">`;
    s += `<rect x="0" y="${y}" width="${W}" height="${rowH}" fill="transparent"/>`;
    s += `<text x="${labelW - 8}" y="${y + 15}" class="row-label" text-anchor="end">${esc(features[j].label)}</text>`;
    s += `<rect x="${labelW}" y="${y + 5}" width="${w}" height="${rowH - 10}" rx="3" fill="var(--series-1)"/>`;
    s += `<text x="${labelW + w + 6}" y="${y + 15}" class="val">${values[j].toFixed(2)}</text></g>`;
  });
  el.innerHTML = s + `</svg>`;
}

function axes(W, H, m, xl, yl) {
  let s = "";
  for (const t of [0, 0.25, 0.5, 0.75, 1]) {
    const x = m.l + t * (W - m.l - m.r), y = H - m.b - t * (H - m.t - m.b);
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${y}" y2="${y}" class="grid"/>`;
    s += `<text x="${m.l - 6}" y="${y + 4}" class="tick" text-anchor="end">${t}</text>`;
    s += `<text x="${x}" y="${H - m.b + 16}" class="tick" text-anchor="middle">${t}</text>`;
  }
  s += `<line x1="${m.l}" x2="${W - m.r}" y1="${H - m.b}" y2="${H - m.b}" class="axis"/>`;
  s += `<text x="${(W + m.l - m.r) / 2}" y="${H - 4}" class="axis-label" text-anchor="middle">${xl}</text>`;
  s += `<text x="12" y="${(H - m.b + m.t) / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 12 ${(H - m.b + m.t) / 2})">${yl}</text>`;
  return s;
}

/** ROC curve with chance diagonal; hover shows the nearest operating point. */
export function roc(el, [fpr, tpr], auc) {
  const W = 360, H = 300, m = { l: 46, r: 14, t: 14, b: 42 };
  const X = (v) => m.l + v * (W - m.l - m.r), Y = (v) => H - m.b - v * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="ROC curve, AUC ${auc.toFixed(3)}">`;
  s += axes(W, H, m, "False positive rate", "True positive rate");
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" class="ref dashed"/>`;
  s += `<path d="${fpr.map((f, i) => `${i ? "L" : "M"}${X(f).toFixed(1)},${Y(tpr[i]).toFixed(1)}`).join("")}" class="line"/>`;
  s += `<text x="${X(0.97)}" y="${Y(0.06)}" class="annot" text-anchor="end">AUC ${auc.toFixed(3)}</text>`;
  fpr.forEach((f, i) => {
    s += `<circle cx="${X(f)}" cy="${Y(tpr[i])}" r="7" class="hit" data-tip="FPR ${f.toFixed(2)} · TPR ${tpr[i].toFixed(2)}"/>`;
  });
  el.innerHTML = s + `</svg>`;
}

/** Reliability diagram: mean predicted vs observed rate per probability bin. */
export function calibration(el, bins) {
  const W = 360, H = 300, m = { l: 46, r: 14, t: 14, b: 42 };
  const X = (v) => m.l + v * (W - m.l - m.r), Y = (v) => H - m.b - v * (H - m.t - m.b);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="Calibration plot">`;
  s += axes(W, H, m, "Mean predicted probability", "Observed stenosis rate");
  s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" class="ref dashed"/>`;
  s += `<path d="${bins.map(([p, o], i) => `${i ? "L" : "M"}${X(p)},${Y(o)}`).join("")}" class="line"/>`;
  for (const [p, o, n] of bins) {
    s += `<circle cx="${X(p)}" cy="${Y(o)}" r="5" class="dot"/>`;
    s += `<circle cx="${X(p)}" cy="${Y(o)}" r="12" class="hit" data-tip="predicted ${pct(p)} · observed ${pct(o)}<br>n = ${n} patients"/>`;
  }
  el.innerHTML = s + `</svg>`;
}
