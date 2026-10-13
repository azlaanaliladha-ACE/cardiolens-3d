import { CardioModel, TARGETS } from "./model.js";
import { HeartScene, VESSELS } from "./heart3d.js";
import { riskCss, RISK_STOPS } from "./colors.js";
import { waterfall, importanceBars, roc, calibration } from "./charts.js";

// Adult reference intervals (approximate, for display only).
const REF = {
  BMI: [18.5, 25], BP: [90, 130], PR: [60, 100], FBS: [70, 100], CR: [0.6, 1.3], TG: [0, 150],
  LDL: [0, 130], HDL: [40, 200], BUN: [7, 20], ESR: [0, 20], HB: [12, 17.5], K: [3.5, 5.0],
  Na: [135, 145], WBC: [4000, 11000], Lymph: [20, 40], Neut: [40, 70], PLT: [150, 450],
  "EF-TTE": [50, 75], "Region RWMA": [0, 0], VHD: [0, 0], "Function Class": [0, 1],
};
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const pct = (p, d = 0) => `${(p * 100).toFixed(d)}%`;
const fmtV = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString() : +(+v).toFixed(2));

const state = {
  x: null, loaded: null, loadedPred: null, patientIdx: null, explainT: "CAD", mcT: "CAD", pred: null,
};
let model, scene;

// ------------------------------------------------------------------ boot
async function boot() {
  const json = await (await fetch("model/model.json")).json();
  model = new CardioModel(json);
  try {
    if (!localStorage.getItem("cl-consent")) $("#consent").showModal();
    $("#consent").addEventListener("close", () => { try { localStorage.setItem("cl-consent", "1"); } catch {} });
  } catch { /* storage blocked: still fine */ }

  buildInputs();
  buildPresets();
  $(".legend .ramp").style.background = `linear-gradient(90deg, ${RISK_STOPS.join(",")})`;

  try {
    scene = new HeartScene($("#canvasHost"), { onSelect: onVesselSelect, onHover: onVesselHover });
  } catch (e) {
    console.error(e);
    $("#loading").textContent = "WebGL is unavailable in this browser — the 3D view is disabled, predictions still work.";
  }
  $("#loading")?.classList.add("hidden");
  wireControls();
  loadPreset("example:multi");
  renderModelCard();
  renderAbout();
}

// ------------------------------------------------------------------ inputs
function buildInputs() {
  const host = $("#featureGroups");
  const groups = {};
  model.features.forEach((f, j) => (groups[f.group] ??= []).push([f, j]));
  host.innerHTML = Object.entries(groups).map(([g, items], gi) => `
    <details class="group" ${gi < 3 ? "open" : ""}>
      <summary><span>${g}</span><span class="count">${items.length}</span></summary>
      <div class="fields">${items.map(([f, j]) => fieldHtml(f, j)).join("")}</div>
    </details>`).join("");

  host.addEventListener("input", (e) => {
    const j = +e.target.dataset.j;
    if (Number.isNaN(j)) return;
    const f = model.features[j];
    let v = f.kind === "binary" ? (e.target.checked ? 1 : 0) : +e.target.value;
    if (f.kind !== "binary") {
      v = Math.min(f.max * 1.5, Math.max(f.min * 0.5, v));
      const row = e.target.closest(".field");
      for (const inp of $$("input", row)) if (inp !== e.target) inp.value = v;
    }
    state.x[j] = v;
    markChanged(j);
    schedule();
  });
}

function fieldHtml(f, j) {
  if (f.kind === "binary") {
    return `<label class="field binary" data-key="${f.key}" data-search="${(f.label + " " + f.key).toLowerCase()}">
      <span class="fl">${f.label}</span>
      <span class="switch"><input type="checkbox" data-j="${j}" /><i></i></span></label>`;
  }
  return `<div class="field numeric" data-key="${f.key}" data-search="${(f.label + " " + f.key).toLowerCase()}">
    <div class="fl-row"><span class="fl">${f.label}</span>
      <span class="num"><input type="number" data-j="${j}" step="${f.step}" min="${f.min}" max="${f.max}" aria-label="${f.label}" /><em>${f.unit}</em></span></div>
    <input type="range" data-j="${j}" min="${f.min}" max="${f.max}" step="${f.step}" aria-label="${f.label} slider" />
    <div class="ref-line" id="ref-${j}"></div></div>`;
}

function syncInputs() {
  model.features.forEach((f, j) => {
    for (const inp of $$(`input[data-j="${j}"]`)) {
      if (f.kind === "binary") inp.checked = !!state.x[j];
      else inp.value = f.step < 1 ? +(+state.x[j]).toFixed(2) : state.x[j];
    }
    markChanged(j);
  });
}

function markChanged(j) {
  const row = $(`[data-key="${CSS.escape(model.features[j].key)}"]`);
  if (row && state.loaded) row.classList.toggle("changed", state.x[j] !== state.loaded[j]);
}

// ------------------------------------------------------------------ presets
function buildPresets() {
  const Y = model.cohort.Y; // [CAD, LAD, LCX, RCA]
  const oof = model.cohort.oof_CAD;
  const find = (pred) => Y.findIndex((y, i) => pred(y, i));
  // cohort examples whose cross-validated CAD prediction agrees with angiography
  state.examples = {
    "example:normal": find((y, i) => !y[0] && !y[1] && !y[2] && !y[3] && oof[i] < 0.2),
    "example:lad": find((y, i) => y[0] && y[1] && !y[2] && !y[3] && oof[i] > 0.7),
    "example:rca": find((y, i) => y[0] && !y[1] && !y[2] && y[3] && oof[i] > 0.7),
    "example:multi": find((y, i) => y[1] && y[2] && y[3] && oof[i] > 0.85),
  };
  const opts = [
    ["Examples from the cohort", [
      ["example:multi", "Three-vessel disease"], ["example:lad", "Single-vessel (LAD)"],
      ["example:rca", "Single-vessel (RCA)"], ["example:normal", "Normal coronaries"],
    ]],
    ["Templates", [["median", "Cohort-median patient"]]],
    ["All 303 cohort patients", Y.map((y, i) => [`p:${i}`, `#${i + 1} · ${y[0] ? "CAD" : "Normal"}${y[0] ? ` (${["LAD", "LCX", "RCA"].filter((_, k) => y[k + 1]).join("+") || "other"})` : ""}`])],
  ];
  $("#preset").innerHTML = opts.map(([g, items]) =>
    `<optgroup label="${g}">${items.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</optgroup>`).join("");
}

function loadPreset(v) {
  $("#preset").value = v;
  let idx = null;
  if (v === "median") state.x = model.medianPatient();
  else {
    idx = v.startsWith("p:") ? +v.slice(2) : state.examples[v];
    state.x = model.cohort.X[idx].slice();
  }
  state.patientIdx = idx;
  state.loaded = state.x.slice();
  state.loadedPred = model.predict(state.x);
  syncInputs();
  renderBadge();
  update();
}

function renderBadge() {
  const i = state.patientIdx;
  if (i == null) { $("#patientBadge").innerHTML = `<span class="muted">Synthetic template — no angiography label.</span>`; return; }
  const y = model.cohort.Y[i];
  const cells = TARGETS.map((t, k) => {
    const oof = model.cohort[`oof_${t}`][i];
    const agree = (oof >= 0.5) === !!y[k];
    return `<div class="gt ${y[k] ? "pos" : "neg"}" title="Cross-validated probability (model never saw this patient): ${pct(oof, 1)}">
      <b>${t}</b><span>${y[k] ? (t === "CAD" ? "CAD" : "Stenotic") : "Normal"}</span>
      <em class="${agree ? "ok" : "miss"}">${agree ? "✓" : "✗"} CV ${pct(oof)}</em></div>`;
  }).join("");
  $("#patientBadge").innerHTML = `<div class="gt-title">Cohort patient #${i + 1} · angiography ground truth</div><div class="gt-row">${cells}</div>
    <p class="muted tiny">“CV” = out-of-fold probability from a model that never saw this patient. The live prediction below uses the final model trained on all 303 patients.</p>`;
}

// ------------------------------------------------------------------ update loop
let raf = 0;
function schedule() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(); }); }

function update() {
  state.pred = model.predict(state.x);
  const P = state.pred;
  // hero
  const cad = P.CAD.p;
  $("#cadValue").textContent = pct(cad);
  $("#cadValue").style.setProperty("--c", riskCss(cad));
  $("#cadStatus").innerHTML = statusChip(cad, "CAD likely", "CAD unlikely");
  const d = cad - state.loadedPred.CAD.p;
  $("#cadDelta").textContent = Math.abs(d) >= 0.005 ? `${d > 0 ? "▲" : "▼"} ${Math.abs(d * 100).toFixed(1)} pts vs. loaded patient` : "";
  // vessel bars
  $("#vesselBars").innerHTML = ["LAD", "LCX", "RCA"].map((v) => {
    const p = P[v].p, d0 = p - state.loadedPred[v].p;
    return `<button class="vbar ${state.explainT === v ? "active" : ""}" data-v="${v}" title="${VESSELS[v].name}">
      <div class="vb-head"><b>${v}</b><span class="vb-name">${VESSELS[v].name}</span><span class="vb-p">${pct(p)}</span></div>
      <div class="vb-track"><div class="vb-fill" style="width:${p * 100}%;background:${riskCss(p)}"></div><i style="left:50%"></i></div>
      <div class="vb-foot">${statusChip(p, "≥50% stenosis likely", "Stenosis unlikely")}${Math.abs(d0) >= 0.005 ? `<span class="delta">${d0 > 0 ? "▲" : "▼"} ${Math.abs(d0 * 100).toFixed(1)}</span>` : ""}</div>
    </button>`;
  }).join("");
  $$(".vbar").forEach((b) => b.addEventListener("click", () => { setExplainTarget(b.dataset.v); scene?.select(b.dataset.v); }));

  scene?.update({ CAD: cad, LAD: P.LAD.p, LCX: P.LCX.p, RCA: P.RCA.p }, state.x[model.index.PR]);
  renderExplanation();
  renderPhysio();
  renderRefLines();
  if (scene?.selected) renderVesselCard(scene.selected);
}

function statusChip(p, hi, lo) {
  const pos = p >= 0.5;
  return `<span class="chip ${pos ? "pos" : "neg"}"><svg viewBox="0 0 16 16" aria-hidden="true">${pos
    ? '<path d="M8 2 15 14H1z"/>' : '<circle cx="8" cy="8" r="6"/>'}</svg>${pos ? hi : lo}</span>`;
}

function setExplainTarget(t) {
  state.explainT = t;
  $$("#targetSeg button").forEach((b) => b.classList.toggle("active", b.dataset.t === t));
  $$(".vbar").forEach((b) => b.classList.toggle("active", b.dataset.v === t));
  renderExplanation();
  renderPhysio();
}

function renderExplanation() {
  const t = state.explainT, p = state.pred[t];
  waterfall($("#waterfall"), p, model.features, state.x, 10);
  $("#waterfallCaption").innerHTML = `Exact SHAP contributions to the <b>${t === "CAD" ? "overall CAD" : `${t} stenosis`}</b> log-odds.
    <span class="up">Red</span> pushes the risk up, <span class="down">blue</span> pulls it down; bars start at the cohort baseline (${pct(p.pBase)}) and end at this patient's prediction.`;
  $("#additivity").innerHTML = `<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m3 8 3 3 7-7"/></svg>
    Additivity check: baseline ${p.base.toFixed(3)} + Σ SHAP ${(p.logit - p.base).toFixed(3)} = ${p.logit.toFixed(3)} log-odds (error ${p.additivityError.toExponential(1)})`;
}

function refFlag(f, v) {
  const r = REF[f.key];
  if (!r) return null;
  if (v < r[0]) return "low";
  if (v > r[1]) return "high";
  return "ok";
}

function renderPhysio() {
  const t = state.explainT, phi = state.pred[t].phi;
  const only = $("#onlyAbnormal").checked;
  const maxAbs = Math.max(...phi.map(Math.abs), 1e-6);
  const rows = model.features.map((f, j) => ({ f, j, v: state.x[j], phi: phi[j] }))
    .filter((r) => !only || (r.f.kind === "binary" ? r.v === 1 : ["low", "high"].includes(refFlag(r.f, r.v))))
    .sort((a, b) => Math.abs(b.phi) - Math.abs(a.phi));
  $("#physioTable").innerHTML = `<table><thead><tr><th>Measurement</th><th>Value</th><th title="Percentile within the 303-patient cohort">Cohort pct.</th><th>Contribution (${t})</th></tr></thead><tbody>${
    rows.map(({ f, j, v, phi: c }) => {
      const flag = f.kind === "binary" ? (v ? "present" : "") : refFlag(f, v);
      const r = REF[f.key];
      const val = f.kind === "binary" ? (v ? "Yes" : "No") : `${fmtV(v)} <em>${f.unit}</em>`;
      const w = (Math.abs(c) / maxAbs) * 50;
      return `<tr class="${flag === "high" || flag === "low" ? "abn" : ""}">
        <td><span class="fname">${f.label}</span>${r && f.kind !== "binary" ? `<span class="rng">ref ${fmtV(r[0])}–${fmtV(r[1])}</span>` : ""}</td>
        <td class="v">${val}${flag === "high" ? ' <b class="flag">H</b>' : flag === "low" ? ' <b class="flag">L</b>' : ""}</td>
        <td class="pc">${f.kind === "binary" ? "—" : `${Math.round(model.percentile(j, v))}`}</td>
        <td class="contrib" data-tip="${f.label}: ${c >= 0 ? "+" : "−"}${Math.abs(c).toFixed(3)} log-odds">
          <div class="cbar"><i class="mid"></i><span style="${c >= 0 ? `left:50%;width:${w}%` : `right:50%;width:${w}%`};background:${c >= 0 ? "var(--up)" : "var(--down)"}"></span></div>
          <span class="cval">${c >= 0 ? "+" : "−"}${Math.abs(c).toFixed(2)}</span></td></tr>`;
    }).join("")}</tbody></table>`;
}

function renderRefLines() {
  model.features.forEach((f, j) => {
    const el = document.getElementById(`ref-${j}`);
    if (!el) return;
    const flag = refFlag(f, state.x[j]);
    const p = Math.round(model.percentile(j, state.x[j]));
    el.innerHTML = `<span>${p}th pct. of cohort</span>${flag && flag !== "ok" ? `<b class="flag">${flag === "high" ? "above" : "below"} ref.</b>` : ""}`;
  });
}

// ------------------------------------------------------------------ 3D callbacks
function onVesselSelect(v) {
  setExplainTarget(v);
  renderVesselCard(v);
}

function renderVesselCard(v) {
  const p = state.pred[v];
  const top = [...p.phi.keys()].sort((a, b) => Math.abs(p.phi[b]) - Math.abs(p.phi[a])).slice(0, 4);
  const card = $("#vesselCard");
  card.classList.remove("hidden");
  card.innerHTML = `<button class="x" aria-label="Close">×</button>
    <div class="vc-head"><span class="sw" style="background:${riskCss(p.p)}"></span><b>${v}</b> · ${VESSELS[v].name}</div>
    <div class="vc-p">${pct(p.p)} <small>probability of ≥50% stenosis</small></div>
    ${statusChip(p.p, "Stenosis likely", "Stenosis unlikely")}
    <p class="vc-sup"><b>Supplies:</b> ${VESSELS[v].supplies}.</p>
    <div class="vc-drivers"><b>Top drivers</b>${top.map((j) => `<div><span>${model.features[j].label}</span><em class="${p.phi[j] >= 0 ? "up" : "down"}">${p.phi[j] >= 0 ? "▲" : "▼"} ${Math.abs(p.phi[j]).toFixed(2)}</em></div>`).join("")}</div>
    <p class="muted tiny">Colour encodes the vessel-level probability. The dataset has no lesion position, so the whole vessel is coloured; the shaded wall is its typical perfusion territory.</p>`;
  card.querySelector(".x").onclick = () => { card.classList.add("hidden"); scene.selected = null; scene.paintMyocardium(); scene.refreshHighlight(); };
}

function onVesselHover(hit, e) {
  const tip = $("#hoverTip");
  if (!hit || !e) { tip.classList.add("hidden"); return; }
  const p = state.pred[hit.vessel].p;
  const what = hit.kind === "vessel" ? (hit.segment === "LM" ? "Left main → LAD / LCX" : `${hit.vessel}${hit.segment !== hit.vessel ? ` · ${segName(hit.segment)}` : ""}`) : `${hit.vessel} territory (myocardium)`;
  tip.innerHTML = `<b>${what}</b><span>${pct(p)} stenosis probability · click to inspect</span>`;
  const r = $("#canvasHost").getBoundingClientRect();
  tip.style.left = `${e.clientX - r.left + 14}px`;
  tip.style.top = `${e.clientY - r.top + 14}px`;
  tip.classList.remove("hidden");
}
const segName = (s) => ({ D1: "1st diagonal", D2: "2nd diagonal", OM1: "1st obtuse marginal", OM2: "2nd obtuse marginal", AM: "acute marginal", PDA: "posterior descending", PL: "posterolateral" }[s] || s);

// ------------------------------------------------------------------ controls
function wireControls() {
  $("#preset").addEventListener("change", (e) => loadPreset(e.target.value));
  $("#randomBtn").addEventListener("click", () => loadPreset(`p:${Math.floor(Math.random() * model.cohort.X.length)}`));
  $("#search").addEventListener("input", (e) => {
    const q = e.target.value.trim().toLowerCase();
    $$(".field").forEach((f) => f.classList.toggle("hidden", q && !f.dataset.search.includes(q)));
    $$(".group").forEach((g) => { if (q) g.open = true; g.classList.toggle("hidden", q && !$$(".field:not(.hidden)", g).length); });
  });
  $$("#targetSeg button").forEach((b) => b.addEventListener("click", () => setExplainTarget(b.dataset.t)));
  $("#onlyAbnormal").addEventListener("change", renderPhysio);
  $$("[data-cam]").forEach((b) => b.addEventListener("click", () => {
    $$("[data-cam]").forEach((x) => x.classList.toggle("active", x === b));
    scene?.setView(b.dataset.cam);
  }));
  const tg = { tgTerritory: "territory", tgBody: "body", tgFlow: "flow", tgBeat: "beat", tgSpin: "spin" };
  for (const [id, k] of Object.entries(tg)) $(`#${id}`).addEventListener("change", (e) => scene?.setOption(k, e.target.checked));

  $$(".tab").forEach((b) => b.addEventListener("click", () => {
    $$(".tab").forEach((x) => { x.classList.toggle("active", x === b); x.setAttribute("aria-selected", x === b); });
    for (const v of ["explorer", "modelcard", "about"]) $(`#view-${v}`).classList.toggle("hidden", v !== b.dataset.view);
    if (b.dataset.view === "explorer") scene?.resize();
  }));
  $$("#mcSeg button").forEach((b) => b.addEventListener("click", () => {
    $$("#mcSeg button").forEach((x) => x.classList.toggle("active", x === b));
    state.mcT = b.dataset.t;
    renderModelCard();
  }));

  // shared tooltip for chart marks
  const tip = $("#tooltip");
  document.addEventListener("pointerover", (e) => {
    const m = e.target.closest("[data-tip]");
    if (!m) { tip.classList.add("hidden"); return; }
    tip.innerHTML = m.dataset.tip;
    tip.classList.remove("hidden");
  });
  document.addEventListener("pointermove", (e) => {
    if (tip.classList.contains("hidden")) return;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${x}px`;
    tip.style.top = `${e.clientY + 16}px`;
  });
}

// ------------------------------------------------------------------ model card
const MODEL_NAMES = { "LogReg-L2": "Logistic regression (L2)", "LogReg-L1": "Logistic regression (L1)", RandomForest: "Random forest", HistGB: "Hist. gradient boosting", XGBoost: "XGBoost (depth 2)", "Ensemble(LR+XGB)": "Deployed: L1-LR + XGBoost ensemble" };

function renderModelCard() {
  const t = state.mcT, e = model.json.targets[t], m = e.metrics, cv = m.cv;
  const what = t === "CAD" ? "overall coronary artery disease (Cath label)" : `≥50% stenosis of the ${VESSELS[t].name} (${t})`;
  $("#mcLede").innerHTML = `Predicting <b>${what}</b>. Prevalence in the cohort: <b>${pct(m.prevalence)}</b>. All metrics are on held-out folds; the 0.5 threshold is used for the class metrics.`;
  const stat = (k, l) => `<div class="stat"><div class="sv">${cv[k][0].toFixed(3)}</div><div class="sl">${l}</div><div class="ss">± ${cv[k][1].toFixed(3)}</div></div>`;
  $("#mcStats").innerHTML = stat("roc_auc", "ROC-AUC") + stat("accuracy", "Accuracy") + stat("precision", "Precision") + stat("recall", "Recall") + stat("f1", "F1-score") + stat("brier", "Brier score ↓");
  roc($("#rocChart"), m.roc, m.oof_single_split.roc_auc);
  calibration($("#calChart"), m.calibration);
  importanceBars($("#globalImp"), e.global_importance, model.features, 15);

  const bench = model.json.benchmark[t];
  const keys = ["roc_auc", "accuracy", "precision", "recall", "f1", "brier"];
  const best = Object.fromEntries(keys.map((k) => {
    const vals = Object.values(bench).map((b) => b[k][0]);
    return [k, k === "brier" ? Math.min(...vals) : Math.max(...vals)];
  }));
  $("#benchTable").innerHTML = `<table class="bench"><thead><tr><th>Model</th><th>ROC-AUC</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Brier ↓</th></tr></thead><tbody>${
    Object.entries(bench).map(([name, b]) => `<tr class="${name.startsWith("Ensemble") ? "deployed" : ""}"><td>${MODEL_NAMES[name] || name}</td>${
      keys.map((k) => `<td class="${Math.abs(b[k][0] - best[k]) < 1e-9 ? "best" : ""}">${b[k][0].toFixed(3)}<small> ±${b[k][1].toFixed(3)}</small></td>`).join("")}</tr>`).join("")
  }</tbody></table><p class="muted tiny">Bold = best in column. The ensemble is not the single best model for every vessel, but it is the most consistent across all four targets and its SHAP values are exact.</p>`;

  // live fidelity check: compute SHAP for every cohort patient in the browser
  let maxErr = 0, maxPy = 0;
  const t0 = performance.now();
  model.cohort.X.forEach((x, i) => {
    const r = model.predictTarget(t, x);
    maxErr = Math.max(maxErr, r.additivityError);
    maxPy = Math.max(maxPy, Math.abs(r.p - model.cohort[`py_${t}`][i]));
  });
  const ms = performance.now() - t0;
  $("#fidelity").innerHTML = `<ul class="checks">
    <li><b>Exact, not approximate.</b> TreeSHAP for the XGBoost member is computed exactly in your browser by subset enumeration (depth-2 trees use ≤3 features); the logistic member's SHAP is closed-form. Offline, the implementation matches XGBoost's native <code>pred_contribs</code> to <b>&lt; 3×10⁻⁷</b> on all targets.</li>
    <li><b>Additivity verified live:</b> across all 303 patients, |baseline + ΣSHAP − model log-odds| ≤ <b>${maxErr.toExponential(1)}</b> (computed just now in ${ms.toFixed(0)} ms).</li>
    <li><b>Browser = Python:</b> the in-browser model reproduces the Python/XGBoost probabilities for all 303 patients to within <b>${maxPy.toExponential(1)}</b>.</li>
    <li><b>No target leakage:</b> LAD, LCX, RCA and Cath are excluded from every model's inputs.</li></ul>`;
}

// ------------------------------------------------------------------ about
function renderAbout() {
  const b = model.json.benchmark;
  const row = (t) => `<tr><td>${t}</td><td>${b[t]["Ensemble(LR+XGB)"].roc_auc[0].toFixed(3)}</td><td>${b[t]["Ensemble(LR+XGB)"].f1[0].toFixed(3)}</td><td>${b[t]["Ensemble(LR+XGB)"].accuracy[0].toFixed(3)}</td></tr>`;
  $("#aboutContent").innerHTML = `
  <h2>How CardioLens 3D works</h2>
  <p class="lede">Raw risk percentages are hard to picture. CardioLens predicts overall CAD and vessel-level stenosis from routine clinical data, then puts each probability on the coronary artery it belongs to, so you can see where the model thinks disease is.</p>
  <ol class="pipeline">
    <li><b>Data.</b> Extension of the Z-Alizadeh Sani dataset (UCI #411): 303 patients, 55 inputs across demographics, risk factors, examination, symptoms, ECG, labs and echocardiography. Targets: overall CAD (Cath) and ≥50% stenosis of the LAD, LCX and RCA. Those four columns are never used as inputs.</li>
    <li><b>Models.</b> One model per target. Each one averages, in log-odds, a sparse L1 logistic regression (captures linear clinical risk) with a shallow depth-2 XGBoost (captures thresholds and pairwise interactions). Five model families were benchmarked with 5×5 repeated stratified cross-validation.</li>
    <li><b>Explanations.</b> Because both members are explained in log-odds, their SHAP values add exactly. The browser computes exact SHAP for every input change (no server, no sampling), and the UI verifies additivity live.</li>
    <li><b>3D mapping.</b> A procedural heart routes the LAD (with diagonals), LCX (with obtuse marginals) and RCA (with acute marginal, PDA and posterolateral branches) along the real interventricular and atrioventricular grooves. Every surface vertex is softly assigned to its supplying artery, so both the vessel and its myocardial territory take the vessel's risk colour. The left main is coloured by the worse of LAD and LCX.</li>
    <li><b>Real-time loop.</b> Any slider change re-runs all four models and their SHAP values (≈1 ms), recolours the vessels and territories, slows the flow particles in high-risk vessels, and beats the heart at the patient's own pulse rate.</li>
  </ol>
  <h3>Cross-validated performance (deployed ensemble)</h3>
  <table class="bench"><thead><tr><th>Target</th><th>ROC-AUC</th><th>F1</th><th>Accuracy</th></tr></thead><tbody>${TARGETS.map(row).join("")}</tbody></table>
  <h3>Limitations: please read</h3>
  <ul>
    <li>Single-centre, single-cohort data (303 patients, Iran). There is no external validation, so performance on other populations is unknown.</li>
    <li>LCX and RCA are much harder to predict (AUC ≈ 0.73) than overall CAD (≈ 0.93). Vessel-level probabilities are only moderately informative.</li>
    <li>Labels are vessel-level. The model cannot say <em>where</em> along a vessel a lesion is, so the whole vessel is coloured. Perfusion territories assume a typical right-dominant anatomy.</li>
    <li>The heart is schematic, not patient-specific imaging. Echo inputs such as “Region RWMA” are counts, not mapped locations.</li>
    <li>SHAP describes the model, not causation: a large contribution does not mean changing that factor would change the patient's disease.</li>
  </ul>
  <h3>Credits</h3>
  <p class="muted">Dataset: Alizadehsani R. et al., “Extension of Z-Alizadeh Sani dataset”, UCI Machine Learning Repository, 2017 (CC BY 4.0). Rendering: three.js. Explanations: SHAP (Lundberg &amp; Lee, 2017; Lundberg et al., 2020). Built for the Multimodal AI Hackathon 2026 (Track A).</p>`;
}

boot().catch((e) => {
  console.error(e);
  document.body.insertAdjacentHTML("beforeend", `<div class="fatal">Failed to start: ${e.message}</div>`);
});
