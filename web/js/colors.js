// Diverging risk scale: blue (likely normal) <- neutral grey at the 0.5 decision
// threshold -> red (likely stenotic). Same stops are used by the 3D view, the bars
// and the CSS legend so every view reads identically.
import * as THREE from "three";

export const RISK_STOPS = ["#3987e5", "#9a9890", "#e34948"];
const stops = RISK_STOPS.map((h) => new THREE.Color(h));

export function riskColor(p) {
  const x = Math.min(1, Math.max(0, p));
  const c = new THREE.Color();
  return x < 0.5 ? c.lerpColors(stops[0], stops[1], x / 0.5) : c.lerpColors(stops[1], stops[2], (x - 0.5) / 0.5);
}

export function riskCss(p) {
  return `#${riskColor(p).getHexString()}`;
}

export const SHAP_UP = "#e66767";   // pushes risk up
export const SHAP_DOWN = "#3987e5"; // pushes risk down
