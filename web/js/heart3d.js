// Procedural, anatomically-oriented heart + coronary tree in Three.js.
//
// The ventricular surface is an analytic function S(theta, phi) of polar angle theta
// (0 = base, PI = apex) and azimuth phi (0 = anterior, +PI/2 = patient's left).
// Coronary arteries are routed in (theta, phi) space along the real grooves
// (anterior/posterior interventricular, left/right atrioventricular) and lifted onto
// the surface, so every vessel is a named, selectable object that stays glued to the
// myocardium. Perfusion territories are assigned to every surface vertex by soft
// nearest-vessel weighting, so the wall itself can be coloured by vessel risk.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { riskColor } from "./colors.js";

const AV = 0.95;                 // theta of the atrioventricular groove
const ANT_IV = 0.45;             // phi of the anterior interventricular groove
const POST_IV = -2.95;           // phi of the posterior interventricular groove

export const VESSELS = {
  LAD: { name: "Left anterior descending", supplies: "Anterior wall, anterior two-thirds of the septum and the apex" },
  LCX: { name: "Left circumflex", supplies: "Lateral and posterolateral walls of the left ventricle" },
  RCA: { name: "Right coronary artery", supplies: "Right ventricle, inferior wall and posterior third of the septum (right-dominant)" },
};

// ---------------------------------------------------------------- surface
function angDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function radius(theta, phi) {
  let r = 1;
  r -= 0.06 * Math.exp(-(((theta - AV) / 0.085) ** 2));                       // AV groove
  if (theta > AV) {
    const fade = Math.min(1, (theta - AV) / 0.3) * Math.min(1, (Math.PI - theta) / 0.5);
    r -= 0.035 * fade * Math.exp(-((angDiff(phi, ANT_IV) / 0.13) ** 2));      // anterior IV groove
    r -= 0.03 * fade * Math.exp(-((angDiff(phi, POST_IV) / 0.13) ** 2));     // posterior IV groove
  } else {
    r *= 0.9;                                                                 // atrial cap sits inside
  }
  return r;
}

function shape(p) {
  // unit-sphere point -> heart-like ventricle (pointed apex, bulkier LV)
  const y = p.y;
  const taper = y < 0 ? 1 + y * 0.42 : 1 - y * y * 0.08;
  p.x *= taper * (p.x > 0 ? 1.1 : 0.97);
  p.z *= taper * (p.z > 0 ? 0.93 : 0.86);
  p.y *= 1.3;
  return p;
}

export function surf(theta, phi, lift = 0) {
  const r = radius(theta, phi) + lift;
  const d = new THREE.Vector3(Math.sin(theta) * Math.sin(phi), Math.cos(theta), Math.sin(theta) * Math.cos(phi));
  return shape(d.multiplyScalar(r));
}

// ---------------------------------------------------------------- coronary tree (theta, phi)
const TREE = {
  LM:  { vessel: "LAD", r: 0.05, pts: [[0.6, 0.15], [0.74, 0.5], [0.86, 0.78]], shared: true },
  LAD: { vessel: "LAD", r: 0.042, pts: [[0.86, 0.78], [1.02, 0.56], [1.35, 0.46], [1.85, 0.45], [2.35, 0.48], [2.8, 0.62], [3.03, 1.6], [3.0, 2.6]] },
  D1:  { vessel: "LAD", r: 0.026, pts: [[1.45, 0.46], [1.75, 0.78], [2.1, 1.02], [2.45, 1.18]] },
  D2:  { vessel: "LAD", r: 0.022, pts: [[2.0, 0.46], [2.3, 0.76], [2.62, 0.96]] },
  LCX: { vessel: "LCX", r: 0.04, pts: [[0.86, 0.78], [0.93, 1.1], [0.955, 1.6], [0.96, 2.1], [0.97, 2.55], [1.0, 2.85]] },
  OM1: { vessel: "LCX", r: 0.026, pts: [[0.955, 1.55], [1.35, 1.6], [1.85, 1.66], [2.3, 1.7]] },
  OM2: { vessel: "LCX", r: 0.023, pts: [[0.965, 2.2], [1.45, 2.25], [1.95, 2.3], [2.3, 2.32]] },
  RCA: { vessel: "RCA", r: 0.044, pts: [[0.6, -0.25], [0.82, -0.55], [0.94, -0.95], [0.96, -1.55], [0.96, -2.15], [0.97, -2.65], [1.0, -2.95]] },
  AM:  { vessel: "RCA", r: 0.025, pts: [[0.96, -1.5], [1.4, -1.58], [1.9, -1.66], [2.35, -1.7]] },
  PDA: { vessel: "RCA", r: 0.032, pts: [[1.0, -2.95], [1.4, -2.95], [1.9, -2.93], [2.4, -2.9], [2.8, -2.78]] },
  PL:  { vessel: "RCA", r: 0.022, pts: [[1.0, -2.95], [1.05, 2.95], [1.25, 2.75], [1.6, 2.6]] },
};

function unwrapPhi(pts) {
  // keep phi continuous so interpolation never jumps across +-PI
  const out = [pts[0].slice()];
  for (let i = 1; i < pts.length; i++) {
    const prev = out[i - 1][1];
    out.push([pts[i][0], prev + angDiff(pts[i][1], prev)]);
  }
  return out;
}

function vesselCurve(pts, lift) {
  const param = new THREE.CatmullRomCurve3(unwrapPhi(pts).map(([t, p]) => new THREE.Vector3(t, p, 0)));
  const samples = param.getPoints(90).map((v) => surf(v.x, v.y, lift));
  return new THREE.CatmullRomCurve3(samples);
}

// ---------------------------------------------------------------- materials
function ghostMaterial(color, power = 2.2, strength = 0.55) {
  return new THREE.ShaderMaterial({
    uniforms: { uColor: { value: new THREE.Color(color) }, uPower: { value: power }, uStrength: { value: strength } },
    vertexShader: `varying vec3 vN; varying vec3 vV;
      void main(){ vec4 mv = modelViewMatrix*vec4(position,1.0); vN = normalize(normalMatrix*normal); vV = normalize(-mv.xyz); gl_Position = projectionMatrix*mv; }`,
    fragmentShader: `uniform vec3 uColor; uniform float uPower; uniform float uStrength; varying vec3 vN; varying vec3 vV;
      void main(){ float f = pow(1.0 - abs(dot(vN, vV)), uPower); gl_FragColor = vec4(uColor, f*uStrength); }`,
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, side: THREE.DoubleSide,
  });
}

// ---------------------------------------------------------------- scene
export class HeartScene {
  constructor(host, { onSelect, onHover } = {}) {
    this.host = host;
    this.onSelect = onSelect;
    this.onHover = onHover;
    this.probs = { LAD: 0.5, LCX: 0.5, RCA: 0.5, CAD: 0.5 };
    this.pulseRate = 75;
    this.opts = { territory: true, body: true, flow: true, beat: true, spin: false };
    this.selected = null;
    this.hovered = null;

    const w = host.clientWidth, h = host.clientHeight;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.setSize(w, h);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.setSize(w, h);
    this.labels.domElement.className = "label-layer";
    host.appendChild(this.labels.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(36, w / h, 0.05, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.minDistance = 2.2;
    this.controls.maxDistance = 26;
    this.controls.autoRotateSpeed = 1.2;

    this.scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x2a1d1d, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(3, 5, 6);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x8fb4ff, 1.4);
    rim.position.set(-5, 2, -6);
    this.scene.add(rim);

    this.root = new THREE.Group();          // torso coordinates
    this.scene.add(this.root);
    this.heartPivot = new THREE.Group();    // anatomical orientation of the heart
    this.heartPivot.position.set(0.35, 0.2, 0.55);
    this.heartPivot.rotation.set(0.38, -0.5, 0.72, "YXZ");
    this.root.add(this.heartPivot);
    this.heart = new THREE.Group();         // beating group
    this.heartPivot.add(this.heart);

    this.buildHeart();
    this.buildGreatVessels();
    this.buildCoronaries();
    this.buildBody();

    this.clock = new THREE.Clock();
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.bindEvents();
    this.setView("anterior", false);
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.renderer.setAnimationLoop(() => this.tick());
  }

  // ---------------- ventricles with per-vertex territory weights
  buildHeart() {
    const NT = 110, NP = 150;
    const pos = [], uvs = [], idx = [];
    this.vertTP = [];
    for (let i = 0; i <= NT; i++) {
      const th = (i / NT) * Math.PI;
      for (let j = 0; j <= NP; j++) {
        const ph = (j / NP) * 2 * Math.PI - Math.PI;
        const p = surf(th, ph);
        pos.push(p.x, p.y, p.z);
        uvs.push(j / NP, i / NT);
        this.vertTP.push([th, ph]);
      }
    }
    for (let i = 0; i < NT; i++) {
      for (let j = 0; j < NP; j++) {
        const a = i * (NP + 1) + j, b = a + NP + 1;
        idx.push(a, b, a + 1, b, b + 1, a + 1);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
    g.setIndex(idx);
    g.computeVertexNormals();
    // weld the seam normals (phi = -PI and +PI are the same meridian)
    const n = g.attributes.normal;
    for (let i = 0; i <= NT; i++) {
      const a = i * (NP + 1), b = a + NP;
      const v = new THREE.Vector3(n.getX(a) + n.getX(b), n.getY(a) + n.getY(b), n.getZ(a) + n.getZ(b)).normalize();
      n.setXYZ(a, v.x, v.y, v.z); n.setXYZ(b, v.x, v.y, v.z);
    }
    this.nVert = pos.length / 3;
    this.colors = new Float32Array(this.nVert * 3);
    g.setAttribute("color", new THREE.BufferAttribute(this.colors, 3));
    this.heartGeom = g;
    this.heartMesh = new THREE.Mesh(g, new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.62, metalness: 0.0,
    }));
    this.heartMesh.userData.kind = "myocardium";
    this.heart.add(this.heartMesh);

    // auricles (atrial appendages)
    const aur = new THREE.MeshStandardMaterial({ color: 0x6d5a5a, roughness: 0.7 });
    const la = new THREE.Mesh(new THREE.SphereGeometry(0.2, 32, 24), aur);
    la.scale.set(1.3, 0.7, 0.9);
    la.position.copy(surf(0.62, 0.95, 0.06));
    la.rotation.z = -0.5;
    const ra = new THREE.Mesh(new THREE.SphereGeometry(0.24, 32, 24), aur);
    ra.scale.set(1.2, 0.8, 1.0);
    ra.position.copy(surf(0.55, -0.75, 0.08));
    this.heart.add(la, ra);
  }

  buildGreatVessels() {
    const tube = (pts, r, color) => {
      const c = new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(...p)));
      const m = new THREE.Mesh(new THREE.TubeGeometry(c, 80, r, 24, false),
        new THREE.MeshStandardMaterial({ color, roughness: 0.55 }));
      this.heart.add(m);
      return m;
    };
    const art = 0x8a7470, ven = 0x6b6f80;
    tube([[-0.05, 0.75, 0.08], [-0.12, 1.3, 0.12], [-0.08, 1.85, 0.02], [0.3, 2.12, -0.3], [0.58, 1.9, -0.58], [0.62, 1.2, -0.66], [0.6, 0.0, -0.66], [0.58, -1.6, -0.6]], 0.17, art); // aorta
    tube([[-0.02, 1.95, -0.02], [-0.12, 2.5, -0.02]], 0.055, art);
    tube([[0.16, 2.08, -0.18], [0.16, 2.58, -0.18]], 0.05, art);
    tube([[0.36, 2.08, -0.36], [0.44, 2.55, -0.36]], 0.05, art);
    tube([[0.24, 0.78, 0.42], [0.34, 1.25, 0.38], [0.45, 1.5, 0.15], [0.9, 1.55, -0.12]], 0.14, 0x7a7a8c);    // pulmonary trunk -> LPA
    tube([[0.42, 1.48, 0.2], [0.05, 1.52, -0.22], [-0.55, 1.5, -0.3]], 0.11, 0x7a7a8c);                       // RPA
    tube([[-0.58, 0.7, -0.05], [-0.6, 1.4, -0.05], [-0.6, 2.2, -0.08]], 0.12, ven);                          // SVC
    tube([[-0.42, -0.6, -0.42], [-0.45, -1.2, -0.45], [-0.45, -1.9, -0.45]], 0.13, ven);                     // IVC
  }

  // ---------------- coronary tree, flow particles, labels, territory weights
  buildCoronaries() {
    this.vesselMeshes = { LAD: [], LCX: [], RCA: [] };
    this.vesselCurves = { LAD: [], LCX: [], RCA: [] };
    this.flow = [];
    for (const [seg, def] of Object.entries(TREE)) {
      const curve = vesselCurve(def.pts, def.r * 0.55);
      const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, emissive: 0x000000 });
      const mesh = new THREE.Mesh(new THREE.TubeGeometry(curve, 120, def.r, 14, false), mat);
      mesh.userData = { kind: "vessel", vessel: def.vessel, segment: seg };
      this.heart.add(mesh);
      // rounded caps
      const capGeom = new THREE.SphereGeometry(def.r, 14, 10);
      for (const tp of [0, 1]) {
        const cap = new THREE.Mesh(capGeom, mat);
        cap.position.copy(curve.getPointAt(tp));
        cap.userData = mesh.userData;
        this.heart.add(cap);
        this.vesselMeshes[def.vessel].push(cap);
      }
      this.vesselMeshes[def.vessel].push(mesh);
      if (!def.shared) this.vesselCurves[def.vessel].push(curve);

      // flow particles
      const count = Math.max(6, Math.round(curve.getLength() * 9));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(count * 3), 3));
      const pts = new THREE.Points(g, new THREE.PointsMaterial({
        color: 0xffffff, size: def.r * 2.1, transparent: true, opacity: 0.85, depthWrite: false,
        blending: THREE.AdditiveBlending,
      }));
      pts.userData.ignoreRay = true;
      this.heart.add(pts);
      this.flow.push({ curve, pts, count, vessel: def.vessel, offset: Math.random(), len: curve.getLength() });
    }
    // the left main feeds both LAD and LCX: colour it by the worse of the two
    this.leftMain = this.vesselMeshes.LAD.filter((m) => m.userData.segment === "LM");

    // labels
    this.labelEls = {};
    const anchors = { LAD: [1.9, 0.45], LCX: [0.96, 1.9], RCA: [0.96, -1.6] };
    for (const [v, [th, ph]] of Object.entries(anchors)) {
      const el = document.createElement("button");
      el.className = "vlabel";
      el.innerHTML = `<b>${v}</b><span>–</span>`;
      el.addEventListener("click", (e) => { e.stopPropagation(); this.select(v); });
      const obj = new CSS2DObject(el);
      obj.position.copy(surf(th, ph, 0.28));
      this.heart.add(obj);
      this.labelEls[v] = el;
    }

    // territory weights per ventricular vertex
    const samples = [];
    for (const v of Object.keys(this.vesselCurves)) {
      for (const c of this.vesselCurves[v]) for (const p of c.getPoints(60)) samples.push([v, p]);
    }
    const order = ["LAD", "LCX", "RCA"];
    this.territory = new Float32Array(this.nVert * 3);
    this.ventricular = new Float32Array(this.nVert);
    const pos = this.heartGeom.attributes.position;
    const p = new THREE.Vector3();
    const sigma2 = 0.09;
    for (let i = 0; i < this.nVert; i++) {
      const [th] = this.vertTP[i];
      this.ventricular[i] = THREE.MathUtils.smoothstep(th, AV + 0.02, AV + 0.16);
      p.fromBufferAttribute(pos, i);
      const best = [Infinity, Infinity, Infinity];
      for (const [v, s] of samples) {
        const k = order.indexOf(v);
        const d = p.distanceToSquared(s);
        if (d < best[k]) best[k] = d;
      }
      const m = Math.min(...best);
      let sum = 0;
      const wts = best.map((d) => { const x = Math.exp(-(d - m) / sigma2); sum += x; return x; });
      for (let k = 0; k < 3; k++) this.territory[i * 3 + k] = wts[k] / sum;
    }
  }

  // ---------------- torso, ribs, spine
  buildBody() {
    this.body = new THREE.Group();
    const prof = [[0.01, -6.2], [2.15, -6.2], [2.25, -4.5], [2.45, -3], [2.75, -1.4], [2.95, 0.2], [3.05, 1.8], [3.0, 3.0],
      [2.55, 3.8], [1.6, 4.25], [0.82, 4.45], [0.72, 4.9], [0.72, 5.8]].map(([r, y]) => new THREE.Vector2(r, y));
    const torso = new THREE.Mesh(new THREE.LatheGeometry(prof, 96), ghostMaterial(0x7fa7ff, 2.4, 0.5));
    torso.scale.z = 0.66;
    torso.renderOrder = 2;
    this.body.add(torso);

    const bone = new THREE.MeshStandardMaterial({ color: 0xd9d2c3, roughness: 0.8, transparent: true, opacity: 0.28, depthWrite: false });
    for (let i = 0; i < 10; i++) {
      const yBack = 3.2 - i * 0.62, drop = 0.55 + i * 0.05;
      const rx = 2.35 + Math.sin((i / 9) * Math.PI) * 0.45, rz = 1.62 + Math.sin((i / 9) * Math.PI) * 0.22;
      const front = 1.25 - i * 0.07; // radians from front where the cartilage ends
      for (const side of [1, -1]) {
        const pts = [];
        for (let k = 0; k <= 24; k++) {
          const a = Math.PI - (k / 24) * (Math.PI - front);   // back (PI) -> near sternum
          pts.push(new THREE.Vector3(side * rx * Math.sin(a), yBack - drop * (k / 24) ** 1.4, rz * Math.cos(a)));
        }
        const rib = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 48, 0.07, 8), bone);
        rib.renderOrder = 1;
        this.body.add(rib);
      }
    }
    const sternum = new THREE.Mesh(new THREE.BoxGeometry(0.42, 3.6, 0.14), bone);
    sternum.position.set(0, 1.45, 1.86);
    sternum.rotation.x = 0.12;
    this.body.add(sternum);
    for (let i = 0; i < 17; i++) {
      const vert = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.22, 0.32, 16), bone);
      vert.position.set(0, 4.2 - i * 0.6, -1.45 + Math.sin(i / 5) * 0.12);
      this.body.add(vert);
    }
    this.root.add(this.body);
  }

  // ---------------- state updates
  update(probs, pulseRate) {
    this.probs = { ...probs };
    if (pulseRate) this.pulseRate = Math.min(160, Math.max(40, pulseRate));
    for (const v of ["LAD", "LCX", "RCA"]) {
      const c = riskColor(probs[v]);
      for (const m of this.vesselMeshes[v]) {
        if (m.userData.segment === "LM") continue;
        m.material.color.copy(c);
      }
      const el = this.labelEls[v];
      el.querySelector("span").textContent = `${Math.round(probs[v] * 100)}%`;
      el.style.setProperty("--c", `#${c.getHexString()}`);
    }
    const lmc = riskColor(Math.max(probs.LAD, probs.LCX));
    for (const m of this.leftMain) m.material.color.copy(lmc);
    this.paintMyocardium();
    this.refreshHighlight();
  }

  paintMyocardium() {
    const base = new THREE.Color(0x8a6f6f), atrium = new THREE.Color(0x6d5a5a);
    const cs = ["LAD", "LCX", "RCA"].map((v) => riskColor(this.probs[v]));
    const sel = this.selected ? ["LAD", "LCX", "RCA"].indexOf(this.selected) : -1;
    const tmp = new THREE.Color();
    for (let i = 0; i < this.nVert; i++) {
      const vent = this.ventricular[i];
      tmp.copy(atrium).lerp(base, vent);
      if (this.opts.territory && vent > 0) {
        const r = cs[0].r * this.territory[i * 3] + cs[1].r * this.territory[i * 3 + 1] + cs[2].r * this.territory[i * 3 + 2];
        const g = cs[0].g * this.territory[i * 3] + cs[1].g * this.territory[i * 3 + 1] + cs[2].g * this.territory[i * 3 + 2];
        const b = cs[0].b * this.territory[i * 3] + cs[1].b * this.territory[i * 3 + 1] + cs[2].b * this.territory[i * 3 + 2];
        let k = 0.62 * vent;
        if (sel >= 0) k *= 0.35 + 0.65 * this.territory[i * 3 + sel];
        tmp.lerp(new THREE.Color(r, g, b), k);
      }
      this.colors[i * 3] = tmp.r; this.colors[i * 3 + 1] = tmp.g; this.colors[i * 3 + 2] = tmp.b;
    }
    this.heartGeom.attributes.color.needsUpdate = true;
  }

  refreshHighlight() {
    for (const v of ["LAD", "LCX", "RCA"]) {
      const on = v === this.selected || v === this.hovered;
      for (const m of this.vesselMeshes[v]) {
        m.material.emissive.copy(on ? m.material.color : new THREE.Color(0));
        m.material.emissiveIntensity = on ? 0.55 : 0;
      }
      this.labelEls[v].classList.toggle("active", v === this.selected);
    }
  }

  select(v) {
    this.selected = v;
    this.paintMyocardium();
    this.refreshHighlight();
    this.onSelect?.(v);
  }

  setOption(k, v) {
    this.opts[k] = v;
    if (k === "body") this.body.visible = v;
    if (k === "flow") for (const f of this.flow) f.pts.visible = v;
    if (k === "spin") this.controls.autoRotate = v;
    if (k === "territory") this.paintMyocardium();
  }

  // ---------------- camera
  setView(name, animate = true) {
    const c = this.heartPivot.getWorldPosition(new THREE.Vector3());
    const views = {
      anterior: [0, 0.4, 7.2], lao: [4.6, 0.9, 5.4], lateral: [7.2, 0.4, -0.4],
      posterior: [0, 0.4, -7.2], inferior: [0.4, -6.8, 2.0], body: [0, 1.2, 17],
    };
    const off = new THREE.Vector3(...views[name]);
    const target = name === "body" ? new THREE.Vector3(0, -0.4, 0) : c;
    const to = target.clone().add(off);
    if (!animate) {
      this.camera.position.copy(to);
      this.controls.target.copy(target);
      this.controls.update();
      return;
    }
    this.camAnim = { from: this.camera.position.clone(), to, tFrom: this.controls.target.clone(), tTo: target, t: 0 };
  }

  // ---------------- interaction
  bindEvents() {
    const el = this.renderer.domElement;
    let down = null;
    el.addEventListener("pointerdown", (e) => { down = [e.clientX, e.clientY]; });
    el.addEventListener("pointerup", (e) => {
      if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const hit = this.pick(e);
      if (hit) this.select(hit.vessel);
    });
    el.addEventListener("pointermove", (e) => {
      const hit = this.pick(e);
      const v = hit?.vessel ?? null;
      if (v !== this.hovered) { this.hovered = v; this.refreshHighlight(); }
      el.style.cursor = v ? "pointer" : "grab";
      this.onHover?.(hit, e);
    });
    el.addEventListener("pointerleave", () => { this.hovered = null; this.refreshHighlight(); this.onHover?.(null); });
  }

  pick(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const targets = [this.heartMesh, ...Object.values(this.vesselMeshes).flat()];
    const hits = this.raycaster.intersectObjects(targets, false);
    if (!hits.length) return null;
    const h = hits[0];
    if (h.object.userData.kind === "vessel") {
      const seg = h.object.userData.segment;
      if (seg === "LM") return { vessel: "LAD", segment: "LM", kind: "vessel" };
      return { vessel: h.object.userData.vessel, segment: seg, kind: "vessel" };
    }
    // myocardium -> dominant supplying artery at the nearest vertex of the hit face
    const i = h.face.a;
    if (this.ventricular[i] < 0.5) return null;
    const w = [0, 1, 2].map((k) => this.territory[i * 3 + k]);
    const k = w.indexOf(Math.max(...w));
    return { vessel: ["LAD", "LCX", "RCA"][k], kind: "territory", weight: w[k] };
  }

  resize() {
    const w = this.host.clientWidth, h = this.host.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.labels.setSize(w, h);
  }

  tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    if (this.camAnim) {
      const a = this.camAnim;
      a.t = Math.min(1, a.t + dt / 0.9);
      const k = a.t < 0.5 ? 4 * a.t ** 3 : 1 - (-2 * a.t + 2) ** 3 / 2;
      this.camera.position.lerpVectors(a.from, a.to, k);
      this.controls.target.lerpVectors(a.tFrom, a.tTo, k);
      if (a.t >= 1) this.camAnim = null;
    }
    // heartbeat at the patient's pulse rate: sharp systole, slower relaxation
    if (this.opts.beat) {
      const phase = (t * this.pulseRate / 60) % 1;
      const s = phase < 0.18 ? Math.sin((phase / 0.18) * Math.PI) : 0;
      const d = phase > 0.18 && phase < 0.42 ? Math.sin(((phase - 0.18) / 0.24) * Math.PI) * 0.25 : 0;
      const k = 1 - 0.035 * s + 0.01 * d;
      this.heart.scale.set(k, 1 - 0.02 * s, k);
    } else this.heart.scale.set(1, 1, 1);

    // flow: particles slow down in vessels with higher stenosis probability
    if (this.opts.flow) {
      for (const f of this.flow) {
        const p = this.probs[f.vessel];
        f.offset = (f.offset + dt * (0.32 * (1 - 0.75 * p)) / Math.max(f.len, 0.4)) % 1;
        const arr = f.pts.geometry.attributes.position.array;
        for (let i = 0; i < f.count; i++) {
          const u = (i / f.count + f.offset) % 1;
          const q = f.curve.getPointAt(u);
          arr[i * 3] = q.x; arr[i * 3 + 1] = q.y; arr[i * 3 + 2] = q.z;
        }
        f.pts.geometry.attributes.position.needsUpdate = true;
      }
    }
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this.labels.render(this.scene, this.camera);
  }
}
