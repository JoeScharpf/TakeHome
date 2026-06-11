/**
 * DOM wiring, canvas render loop, and controls for the Problem 1 browser demo.
 */

const L = 5.0;
const DT = 0.25;
const DEFAULT_SEED = 42;
const STEPS_PER_FRAME = 2;
const HISTORY_MAX = 200;
const ARROW_LEN = 0.2;

const simCanvas = document.getElementById("sim");
const sparkCanvas = document.getElementById("sparkline");
const simCtx = simCanvas.getContext("2d");
const sparkCtx = sparkCanvas.getContext("2d");

const els = {
  eta: document.getElementById("eta"),
  etaVal: document.getElementById("eta-val"),
  R: document.getElementById("R"),
  RVal: document.getElementById("R-val"),
  v: document.getElementById("v"),
  vVal: document.getElementById("v-val"),
  N: document.getElementById("N"),
  NVal: document.getElementById("N-val"),
  nWarn: document.getElementById("n-warn"),
  polarization: document.getElementById("polarization"),
  time: document.getElementById("time"),
  steps: document.getElementById("steps"),
  btnToggle: document.getElementById("btn-toggle"),
  btnReset: document.getElementById("btn-reset"),
  btnLow: document.getElementById("btn-low"),
  btnHigh: document.getElementById("btn-high"),
};

let sim = new VicsekSim({
  N: parseInt(els.N.value, 10),
  R: parseFloat(els.R.value),
  v: parseFloat(els.v.value),
  eta: parseFloat(els.eta.value),
  L,
  dt: DT,
  seed: DEFAULT_SEED,
});

let playing = false;
let animId = null;
let pHistory = [];
let pendingN = null;

function readParams() {
  return {
    N: parseInt(els.N.value, 10),
    R: parseFloat(els.R.value),
    v: parseFloat(els.v.value),
    eta: parseFloat(els.eta.value),
  };
}

function syncLabels() {
  els.etaVal.textContent = parseFloat(els.eta.value).toFixed(2);
  els.RVal.textContent = parseFloat(els.R.value).toFixed(2);
  els.vVal.textContent = parseFloat(els.v.value).toFixed(2);
  els.NVal.textContent = els.N.value;
  els.nWarn.hidden = parseInt(els.N.value, 10) <= 200;
}

function applyLiveParams() {
  const { R, v, eta } = readParams();
  sim.R = R;
  sim.v = v;
  sim.eta = eta;
}

function resizeCanvas(canvas, ctx) {
  const size = Math.max(200, Math.min(canvas.clientWidth || 480, 520));
  const dpr = window.devicePixelRatio || 1;
  canvas.width = size * dpr;
  canvas.height = size * dpr;
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return size;
}

function worldToScreen(x, y, size) {
  const scale = size / L;
  return {
    sx: x * scale,
    sy: (L - y) * scale,
  };
}

function drawSimulation(size) {
  simCtx.clearRect(0, 0, size, size);
  simCtx.fillStyle = "#0f1419";
  simCtx.fillRect(0, 0, size, size);

  const scale = size / L;
  simCtx.strokeStyle = "#3d4f5f";
  simCtx.lineWidth = 2;
  simCtx.strokeRect(0.5, 0.5, size - 1, size - 1);

  simCtx.strokeStyle = "rgba(61, 79, 95, 0.35)";
  simCtx.lineWidth = 1;
  for (let g = 1; g < 5; g += 1) {
    const p = g * scale;
    simCtx.beginPath();
    simCtx.moveTo(p, 0);
    simCtx.lineTo(p, size);
    simCtx.stroke();
    simCtx.beginPath();
    simCtx.moveTo(0, p);
    simCtx.lineTo(size, p);
    simCtx.stroke();
  }

  const radius = Math.max(2, 4 - sim.N / 80);
  const arrowPx = ARROW_LEN * scale;

  for (let i = 0; i < sim.N; i += 1) {
    const { sx, sy } = worldToScreen(sim.x[i], sim.y[i], size);

    simCtx.beginPath();
    simCtx.fillStyle = "#5b9bd5";
    simCtx.arc(sx, sy, radius, 0, Math.PI * 2);
    simCtx.fill();

    const tx = sx + arrowPx * Math.cos(sim.theta[i]);
    const ty = sy - arrowPx * Math.sin(sim.theta[i]);
    simCtx.strokeStyle = "#e85d5d";
    simCtx.lineWidth = 1.2;
    simCtx.beginPath();
    simCtx.moveTo(sx, sy);
    simCtx.lineTo(tx, ty);
    simCtx.stroke();
  }
}

function drawSparkline() {
  const w = sparkCanvas.clientWidth;
  const h = sparkCanvas.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  sparkCanvas.width = w * dpr;
  sparkCanvas.height = h * dpr;
  sparkCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  sparkCtx.clearRect(0, 0, w, h);

  sparkCtx.fillStyle = "#f8fafc";
  sparkCtx.fillRect(0, 0, w, h);
  sparkCtx.strokeStyle = "#e2e8f0";
  sparkCtx.lineWidth = 1;
  sparkCtx.strokeRect(0.5, 0.5, w - 1, h - 1);

  if (pHistory.length < 2) return;

  sparkCtx.strokeStyle = "#2563eb";
  sparkCtx.lineWidth = 1.5;
  sparkCtx.beginPath();
  const n = pHistory.length;
  for (let i = 0; i < n; i += 1) {
    const x = (i / (HISTORY_MAX - 1)) * (w - 8) + 4;
    const y = h - 4 - pHistory[i] * (h - 8);
    if (i === 0) sparkCtx.moveTo(x, y);
    else sparkCtx.lineTo(x, y);
  }
  sparkCtx.stroke();

  sparkCtx.strokeStyle = "rgba(100,116,139,0.5)";
  sparkCtx.setLineDash([4, 4]);
  const yHalf = h - 4 - 0.5 * (h - 8);
  sparkCtx.beginPath();
  sparkCtx.moveTo(4, yHalf);
  sparkCtx.lineTo(w - 4, yHalf);
  sparkCtx.stroke();
  sparkCtx.setLineDash([]);
}

function updateMetrics() {
  const p = sim.polarization();
  pHistory.push(p);
  if (pHistory.length > HISTORY_MAX) pHistory.shift();

  els.polarization.textContent = p.toFixed(3);
  els.time.textContent = (sim.stepCount * DT).toFixed(1);
  els.steps.textContent = String(sim.stepCount);
}

function renderFrame() {
  const size = resizeCanvas(simCanvas, simCtx);
  drawSimulation(size);
  drawSparkline();
}

function pause() {
  playing = false;
  els.btnToggle.textContent = "Start";
  if (animId !== null) {
    cancelAnimationFrame(animId);
    animId = null;
  }
}

function resetSimulation(seed = DEFAULT_SEED) {
  pause();
  const { N, R, v, eta } = readParams();
  if (sim.N !== N) {
    sim.resize(N);
  }
  sim.R = R;
  sim.v = v;
  sim.eta = eta;
  sim.reset(seed);
  pHistory = [sim.polarization()];
  pendingN = null;
  updateMetrics();
  renderFrame();
}

function togglePlay() {
  if (pendingN !== null) {
    resetSimulation(DEFAULT_SEED);
  }
  playing = !playing;
  els.btnToggle.textContent = playing ? "Pause" : "Start";
  if (playing) loop();
  else pause();
}

function loop() {
  if (!playing) return;
  applyLiveParams();
  for (let s = 0; s < STEPS_PER_FRAME; s += 1) {
    sim.step();
  }
  updateMetrics();
  renderFrame();
  animId = requestAnimationFrame(loop);
}

function onSliderInput() {
  syncLabels();
  const { N } = readParams();
  if (N !== sim.N) {
    pendingN = N;
    els.btnToggle.textContent = "Start (reset for N)";
    pause();
  } else {
    pendingN = null;
    applyLiveParams();
    if (!playing) {
      updateMetrics();
      renderFrame();
    }
  }
}

function applyPreset(eta) {
  els.eta.value = String(eta);
  syncLabels();
  resetSimulation(DEFAULT_SEED);
}

els.eta.addEventListener("input", onSliderInput);
els.R.addEventListener("input", onSliderInput);
els.v.addEventListener("input", onSliderInput);
els.N.addEventListener("input", onSliderInput);

els.btnToggle.addEventListener("click", togglePlay);
els.btnReset.addEventListener("click", () => resetSimulation(DEFAULT_SEED));
els.btnLow.addEventListener("click", () => applyPreset(0.1));
els.btnHigh.addEventListener("click", () => applyPreset(0.9));

window.addEventListener("resize", () => {
  if (document.getElementById("panel-simulate").classList.contains("active")) {
    renderFrame();
  }
});

function initTabs() {
  const tabButtons = document.querySelectorAll(".tab");
  const panels = {
    simulate: document.getElementById("panel-simulate"),
    about: document.getElementById("panel-about"),
  };
  let mathRendered = false;

  function renderAboutMath() {
    if (mathRendered || typeof renderMathInElement !== "function") return;
    renderMathInElement(panels.about, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
    });
    mathRendered = true;
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      tabButtons.forEach((b) => {
        const active = b.dataset.tab === target;
        b.classList.toggle("active", active);
        b.setAttribute("aria-selected", active ? "true" : "false");
      });
      Object.entries(panels).forEach(([key, panel]) => {
        panel.classList.toggle("active", key === target);
      });
      if (target === "about") renderAboutMath();
      if (target === "simulate") renderFrame();
    });
  });

}

syncLabels();
resetSimulation(DEFAULT_SEED);
initTabs();
