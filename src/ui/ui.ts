import katex from 'katex';
import 'katex/dist/katex.min.css';
import './style.css';
import type { App, ModelId, Stats } from '../app/app';
import { PRESETS, type Preset } from '../app/presets';
import { runValidation } from '../validation/validate';
import { MATHS_HTML } from './maths';
import { ANALYTIC, SPHERE, SPHERE_STUDY, schillerNaumann } from './validation-data';

const $ = <T extends HTMLElement>(root: ParentNode, sel: string) => root.querySelector(sel) as T;

const sci = (x: number) => {
  if (!isFinite(x) || x <= 0) return '–';
  const e = Math.floor(Math.log10(x));
  const m = x / Math.pow(10, e);
  return `${m.toFixed(1)}×10<sup>${e}</sup>`;
};

function fillRange(el: HTMLInputElement) {
  const p = ((+el.value - +el.min) / (+el.max - +el.min)) * 100;
  el.style.setProperty('--p', p + '%');
}

export function buildUI(app: App) {
  const hud = document.createElement('div');
  hud.className = 'hud';
  hud.innerHTML = /* html */ `
  <div class="loading" id="loading"><div class="ring"></div><h2>AIRFLOW LAB</h2><p id="loadmsg">Building the wind tunnel…</p><div class="settle" style="width:220px"><div id="loadBar"></div></div></div>
  <div class="toast" id="toast"></div>

  <div class="panel brand">
    <h1><i></i>Airflow Lab</h1>
    <p>A live 3D Navier–Stokes wind tunnel. Lattice Boltzmann solver running on your GPU.</p>
    <div class="seg" id="modelSeg">
      <button data-model="a380" aria-pressed="false">Airbus A380</button>
      <button data-model="f1" aria-pressed="true">Formula 1</button>
    </div>
  </div>

  <div class="panel conditions">
    <div class="eyebrow" style="margin-bottom:12px">Conditions</div>
    <div class="field">
      <div class="field-head"><span>Wind speed</span><span class="val mono"><span id="speedVal">250</span><small>km/h</small></span></div>
      <input type="range" id="speed">
      <div class="sub" id="reLine"></div>
    </div>
    <div class="field" id="aoaField">
      <div class="field-head"><span>Angle of attack</span><span class="val mono"><span id="aoaVal">3</span><small>°</small></span></div>
      <input type="range" id="aoa" min="-4" max="16" step="0.5" value="3">
      <div class="sub">Nose-up angle of the whole aircraft. The body is re-voxelised live and the flow adapts.</div>
    </div>
    <div class="field" id="yawField">
      <div class="field-head"><span>Crosswind yaw</span><span class="val mono"><span id="yawVal">0</span><small>°</small></span></div>
      <input type="range" id="yaw" min="0" max="15" step="0.5" value="0">
      <div class="sub">Turns the car against the wind, like a gust from the side. Watch the side force and how the wake swings.</div>
    </div>
    <div class="divider"></div>
    <div class="field">
      <div class="field-head"><span>Camera</span></div>
      <div class="seg small" id="viewSeg">
        <button data-view="hero">Hero</button><button data-view="side">Side</button><button data-view="top">Top</button><button data-view="front">Front</button><button data-view="rear">Rear</button>
      </div>
      <div class="sub"><kbd>drag</kbd> orbit &nbsp;<kbd>scroll</kbd> zoom &nbsp;<kbd>shift</kbd>+<kbd>drag</kbd> pan</div>
    </div>
    <div class="row">
      <button class="btn grow" id="btnPause">Pause</button>
      <button class="btn grow" id="btnReset">Reset flow</button>
      <button class="btn grow" id="btnOrbit">Orbit</button>
    </div>
  </div>

  <div class="panel measure">
    <div class="eyebrow" id="mTitle">Measurements</div>
    <div class="grid">
      <div class="stat"><div class="k" id="k1">Drag CD</div><div class="v warm mono" id="v1">–</div><div class="u" id="u1"></div></div>
      <div class="stat"><div class="k" id="k2">Lift CL</div><div class="v cool mono" id="v2">–</div><div class="u" id="u2"></div></div>
      <div class="stat" style="grid-column:1 / -1"><div class="k" id="k3">Lift ÷ drag</div><div class="v mono" id="v3" style="font-size:22px">–</div></div>
    </div>
    <div class="settle"><div id="settleBar"></div></div>
    <canvas class="spark" id="spark" width="480" height="128"></canvas>
    <div class="legend"><span><i style="background:var(--warm)"></i>CD</span><span><i style="background:var(--accent)" id="legCl"></i><span id="legClName">CL</span></span><span style="margin-left:auto" id="refLine"></span></div>
    <div class="simline" id="simLine"></div>
    <div class="note">Coarse-grid large-eddy simulation: trust the <b>trends</b>, not the third decimal. <a id="lnkMaths">Why?</a></div>
  </div>

  <div class="dock-wrap">
    <div class="panel popover" id="popSlice">
      <div class="seg small" id="sliceAxis"><button data-axis="2" aria-pressed="true">Side (z)</button><button data-axis="1" aria-pressed="false">Top (y)</button><button data-axis="0" aria-pressed="false">Cross (x)</button></div>
      <div class="seg small" id="sliceMode"><button data-mode="0" aria-pressed="true">Speed</button><button data-mode="1" aria-pressed="false">Pressure</button></div>
      <label>Position <input type="range" id="slicePos" min="0.02" max="0.98" step="0.005" value="0.5"></label>
    </div>
    <div class="panel popover" id="popVortex">
      <label>Vortex threshold (Q) <input type="range" id="vortThr" min="0.5" max="14" step="0.1" value="4"></label>
      <span style="font-size:11px;color:var(--dim)">lower = more, fainter tubes</span>
    </div>
    <div class="panel dock">
      <button class="chip" data-layer="streams" aria-pressed="true" style="--c:#5cd6ff"><span class="dot"></span>Streamlines</button>
      <button class="chip" data-layer="pressure" aria-pressed="false" style="--c:#ff5b5b"><span class="dot"></span>Pressure</button>
      <button class="chip" data-layer="slice" aria-pressed="false" style="--c:#ffd35c"><span class="dot"></span>Slice</button>
      <button class="chip" data-layer="vortex" aria-pressed="false" style="--c:#8f7bff"><span class="dot"></span>Vortices</button>
      <span class="sep"></span>
      <button class="chip" id="btnMaths"><span class="dot" style="background:var(--good)"></span>Show the maths</button>
      <button class="chip" id="btnValid"><span class="dot" style="background:var(--warm)"></span>Validation</button>
      <button class="chip" id="btnInfo" title="Credits">i</button>
    </div>
  </div>

  <div class="panel drawer" id="drawer">
    <div class="drawer-head"><h2>The maths behind it</h2><button class="iconbtn" id="closeMaths" aria-label="Close">✕</button></div>
    <div class="drawer-body" id="mathsBody"></div>
  </div>

  <div class="modal-back" id="modalValid"><div class="panel modal" id="validBody"></div></div>
  <div class="modal-back" id="modalInfo"><div class="panel modal" style="max-width:640px">
    <h2>Credits and sources</h2>
    <p class="lead">Built for a Math Exhibition. The physics, the solver and the interface are original; the 3D models are free assets used under their licences.</p>
    <table>
      <tr><th>Item</th><th>Source</th><th>Licence</th></tr>
      <tr><td>Formula 1 car (Alfa Romeo C42)</td><td>github.com/markste-in/c42, based on the model published by Alfa Romeo</td><td>WTFPL</td></tr>
      <tr><td>Airbus A380</td><td>Flightradar24 fr24-3d-models (FlightGear aircraft library)</td><td>GPL-2.0, credit given</td></tr>
      <tr><td>Equation typesetting</td><td>KaTeX</td><td>MIT</td></tr>
      <tr><td>Graphics and compute</td><td>WebGPU (WGSL shaders written for this project)</td><td>n/a</td></tr>
    </table>
    <div class="row" style="margin-top:16px"><button class="btn accent" data-close="modalInfo">Close</button></div>
  </div></div>
  `;
  document.body.appendChild(hud);

  // ------------------------------------------------------------------ helpers
  const toast = (msg: string) => {
    const t = $(hud, '#toast');
    t.textContent = msg;
    t.classList.add('show');
    clearTimeout((toast as any)._t);
    (toast as any)._t = setTimeout(() => t.classList.remove('show'), 2200);
  };
  const setPressed = (el: Element, on: boolean) => el.setAttribute('aria-pressed', String(on));

  // ------------------------------------------------------------------ model switch + conditions
  const speed = $(hud, '#speed') as HTMLInputElement;
  const aoa = $(hud, '#aoa') as HTMLInputElement;
  const yaw = $(hud, '#yaw') as HTMLInputElement;
  let currentPreset: Preset = app.preset;

  function configureFor(p: Preset) {
    currentPreset = p;
    hud.querySelectorAll('#modelSeg button').forEach((b) => setPressed(b, (b as HTMLElement).dataset.model === p.id));
    speed.min = String(p.speed.min);
    speed.max = String(p.speed.max);
    speed.value = String(Math.round(app.state.speedKmh));
    fillRange(speed);
    $(hud, '#speedVal').textContent = speed.value;
    $(hud, '#aoaField').classList.toggle('hidden', p.id !== 'a380');
    $(hud, '#yawField').classList.toggle('hidden', p.id !== 'f1');
    $(hud, '#v3').parentElement!.classList.toggle('hidden', p.id === 'f1');
    aoa.value = String(app.state.aoa);
    yaw.value = String(app.state.yaw);
    fillRange(aoa);
    fillRange(yaw);
    $(hud, '#aoaVal').textContent = aoa.value;
    $(hud, '#yawVal').textContent = yaw.value;
    if (p.id === 'f1') {
      $(hud, '#k1').textContent = 'Drag CD';
      $(hud, '#k2').textContent = 'Side force CY';
      $(hud, '#legClName').textContent = 'CY';
    } else {
      $(hud, '#k1').textContent = 'Drag CD';
      $(hud, '#k2').textContent = 'Lift CL';
      $(hud, '#k3').textContent = 'Lift ÷ drag (L/D)';
      $(hud, '#legClName').textContent = 'CL';
    }
    updateReLine();
  }
  function updateReLine() {
    const re = app.sim?.re ?? 0;
    $(hud, '#reLine').innerHTML = `Simulated Re <b>${sci(re)}</b><br>Real ${currentPreset.id === 'f1' ? 'car' : 'aircraft'} Re <b>${sci(currentPreset.reReal(app.state.speedKmh))}</b>`;
  }

  hud.querySelectorAll('#modelSeg button').forEach((b) =>
    b.addEventListener('click', async () => {
      const id = (b as HTMLElement).dataset.model as ModelId;
      if (id === app.state.model) return;
      $(hud, '#loadmsg').textContent = `Building the ${PRESETS[id].label} wind tunnel…`;
      $(hud, '#loading').classList.remove('hide');
      await new Promise((r) => setTimeout(r, 350));
      await app.setModel(id);
    }),
  );
  app.onModelChange = (p) => configureFor(p);
  app.onProgress = (msg, f) => {
    $(hud, '#loadmsg').textContent = msg;
    ($(hud, '#loadBar') as HTMLElement).style.width = `${Math.round(f * 100)}%`;
  };

  speed.addEventListener('input', () => {
    fillRange(speed);
    $(hud, '#speedVal').textContent = speed.value;
    app.setSpeed(+speed.value);
    updateReLine();
  });
  aoa.addEventListener('input', () => {
    fillRange(aoa);
    $(hud, '#aoaVal').textContent = aoa.value;
    app.setAoa(+aoa.value);
  });
  yaw.addEventListener('input', () => {
    fillRange(yaw);
    $(hud, '#yawVal').textContent = yaw.value;
    app.setYaw(+yaw.value);
  });

  hud.querySelectorAll('#viewSeg button').forEach((b) =>
    b.addEventListener('click', () => app.setView((b as HTMLElement).dataset.view as any)),
  );
  const btnPause = $(hud, '#btnPause');
  const togglePause = () => {
    app.state.paused = !app.state.paused;
    btnPause.textContent = app.state.paused ? 'Resume' : 'Pause';
  };
  btnPause.addEventListener('click', togglePause);
  $(hud, '#btnReset').addEventListener('click', () => { app.resetFlow(); toast('Flow reset: watching it develop from still air'); });
  const btnOrbit = $(hud, '#btnOrbit');
  btnOrbit.addEventListener('click', () => {
    app.camera.autoOrbit = app.camera.autoOrbit ? 0 : 0.16;
    btnOrbit.classList.toggle('accent', !!app.camera.autoOrbit);
  });

  // ------------------------------------------------------------------ layers
  const popSlice = $(hud, '#popSlice');
  const popVortex = $(hud, '#popVortex');
  const layerBtn = (name: string) => hud.querySelector(`[data-layer="${name}"]`) as HTMLElement;
  function setLayer(name: 'streams' | 'pressure' | 'slice' | 'vortex', on: boolean) {
    app.state[name] = on;
    setPressed(layerBtn(name), on);
    popSlice.classList.toggle('open', app.state.slice);
    popVortex.classList.toggle('open', app.state.vortex);
  }
  hud.querySelectorAll('[data-layer]').forEach((b) =>
    b.addEventListener('click', () => {
      const n = (b as HTMLElement).dataset.layer as 'streams' | 'pressure' | 'slice' | 'vortex';
      setLayer(n, !app.state[n]);
    }),
  );
  hud.querySelectorAll('#sliceAxis button').forEach((b) =>
    b.addEventListener('click', () => {
      app.state.sliceAxis = +((b as HTMLElement).dataset.axis!) as 0 | 1 | 2;
      hud.querySelectorAll('#sliceAxis button').forEach((x) => setPressed(x, x === b));
    }),
  );
  hud.querySelectorAll('#sliceMode button').forEach((b) =>
    b.addEventListener('click', () => {
      app.state.sliceMode = +((b as HTMLElement).dataset.mode!) as 0 | 1;
      hud.querySelectorAll('#sliceMode button').forEach((x) => setPressed(x, x === b));
    }),
  );
  const slicePos = $(hud, '#slicePos') as HTMLInputElement;
  slicePos.addEventListener('input', () => { fillRange(slicePos); app.state.slicePos = +slicePos.value; });
  const vortThr = $(hud, '#vortThr') as HTMLInputElement;
  vortThr.addEventListener('input', () => { fillRange(vortThr); app.state.vortexThr = +vortThr.value; });
  fillRange(slicePos); fillRange(vortThr);

  // ------------------------------------------------------------------ maths drawer
  const drawer = $(hud, '#drawer');
  const mathsBody = $(hud, '#mathsBody');
  mathsBody.innerHTML = MATHS_HTML;
  mathsBody.querySelectorAll<HTMLElement>('[data-tex]').forEach((el) => {
    try {
      katex.render(el.dataset.tex!, el, { displayMode: el.classList.contains('eq'), throwOnError: false, strict: 'ignore' });
    } catch (e) {
      el.textContent = el.dataset.tex!;
    }
  });
  mathsBody.querySelectorAll<HTMLElement>('[data-show]').forEach((b) =>
    b.addEventListener('click', () => {
      const w = b.dataset.show!;
      if (w === 'pressure') setLayer('pressure', true);
      if (w === 'vortex') setLayer('vortex', true);
      if (w === 'streams') setLayer('streams', true);
      if (w === 'slicep') {
        app.state.sliceMode = 1;
        hud.querySelectorAll('#sliceMode button').forEach((x) => setPressed(x, (x as HTMLElement).dataset.mode === '1'));
        setLayer('slice', true);
      }
      toast('Layer switched on in the scene');
    }),
  );
  const setDrawer = (open: boolean) => {
    drawer.classList.toggle('open', open);
    hud.classList.toggle('drawer-open', open);
    setPressed($(hud, '#btnMaths'), open);
  };
  $(hud, '#btnMaths').addEventListener('click', () => setDrawer(!drawer.classList.contains('open')));
  $(hud, '#closeMaths').addEventListener('click', () => setDrawer(false));
  $(hud, '#lnkMaths').addEventListener('click', () => setDrawer(true));

  // ------------------------------------------------------------------ validation modal
  const modalValid = $(hud, '#modalValid');
  const validBody = $(hud, '#validBody');
  validBody.innerHTML = /* html */ `
    <h2>Does the solver get the right answer?</h2>
    <p class="lead">A simulation is only worth showing if it reproduces things we already know. These tests compare the solver against <b>exact mathematical solutions</b> and against a <b>published experimental correlation</b>. Numbers below are what it measured, including the imperfect ones.</p>
    <div class="eyebrow" style="margin:6px 0 8px">Exact solutions</div>
    <table>
      <tr><th>Test</th><th>Measured</th><th>Exact</th><th>Error</th></tr>
      ${ANALYTIC.map((a) => `<tr><td><b>${a.name}</b><br><span style="color:var(--dim)">${a.what}</span></td><td>${a.measured}</td><td>${a.reference}</td><td class="pass">${a.error}</td></tr>`).join('')}
    </table>
    <div class="row" style="margin:12px 0 4px;align-items:center;gap:12px">
      <button class="btn accent" id="btnRerun">Re-run these tests live on this GPU</button>
      <span id="rerunOut" style="font-size:12px;color:var(--dim)"></span>
    </div>
    <div class="eyebrow" style="margin:18px 0 8px">Flow past a sphere: drag coefficient against a published curve</div>
    <div class="grid2">
      <div class="chartbox"><canvas id="cdChart"></canvas><div class="cap">Line: Schiller–Naumann correlation from experiments, C<sub>D</sub> = 24/Re · (1 + 0.15 Re<sup>0.687</sup>). Orange dots: this solver (baseline). Yellow dots: same Re 50 case with wider or finer set-ups.</div></div>
      <div>
        <table>
          <tr><th>Re</th><th>Solver</th><th>Published</th><th>Diff</th></tr>
          ${SPHERE.map((s) => `<tr><td>${s.re}</td><td>${s.cd.toFixed(3)}</td><td>${s.ref.toFixed(3)}</td><td class="warn">+${(((s.cd - s.ref) / s.ref) * 100).toFixed(1)}%</td></tr>`).join('')}
        </table>
        <ul class="reslist" style="margin-top:12px;padding-left:18px">
          <li><b>Trend is right</b> across a factor of 20 in Re.</li>
          <li><b>Solver reads 8–14% high</b>, and the excess shrinks as Re rises. That pattern points at a test box that is too tight (inlet only 3 diameters upstream, walls 5D).</li>
          <li><b>Tested at Re 50:</b> open the box up (inlet 8D, walls 8D) and the error falls from <b>12.3% to 7.5%</b>. Refining the sphere (D 20→32) alone gives 9.7%. The remaining few percent is consistent with the staircase surface and the ±5% scatter of the correlation itself.</li>
        </ul>
      </div>
    </div>
    <div class="hint">Honest summary: exact tests pass to under 0.2%. Against experiments the drag is within about 10%. Good enough to trust the <b>trends</b> you see, not to design a car.</div>
    <div class="row" style="margin-top:14px"><button class="btn accent" data-close="modalValid">Close</button></div>
  `;
  function drawChart() {
    const c = $(validBody, '#cdChart') as HTMLCanvasElement;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = c.clientWidth, h = c.clientHeight;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext('2d')!;
    g.scale(dpr, dpr);
    const L = 52, R = 12, T = 12, B = 30;
    const xmin = Math.log10(5), xmax = Math.log10(400), ymin = Math.log10(0.5), ymax = Math.log10(8);
    const X = (re: number) => L + ((Math.log10(re) - xmin) / (xmax - xmin)) * (w - L - R);
    const Y = (cd: number) => T + (1 - (Math.log10(cd) - ymin) / (ymax - ymin)) * (h - T - B);
    g.strokeStyle = 'rgba(255,255,255,0.08)'; g.fillStyle = '#8d9dba'; g.font = '10px Inter, sans-serif'; g.lineWidth = 1;
    for (const re of [10, 20, 50, 100, 200]) { g.beginPath(); g.moveTo(X(re), T); g.lineTo(X(re), h - B); g.stroke(); g.fillText(String(re), X(re) - 8, h - B + 14); }
    for (const cd of [0.5, 1, 2, 4, 8]) { g.beginPath(); g.moveTo(L, Y(cd)); g.lineTo(w - R, Y(cd)); g.stroke(); g.fillText(String(cd), 24, Y(cd) + 3); }
    g.fillText('Reynolds number', w / 2 - 34, h - 4);
    g.save(); g.translate(10, h / 2 + 50); g.rotate(-Math.PI / 2); g.fillText('Drag coefficient CD', 0, 0); g.restore();
    g.strokeStyle = '#5cd6ff'; g.lineWidth = 2; g.beginPath();
    for (let i = 0; i <= 80; i++) { const re = 5 * Math.pow(400 / 5, i / 80); const x = X(re), y = Y(schillerNaumann(re)); i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
    g.fillStyle = '#ff9a3c';
    for (const s of SPHERE) { g.beginPath(); g.arc(X(s.re), Y(s.cd), 4.5, 0, 7); g.fill(); }
    g.fillStyle = '#ffd35c';
    for (const st of SPHERE_STUDY.slice(1)) { g.beginPath(); g.arc(X(50), Y(st.cd), 3.5, 0, 7); g.fill(); }
  }
  const openModal = (id: string) => {
    $(hud, '#' + id).classList.add('open');
    if (id === 'modalValid') requestAnimationFrame(drawChart);
  };
  hud.querySelectorAll<HTMLElement>('[data-close]').forEach((b) => b.addEventListener('click', () => $(hud, '#' + b.dataset.close).classList.remove('open')));
  [modalValid, $(hud, '#modalInfo')].forEach((m) => m.addEventListener('pointerdown', (e) => { if (e.target === m) m.classList.remove('open'); }));
  $(hud, '#btnValid').addEventListener('click', () => openModal('modalValid'));
  $(hud, '#btnInfo').addEventListener('click', () => openModal('modalInfo'));
  let rerunning = false;
  $(validBody, '#btnRerun').addEventListener('click', async () => {
    if (rerunning) return;
    rerunning = true;
    const out = $(validBody, '#rerunOut');
    out.textContent = 'running on a fresh GPU context…';
    try {
      const res = await runValidation('taylor,poiseuille1,poiseuille2');
      out.innerHTML = res.map((r) => `<span class="${r.pass ? 'pass' : 'warn'}">${r.pass ? '✓' : '✗'}</span> ${r.summary}`).join('<br>');
    } catch (e) {
      out.textContent = 'could not run: ' + (e as Error).message;
    }
    rerunning = false;
  });

  // ------------------------------------------------------------------ keyboard
  window.addEventListener('keydown', (e) => {
    if ((e.target as HTMLElement)?.tagName === 'INPUT') return;
    if (e.code === 'Space') { e.preventDefault(); togglePause(); }
    else if (e.key === '1') setLayer('streams', !app.state.streams);
    else if (e.key === '2') setLayer('pressure', !app.state.pressure);
    else if (e.key === '3') setLayer('slice', !app.state.slice);
    else if (e.key === '4') setLayer('vortex', !app.state.vortex);
    else if (e.key.toLowerCase() === 'm') setDrawer(!drawer.classList.contains('open'));
    else if (e.key.toLowerCase() === 'v') openModal('modalValid');
    else if (e.key === 'Escape') { setDrawer(false); hud.querySelectorAll('.modal-back').forEach((m) => m.classList.remove('open')); }
  });

  // ------------------------------------------------------------------ live stats
  const spark = $(hud, '#spark') as HTMLCanvasElement;
  function drawSpark(s: Stats) {
    const g = spark.getContext('2d')!;
    const w = spark.width, h = spark.height;
    g.clearRect(0, 0, w, h);
    const isF1 = currentPreset.id === 'f1';
    const series = [
      { v: s.history.map((p) => p.cd), c: '#ff9a3c' },
      { v: s.history.map((p) => (isF1 ? p.cy : p.cl)), c: '#5cd6ff' },
    ];
    g.strokeStyle = 'rgba(255,255,255,0.06)';
    g.lineWidth = 1;
    for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, (h * i) / 4); g.lineTo(w, (h * i) / 4); g.stroke(); }
    for (const sr of series) {
      if (sr.v.length < 2) continue;
      const lo = Math.min(...sr.v), hi = Math.max(...sr.v);
      const pad = Math.max((hi - lo) * 0.15, 0.02 * Math.max(Math.abs(hi), Math.abs(lo), 0.05));
      const y = (v: number) => h - 8 - ((v - (lo - pad)) / (hi - lo + 2 * pad)) * (h - 16);
      g.strokeStyle = sr.c; g.lineWidth = 2.2; g.lineJoin = 'round'; g.beginPath();
      sr.v.forEach((v, i) => { const x = (i / 239) * w; i ? g.lineTo(x, y(v)) : g.moveTo(x, y(v)); });
      g.stroke();
    }
  }
  const live = (k: string, html: string) => mathsBody.querySelectorAll(`[data-live="${k}"]`).forEach((el) => (el.innerHTML = html));
  let firstReady = false;
  app.onStats = (s) => {
    if (s.ready && !firstReady) { firstReady = true; }
    if (s.ready) $(hud, '#loading').classList.add('hide');
    const isF1 = currentPreset.id === 'f1';
    const has = s.history.length > 0;
    const cd = s.cd, second = isF1 ? s.cy : s.cl;
    const signed = (v: number) => (v < -0.005 ? '−' : v > 0.005 && isF1 ? '+' : '') + Math.abs(v).toFixed(2);
    $(hud, '#v1').textContent = has ? cd.toFixed(2) : '…';
    $(hud, '#v2').textContent = has ? signed(second) : '…';
    $(hud, '#v3').textContent = has && cd > 1e-3 ? (s.cl / cd).toFixed(2) : '…';
    $(hud, '#u1').textContent = isF1 ? 'on frontal area' : 'on wing area';
    $(hud, '#u2').textContent = isF1 ? 'sideways push' : 'positive = up';
    ($(hud, '#settleBar') as HTMLElement).style.width = `${Math.round(s.settled * 100)}%`;
    ($(hud, '.settle') as HTMLElement).style.opacity = s.settled >= 1 ? '0.35' : '1';
    $(hud, '#refLine').textContent = `A = ${s.refArea.toFixed(currentPreset.id === 'f1' ? 2 : 0)} m²`;
    $(hud, '#simLine').innerHTML = `<b>${(s.cells / 1e6).toFixed(2)} M</b> cells of ${(currentPreset.dx * 100).toFixed(0)} cm · <b>${Math.round(s.stepsPerSec)}</b> steps/s · ${Math.round(s.fps)} fps<br>τ ${s.tau.toFixed(4)} · Mach ${s.mach.toFixed(2)} · ${s.solidCells.toLocaleString()} solid cells`;
    updateReLine();
    drawSpark(s);
    if (drawer.classList.contains('open')) {
      const nu = (s.tau - 0.5) / 3;
      live('tau', s.tau.toFixed(4));
      live('nu', nu.toFixed(5));
      live('re', sci(s.re));
      live('reReal', sci(s.reReal));
      live('mach', s.mach.toFixed(2));
      live('dx', `${(currentPreset.dx * 100).toFixed(0)} cm`);
      const d = currentPreset.dims;
      live('grid', `${d[0]}×${d[1]}×${d[2]}`);
    }
  };

  configureFor(app.preset);
  // reflect any state that came from the URL
  (['streams', 'pressure', 'slice', 'vortex'] as const).forEach((n) => setLayer(n, app.state[n]));
  hud.querySelectorAll('#sliceAxis button').forEach((x) => setPressed(x, +(x as HTMLElement).dataset.axis! === app.state.sliceAxis));
  hud.querySelectorAll('#sliceMode button').forEach((x) => setPressed(x, +(x as HTMLElement).dataset.mode! === app.state.sliceMode));
  slicePos.value = String(app.state.slicePos); fillRange(slicePos);
  vortThr.value = String(app.state.vortexThr); fillRange(vortThr);
  (window as any).__ui = { toast };
}
