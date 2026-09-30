import { OrbitCamera } from '../render/camera';
import { SliceLayer, VortexLayer } from '../render/layers';
import { loadModel, type Model } from '../render/model';
import { Post } from '../render/post';
import { Scene } from '../render/scene';
import { requestGPU } from '../solver/lbm';
import { PRESETS, type Preset } from './presets';
import { Sim, U_LAT } from './sim';

export type ModelId = 'f1' | 'a380';

/** Lattice steps run behind the loading screen before the first frame. */
const SPIN_UP_STEPS = 1100;

export interface AppState {
  model: ModelId;
  speedKmh: number;
  aoa: number; // degrees, A380
  rideCm: number; // centimetres, F1 (fixed at its default, not user-adjustable)
  yaw: number; // degrees of crosswind, F1
  streams: boolean;
  pressure: boolean;
  slice: boolean;
  vortex: boolean;
  sliceAxis: 0 | 1 | 2;
  slicePos: number;
  sliceMode: 0 | 1; // speed / pressure
  vortexThr: number;
  paused: boolean;
  exposure: number;
  streamOpacity: number;
}

export interface Stats {
  cd: number;
  cl: number;
  re: number;
  reReal: number;
  stepsPerSec: number;
  fps: number;
  cells: number;
  solidCells: number;
  ready: boolean;
  history: { cd: number; cl: number; cy: number }[];
  cy: number;
  tau: number;
  mach: number;
  settled: number;
  refArea: number;
}

const VIEWS: Record<string, { yaw: number; pitch: number; dist: number }> = {
  side: { yaw: 0, pitch: 0.06, dist: 1 },
  top: { yaw: 0, pitch: 1.45, dist: 1.1 },
  front: { yaw: -Math.PI / 2, pitch: 0.12, dist: 1 },
  rear: { yaw: Math.PI / 2, pitch: 0.16, dist: 1 },
  hero: { yaw: -0.75, pitch: 0.28, dist: 1 },
};

export class App {
  readonly canvas: HTMLCanvasElement;
  device!: GPUDevice;
  private ctx!: GPUCanvasContext;
  scene!: Scene;
  private post!: Post;
  camera!: OrbitCamera;
  private slice!: SliceLayer;
  private vortex!: VortexLayer;
  sim: Sim | null = null;
  private models = new Map<ModelId, Model>();
  state: AppState;
  stats: Stats = {
    cd: 0, cl: 0, re: 0, reReal: 0, stepsPerSec: 0, fps: 0, cells: 0, solidCells: 0, ready: false, history: [], cy: 0, tau: 0, mach: 0, settled: 0, refArea: 0,
  };
  onStats: (s: Stats) => void = () => {};
  onModelChange: (p: Preset) => void = () => {};
  onProgress: (msg: string, frac: number) => void = () => {};
  private t0 = performance.now();
  private last = performance.now();
  private frames = 0;
  private stepsPerFrame = 3;
  private frameAvg = 16;
  private stepAcc = 0;
  private statT = performance.now();
  private forceBusy = false;
  private switching = false;

  constructor(canvas: HTMLCanvasElement, initial: Partial<AppState> = {}) {
    this.canvas = canvas;
    const p = PRESETS[initial.model ?? 'f1'];
    this.state = {
      model: 'f1', speedKmh: p.speed.def, aoa: 3, rideCm: 10, yaw: 0,
      streams: true, pressure: false, slice: false, vortex: false,
      sliceAxis: 2, slicePos: 0.5, sliceMode: 0, vortexThr: 4,
      paused: false, exposure: 1.0, streamOpacity: p.smokeOpacity,
      ...initial,
    };
  }

  async init() {
    this.device = await requestGPU();
    this.ctx = this.canvas.getContext('webgpu')!;
    const format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device: this.device, format, alphaMode: 'opaque' });
    this.scene = new Scene(this.device);
    this.post = new Post(this.device, this.scene.frame.buffer, format);
    this.camera = new OrbitCamera(this.canvas);
    this.slice = new SliceLayer(this.device, this.scene);
    this.vortex = new VortexLayer(this.device, this.scene);
    await this.setModel(this.state.model, true);
    requestAnimationFrame((t) => this.frame(t));
  }

  get preset(): Preset { return PRESETS[this.state.model]; }

  async setModel(id: ModelId, first = false) {
    if (this.switching) return;
    this.switching = true;
    this.stats.ready = false;
    const preset = PRESETS[id];
    let model = this.models.get(id);
    if (!model) {
      model = await loadModel(this.device, preset.model, this.scene.matLayout);
      this.models.set(id, model);
    }
    this.sim?.destroy();
    this.state.model = id;
    if (!first) this.state.streamOpacity = preset.smokeOpacity;
    if (!first || this.state.speedKmh < preset.speed.min || this.state.speedKmh > preset.speed.max) this.state.speedKmh = preset.speed.def;
    this.sim = new Sim(this.device, this.scene, preset, model);
    this.sim.aoaDeg = id === 'a380' ? this.state.aoa : 0; // angle of attack only applies to the aircraft
    this.sim.rideCm = this.state.rideCm;
    this.sim.yawDeg = id === 'f1' ? this.state.yaw : 0;
    this.sim.applySpeed(this.state.speedKmh);
    this.camera.set({ ...preset.camera }, first);
    this.onModelChange(preset);
    this.onProgress('Voxelising the model on the GPU…', 0);
    await this.sim.applyGeometry();
    // spin the flow up behind the loading screen so the first thing you see is already developed
    const total = SPIN_UP_STEPS;
    for (let done = 0; done < total; done += 40) {
      await this.sim.lbm.run(Math.min(40, total - done), 40);
      this.onProgress('Spinning up the airflow…', done / total);
    }
    await this.sim.lbm.readForces().catch(() => {});
    this.sim.history = [];
    this.stats.ready = true;
    this.switching = false;
  }

  setSpeed(kmh: number) { this.state.speedKmh = kmh; this.sim?.applySpeed(kmh); }
  setAoa(deg: number) { this.state.aoa = deg; if (this.sim && this.state.model === 'a380') { this.sim.aoaDeg = deg; void this.sim.applyGeometry(); } }
  setYaw(deg: number) { this.state.yaw = deg; if (this.sim && this.state.model === 'f1') { this.sim.yawDeg = deg; void this.sim.applyGeometry(); } }
  setRide(cm: number) { this.state.rideCm = cm; if (this.sim && this.state.model === 'f1') { this.sim.rideCm = cm; void this.sim.applyGeometry(); } }
  resetFlow() { this.sim?.resetFlow(); }

  setView(name: keyof typeof VIEWS) {
    const v = VIEWS[name];
    const p = this.preset;
    this.camera.set({ target: [...p.camera.target], distance: p.camera.distance * v.dist, yaw: v.yaw, pitch: v.pitch });
  }

  private resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 1.6);
    const w = Math.max(2, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(2, Math.floor(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    this.scene.resize(w, h);
    this.post.resize(w, h);
  }

  private frame(now: number) {
    requestAnimationFrame((t) => this.frame(t));
    const sim = this.sim;
    if (!sim) return;
    this.resize();
    const rawDt = now - this.last;
    this.last = now;
    this.frameAvg += (Math.min(rawDt, 100) - this.frameAvg) * 0.05;
    const dt = Math.min(0.05, rawDt / 1000);
    this.camera.update(dt);

    // adaptive number of physics steps per rendered frame
    if (++this.frames % 20 === 0) {
      if (this.frameAvg > 30 && this.stepsPerFrame > 1) this.stepsPerFrame--;
      else if (this.frameAvg < 22 && this.stepsPerFrame < 10) this.stepsPerFrame++;
    }

    const m = this.camera.matrices(this.canvas.width / this.canvas.height);
    const f = this.scene.frame;
    const st = this.state;
    const p = sim.preset;
    const [nx, ny, nz] = p.dims;
    f.setMatrices(m.viewProj, m.invViewProj);
    f.setEye(m.eye[0], m.eye[1], m.eye[2], (now - this.t0) / 1000);
    f.setScreen(this.canvas.width, this.canvas.height);
    f.setDomain(nx, ny, nz, U_LAT);
    f.setLayers(st.pressure ? 0.92 : 0, st.slice ? 0.78 : 0, st.vortex ? 1 : 0, st.streams ? st.streamOpacity : 0);
    f.setMisc(p.cpScale, st.slicePos, st.vortexThr, st.exposure);
    f.setMisc2(st.sliceMode, st.sliceAxis, p.ground ? 1 : 0, 1.35);
    f.setModel(sim.modelMatrix);
    f.upload();

    const enc = this.device.createCommandEncoder();
    const running = !st.paused && this.stats.ready;
    if (running) {
      sim.step(enc, this.stepsPerFrame);
      this.stepAcc += this.stepsPerFrame;
    }
    sim.viz.update(enc, U_LAT, (U_LAT / 10) ** 2, U_LAT / 10, sim.rhoRef);
    if (running && st.streams) sim.streams.advance(enc);
    this.scene.draw(enc, sim.model, p.ground, (pass) => {
      if (st.slice) this.slice.draw(pass);
      if (st.streams) sim.streams.draw(pass);
    });
    if (st.vortex) this.vortex.draw(enc);
    this.post.run(enc, this.scene.hdr, this.ctx.getCurrentTexture().createView());
    this.device.queue.submit([enc.finish()]);

    if (this.frames % 8 === 0) void sim.sampleRhoRef();
    if (running && this.frames % 12 === 0 && !this.forceBusy) {
      this.forceBusy = true;
      sim.sampleForces().catch(() => {}).finally(() => (this.forceBusy = false));
    }
    const nowMs = performance.now();
    if (nowMs - this.statT > 250) {
      const secs = (nowMs - this.statT) / 1000;
      this.statT = nowMs;
      const s = this.stats;
      s.stepsPerSec = this.stepAcc / secs;
      this.stepAcc = 0;
      s.fps = 1000 / this.frameAvg;
      s.cd = sim.smoothCd;
      s.cl = sim.smoothCl;
      s.re = sim.re;
      s.reReal = p.reReal(this.state.speedKmh);
      s.cells = sim.lbm.N;
      s.solidCells = sim.voxel?.solidCells ?? 0;
      s.history = sim.history.map((h) => ({ cd: h.cd, cl: h.cl, cy: h.cy }));
      s.cy = sim.smoothCy;
      s.tau = sim.tau0;
      s.mach = Math.sqrt(3) * U_LAT;
      s.settled = sim.settled;
      s.refArea = p.refAreaM2 ?? (sim.voxel?.frontalArea ?? 0) * p.dx * p.dx;
      this.onStats(s);
    }
  }
}
