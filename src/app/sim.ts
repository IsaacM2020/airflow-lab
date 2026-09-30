import { LBM } from '../solver/lbm';
import { Viz } from '../solver/viz';
import { Voxeliser, bodyToCell, type VoxelResult } from '../solver/voxel';
import { Streamlines } from '../render/layers';
import type { Model } from '../render/model';
import type { Scene } from '../render/scene';
import type { Preset } from './presets';

/** Lattice inflow speed. Low enough to keep the Mach number small (Ma = sqrt(3) * u). */
export const U_LAT = 0.07;
const SMAGORINSKY = 0.14;
/** Force readings are meaningless while the start-up shock is still ringing (about one flow-through of the body). */
export const SETTLE_STEPS = 1800;

export interface ForceSample { t: number; cd: number; cl: number; cy: number; fx: number; fy: number; fz: number }

/** One running wind tunnel: solver, voxeliser, visualisation volumes and smoke for a single model. */
export class Sim {
  readonly lbm: LBM;
  readonly viz: Viz;
  readonly streams: Streamlines;
  private vox: Voxeliser;
  readonly lengthCells: number;
  aoaDeg = 0;
  rideCm = 0;
  yawDeg = 0;
  speedKmh: number;
  re = 0;
  tau0 = 0.51;
  voxel: VoxelResult | null = null;
  modelMatrix: number[] = [];
  history: ForceSample[] = [];
  smoothCd = 0;
  smoothCl = 0;
  smoothCy = 0;
  running = true;
  private simTime = 0;
  /** Step count at which the current settling window started, and how many steps it needs. */
  private settleStart = 0;
  private settleNeed = SETTLE_STEPS;
  private busy = false;
  /** Free-stream density, measured upstream, used as the pressure reference. */
  rhoRef = 1;
  private refStage: GPUBuffer;
  private refBusy = false;
  private pendingGeom = false;

  constructor(private device: GPUDevice, private scene: Scene, readonly preset: Preset, readonly model: Model) {
    const [nx, ny, nz] = preset.dims;
    this.lengthCells = preset.length / preset.dx;
    this.speedKmh = preset.speed.def;
    this.lbm = new LBM(device, {
      nx, ny, nz, tau0: 0.52, cs: SMAGORINSKY, faces: preset.faces,
      uIn: [U_LAT, 0, 0], uWall: [U_LAT, 0, 0],
    });
    this.viz = new Viz(device, this.lbm);
    this.refStage = device.createBuffer({ size: 16 * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.vox = new Voxeliser(device, { positions: model.positions, indices: model.indices });
    this.streams = new Streamlines(device, scene, preset.smoke, nx);
    this.streams.seed = preset.seedBox;
    this.applySpeed(this.speedKmh);
    scene.setViz(this.viz.t0, this.viz.t1);
    this.updateModelMatrix();
    this.streams.reset(nx, preset.dims);
  }

  /** Map the wind-speed slider (km/h) onto the simulation Reynolds number and the relaxation time. */
  applySpeed(kmh: number) {
    const s = this.preset.speed;
    this.speedKmh = kmh;
    const f = Math.min(1, Math.max(0, (kmh - s.min) / (s.max - s.min)));
    const { min, max } = this.preset.reSim;
    this.re = Math.exp(Math.log(min) + f * (Math.log(max) - Math.log(min)));
    const nu = (U_LAT * this.lengthCells) / this.re;
    this.tau0 = 0.5 + 3 * nu;
    this.lbm.update({ tau0: this.tau0 });
    // visual advection speed follows the wind speed, so faster air really looks faster
    this.streams.dt = 3.5 + 6.5 * f;
  }

  private updateModelMatrix() {
    const p = this.preset;
    const place: [number, number, number] = [p.place[0], p.place[1] + (p.ground ? this.rideCm / 100 / p.dx : 0), p.place[2]];
    this.modelMatrix = bodyToCell({ pitch: -(this.aoaDeg * Math.PI) / 180, yaw: (this.yawDeg * Math.PI) / 180, pivot: p.pivot, dx: p.dx, place });
  }

  /** Re-voxelise after the body moved or rotated. The flow keeps running and adapts. */
  async applyGeometry() {
    if (this.busy) { this.pendingGeom = true; return; }
    this.busy = true;
    do {
      this.pendingGeom = false;
      this.updateModelMatrix();
      const first = !this.voxel;
      this.voxel = await this.vox.voxelise(this.preset.dims, this.modelMatrix, this.lbm.flags);
      this.lbm.classify();
      if (!first) {
        // the body moved: the old force averages describe a different flow, so start counting again
        this.settleStart = this.lbm.totalSteps;
        this.settleNeed = 900;
        this.history = [];
        this.smoothCd = this.smoothCl = this.smoothCy = 0;
      }
    } while (this.pendingGeom);
    this.busy = false;
  }

  /** Advance the physics by n lattice steps inside an existing command encoder. */
  step(enc: GPUCommandEncoder, n: number) {
    if (!this.running || !this.voxel) return;
    this.lbm.step(n, enc);
    this.simTime += n;
  }

  /**
   * Pressure is only meaningful relative to the free stream. The solver's mean density drifts by about 1%, which
   * would show up as a huge fake Cp (Cp = dp / (1/2 U^2), and the dynamic pressure is tiny in lattice units), so
   * the reference density is measured from 16 cells in the far upstream corner of the tunnel.
   */
  async sampleRhoRef() {
    if (this.refBusy || this.destroyed || !this.voxel) return;
    this.refBusy = true;
    try {
      const [nx, ny] = this.preset.dims;
      const cell = 0 + nx * (ny - 3 + ny * 3);
      const enc = this.device.createCommandEncoder();
      enc.copyBufferToBuffer(this.lbm.mac, cell * 16, this.refStage, 0, 16 * 16);
      this.device.queue.submit([enc.finish()]);
      await this.refStage.mapAsync(GPUMapMode.READ);
      const d = new Float32Array(this.refStage.getMappedRange().slice(0));
      this.refStage.unmap();
      let sum = 0;
      for (let i = 0; i < 16; i++) sum += d[4 * i + 3];
      const avg = sum / 16;
      if (avg > 0.5 && avg < 1.5) this.rhoRef += (avg - this.rhoRef) * 0.4;
    } catch {
      /* buffer destroyed while switching models */
    }
    this.refBusy = false;
  }

  /** 0..1: how far the flow has developed past the start-up transient. */
  get settled(): number {
    return Math.min(1, Math.max(0, this.lbm.totalSteps - this.settleStart) / this.settleNeed);
  }

  /** Read forces asynchronously and update Cd / Cl. */
  async sampleForces() {
    if (this.destroyed || !this.voxel || this.lbm.stepsSinceForceRead < 20) return;
    const f = await this.lbm.readForces();
    if (this.destroyed) return;
    if (f.steps < 5 || this.settled < 1) return; // discard the transient while the flow re-settles
    const q = 0.5 * U_LAT * U_LAT;
    const p = this.preset;
    const ref = p.refAreaM2 ? p.refAreaM2 / (p.dx * p.dx) : this.voxel.frontalArea;
    const cd = f.fx / (q * ref);
    const cl = f.fy / (q * ref);
    const cy = f.fz / (q * ref);
    const k = this.history.length ? 0.15 : 1;
    this.smoothCd += (cd - this.smoothCd) * k;
    this.smoothCl += (cl - this.smoothCl) * k;
    this.smoothCy += (cy - this.smoothCy) * k;
    this.history.push({ t: this.simTime, cd: this.smoothCd, cl: this.smoothCl, cy: this.smoothCy, fx: f.fx, fy: f.fy, fz: f.fz });
    if (this.history.length > 240) this.history.shift();
  }

  resetFlow() {
    this.lbm.reset();
    this.history = [];
    this.smoothCd = 0;
    this.smoothCl = 0;
    this.smoothCy = 0;
    this.simTime = 0;
    this.settleStart = 0;
    this.settleNeed = SETTLE_STEPS;
    this.streams.reset(this.preset.dims[0], this.preset.dims);
  }

  private destroyed = false;
  destroy() {
    this.destroyed = true;
    try { this.refStage.destroy(); } catch { /* already gone */ }
    this.lbm.destroy();
    this.viz.destroy();
    this.streams.destroy();
  }
}
