/**
 * 3D lattice Boltzmann solver (D3Q19, BGK collision + Smagorinsky sub-grid viscosity)
 * running entirely on the GPU with WebGPU compute.
 *
 * Lattice units: dx = dt = 1, sound speed c_s^2 = 1/3, density rho ~ 1.
 * Memory layout: structure of arrays, f[q * N + cell], cell = x + nx*(y + ny*z).
 * Streaming is "pull": each cell gathers its 19 incoming populations, applies boundary
 * rules for anything that would come from outside the domain or from a solid cell,
 * collides, and writes the post-collision populations to the other buffer (ping-pong).
 */

export const Face = { Periodic: 0, Inlet: 1, Outlet: 2, Slip: 3, Wall: 4 } as const;
export type FaceType = (typeof Face)[keyof typeof Face];

/** Order of faces: xmin, xmax, ymin, ymax, zmin, zmax. y is "up", x is streamwise. */
export type Faces = [FaceType, FaceType, FaceType, FaceType, FaceType, FaceType];

export interface LBMConfig {
  nx: number;
  ny: number;
  nz: number;
  /** BGK relaxation time before sub-grid model, tau0 = 3*nu + 1/2. */
  tau0: number;
  /** Smagorinsky constant, 0 turns the sub-grid model off. */
  cs?: number;
  faces: Faces;
  /** Inlet velocity (lattice units) and initial velocity. */
  uIn?: [number, number, number];
  /** Velocity of any face set to Wall (moving ground). */
  uWall?: [number, number, number];
  /** Constant body force per unit mass (Guo forcing). */
  gravity?: [number, number, number];
}

// ---- D3Q19 lattice ---------------------------------------------------------
export const C: [number, number, number][] = [
  [0, 0, 0],
  [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
];
export const OPP: number[] = C.map(([x, y, z]) =>
  C.findIndex(([a, b, c]) => a === -x && b === -y && c === -z),
);
export const W: number[] = C.map(([x, y, z]) => {
  const n = Math.abs(x) + Math.abs(y) + Math.abs(z);
  return n === 0 ? 1 / 3 : n === 1 ? 1 / 18 : 1 / 36;
});
const DIRIDX: number[] = new Array(27).fill(-1);
C.forEach(([x, y, z], q) => {
  DIRIDX[x + 1 + 3 * (y + 1 + 3 * (z + 1))] = q;
});

/** Equilibrium distribution, CPU side (used for initial conditions and tests). */
export function feqCPU(q: number, rho: number, ux: number, uy: number, uz: number): number {
  const [cx, cy, cz] = C[q];
  const cu = cx * ux + cy * uy + cz * uz;
  return W[q] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz));
}

const arr = (name: string, type: string, v: number[]) =>
  `var<private> ${name}: array<${type}, ${v.length}> = array<${type}, ${v.length}>(${v
    .map((n) => (type === 'f32' ? (Number.isInteger(n) ? n.toFixed(1) : String(n)) : String(n)))
    .join(', ')});`;


/** Unrolled pull for interior fluid cells with no solid neighbours: pure gathers, no checks. */
function fastPull(): string {
  return C.map(([cx, cy, cz], q) => {
    let src = 'idx';
    if (cx) src += cx > 0 ? ' - 1u' : ' + 1u';
    if (cy) src += cy > 0 ? ' - P.dims.x' : ' + P.dims.x';
    if (cz) src += cz > 0 ? ' - sz' : ' + sz';
    return `fp[${q}] = fin[${q}u * N + (${src})];`;
  }).join('\n        ');
}

const FORCE_SCALE = 1 << 20; // fixed-point scale for atomic force accumulation

const WGSL = /* wgsl */ `
struct Params {
  dims:   vec4<u32>,   // nx, ny, nz, N
  facesA: vec4<u32>,   // xmin, xmax, ymin, ymax
  facesB: vec4<u32>,   // zmin, zmax, unused, unused
  flow:   vec4<f32>,   // uIn.xyz, tau0
  wall:   vec4<f32>,   // uWall.xyz, Smagorinsky Cs
  force:  vec4<f32>,   // gravity.xyz, unused
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read_write> flags: array<u32>;  // 0 fluid, 1 solid, 2 fluid next to solid
@group(0) @binding(4) var<storage, read_write> mac: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> frc: array<atomic<i32>, 3>;

${arr('CX', 'i32', C.map((c) => c[0]))}
${arr('CY', 'i32', C.map((c) => c[1]))}
${arr('CZ', 'i32', C.map((c) => c[2]))}
${arr('OPP', 'i32', OPP)}
${arr('W', 'f32', W)}
${arr('DIRIDX', 'i32', DIRIDX)}

const FORCE_SCALE: f32 = ${FORCE_SCALE}.0;

fn feq(q: i32, rho: f32, u: vec3<f32>) -> f32 {
  let cu = f32(CX[q]) * u.x + f32(CY[q]) * u.y + f32(CZ[q]) * u.z;
  return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * dot(u, u));
}

fn faceType(axis: i32, hi: bool) -> u32 {
  let k = axis * 2 + select(0, 1, hi);
  if (k < 4) { return P.facesA[k]; }
  return P.facesB[k - 4];
}

fn cellIndex(p: vec3<i32>) -> u32 {
  return u32(p.x) + P.dims.x * (u32(p.y) + P.dims.y * u32(p.z));
}

@compute @workgroup_size(16, 4, 4)
fn init(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.dims.x || g.y >= P.dims.y || g.z >= P.dims.z) { return; }
  let idx = cellIndex(vec3<i32>(g));
  let N = P.dims.w;
  var u = P.flow.xyz;
  if (flags[idx] == 1u) { u = vec3<f32>(0.0); }
  for (var q = 0; q < 19; q++) {
    fout[u32(q) * N + idx] = feq(q, 1.0, u);
  }
  mac[idx] = vec4<f32>(u, 1.0);
}

// Tag fluid cells that touch a solid cell (18-neighbourhood) with 2 so only they pay for solid checks.
@compute @workgroup_size(16, 4, 4)
fn classify(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= P.dims.x || g.y >= P.dims.y || g.z >= P.dims.z) { return; }
  let idx = cellIndex(vec3<i32>(g));
  if (flags[idx] == 1u) { return; }
  let dims = vec3<i32>(i32(P.dims.x), i32(P.dims.y), i32(P.dims.z));
  let p = vec3<i32>(g);
  var near = false;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      for (var dx = -1; dx <= 1; dx++) {
        let n = p + vec3<i32>(dx, dy, dz);
        if (all(n >= vec3<i32>(0)) && all(n < dims)) {
          if (flags[cellIndex(n)] == 1u) { near = true; }
        }
      }
    }
  }
  flags[idx] = select(0u, 2u, near);
}

var<workgroup> wf: array<atomic<i32>, 3>;

@compute @workgroup_size(16, 4, 4)
fn step(@builtin(global_invocation_id) g: vec3<u32>,
        @builtin(local_invocation_index) li: u32) {
  if (li == 0u) {
    atomicStore(&wf[0], 0); atomicStore(&wf[1], 0); atomicStore(&wf[2], 0);
  }
  workgroupBarrier();

  let inside = g.x < P.dims.x && g.y < P.dims.y && g.z < P.dims.z;
  if (inside) {
    let N = P.dims.w;
    let dims = vec3<i32>(i32(P.dims.x), i32(P.dims.y), i32(P.dims.z));
    let p = vec3<i32>(g);
    let idx = cellIndex(p);
    let uIn = P.flow.xyz;
    let uw = P.wall.xyz;

    if (flags[idx] == 1u) {
      // solid cell: hold at rest so neighbours never read garbage
      for (var q = 0; q < 19; q++) { fout[u32(q) * N + idx] = feq(q, 1.0, vec3<f32>(0.0)); }
      mac[idx] = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    } else {
      var fp: array<f32, 19>;
      var fbody = vec3<f32>(0.0);
      let sz = P.dims.x * P.dims.y;
      let interior = p.x > 0 && p.y > 0 && p.z > 0
                  && p.x < dims.x - 1 && p.y < dims.y - 1 && p.z < dims.z - 1;
      if (interior && flags[idx] == 0u) {
        ${fastPull()}
      } else {
      for (var q = 0; q < 19; q++) {
        let c = vec3<i32>(CX[q], CY[q], CZ[q]);
        var s = p - c;
        var inlet = false;
        var wall = false;
        var refl = vec3<i32>(1, 1, 1);
        for (var a = 0; a < 3; a++) {
          if (s[a] < 0 || s[a] >= dims[a]) {
            let hi = s[a] >= dims[a];
            let t = faceType(a, hi);
            if (t == 0u) {                       // periodic: wrap
              s[a] = select(s[a] + dims[a], s[a] - dims[a], hi);
            } else if (t == 1u) {                // velocity inlet
              inlet = true;
            } else if (t == 2u) {                // outlet: zero-gradient
              s[a] = select(0, dims[a] - 1, hi);
            } else if (t == 3u) {                // free-slip: specular reflection
              refl[a] = -1;
              s[a] = p[a];
            } else {                             // no-slip / moving wall
              wall = true;
            }
          }
        }

        var val: f32;
        if (inlet) {
          val = feq(q, 1.0, uIn);
        } else if (wall) {
          // half-way bounce-back with wall velocity: f_q = f*_opp + 6 w_q (c_q . u_w)
          let qo = OPP[q];
          let cu = f32(c.x) * uw.x + f32(c.y) * uw.y + f32(c.z) * uw.z;
          val = fin[u32(qo) * N + idx] + 6.0 * W[q] * cu;
        } else {
          let sidx = cellIndex(s);
          if (flags[sidx] == 1u) {
            // half-way bounce-back off a solid body + momentum exchange
            let qo = OPP[q];
            let fs = fin[u32(qo) * N + idx];
            val = fs;
            fbody += 2.0 * fs * vec3<f32>(f32(-c.x), f32(-c.y), f32(-c.z));
          } else {
            let cq = c * refl;
            let qd = DIRIDX[(cq.x + 1) + 3 * ((cq.y + 1) + 3 * (cq.z + 1))];
            val = fin[u32(qd) * N + sidx];
          }
        }
        fp[q] = val;
      }
      }

      // ---- macroscopic moments
      var rho = 0.0;
      var m = vec3<f32>(0.0);
      for (var q = 0; q < 19; q++) {
        rho += fp[q];
        m += vec3<f32>(f32(CX[q]), f32(CY[q]), f32(CZ[q])) * fp[q];
      }
      let gvec = P.force.xyz;
      let u = (m + 0.5 * gvec) / rho;

      // ---- non-equilibrium stress from second moments (Pi_neq = sum c c f - rho u u - rho/3 I)
      var sxx = 0.0; var syy = 0.0; var szz = 0.0;
      var sxy = 0.0; var sxz = 0.0; var syz = 0.0;
      for (var q = 0; q < 19; q++) {
        let cx = f32(CX[q]); let cy = f32(CY[q]); let cz = f32(CZ[q]);
        let fq = fp[q];
        sxx += cx * cx * fq; syy += cy * cy * fq; szz += cz * cz * fq;
        sxy += cx * cy * fq; sxz += cx * cz * fq; syz += cy * cz * fq;
      }
      let pxx = sxx - rho * u.x * u.x - rho / 3.0;
      let pyy = syy - rho * u.y * u.y - rho / 3.0;
      let pzz = szz - rho * u.z * u.z - rho / 3.0;
      let pxy = sxy - rho * u.x * u.y;
      let pxz = sxz - rho * u.x * u.z;
      let pyz = syz - rho * u.y * u.z;
      let pn = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
      let tau0 = P.flow.w;
      let cs = P.wall.w;
      let tau = 0.5 * (tau0 + sqrt(tau0 * tau0 + 18.0 * 1.41421356 * cs * cs * pn / rho));
      let om = 1.0 / tau;

      // ---- BGK collision + Guo forcing, write post-collision populations
      for (var q = 0; q < 19; q++) {
        let fe = feq(q, rho, u);
        let c = vec3<f32>(f32(CX[q]), f32(CY[q]), f32(CZ[q]));
        let cu = dot(c, u);
        let cg = dot(c, gvec);
        let post = fp[q] - om * (fp[q] - fe)
                 + (1.0 - 0.5 * om) * W[q] * (3.0 * (cg - dot(u, gvec)) + 9.0 * cu * cg);
        fout[u32(q) * N + idx] = post;
      }
      mac[idx] = vec4<f32>(u, rho);

      if (fbody.x != 0.0 || fbody.y != 0.0 || fbody.z != 0.0) {
        atomicAdd(&wf[0], i32(round(fbody.x * FORCE_SCALE)));
        atomicAdd(&wf[1], i32(round(fbody.y * FORCE_SCALE)));
        atomicAdd(&wf[2], i32(round(fbody.z * FORCE_SCALE)));
      }
    }
  }

  workgroupBarrier();
  if (li == 0u) {
    let a = atomicLoad(&wf[0]); let b = atomicLoad(&wf[1]); let c = atomicLoad(&wf[2]);
    if (a != 0) { atomicAdd(&frc[0], a); }
    if (b != 0) { atomicAdd(&frc[1], b); }
    if (c != 0) { atomicAdd(&frc[2], c); }
  }
}
`;

export class LBM {
  readonly nx: number;
  readonly ny: number;
  readonly nz: number;
  readonly N: number;
  readonly cfg: Required<LBMConfig>;

  readonly fA: GPUBuffer;
  readonly fB: GPUBuffer;
  readonly flags: GPUBuffer;
  readonly mac: GPUBuffer;
  private forces: GPUBuffer;
  private forceStage: GPUBuffer;
  private params: GPUBuffer;
  private initPipe: GPUComputePipeline;
  private stepPipe: GPUComputePipeline;
  private classifyPipe: GPUComputePipeline;
  private bgClassify: GPUBindGroup;
  private bgAB: GPUBindGroup;
  private bgBA: GPUBindGroup;
  private bgInit: GPUBindGroup;
  /** 0: fA holds the current populations, 1: fB does. */
  private cur = 0;
  stepsSinceForceRead = 0;
  totalSteps = 0;

  constructor(private device: GPUDevice, cfg: LBMConfig) {
    this.cfg = {
      cs: 0,
      uIn: [0, 0, 0],
      uWall: [0, 0, 0],
      gravity: [0, 0, 0],
      ...cfg,
    } as Required<LBMConfig>;
    this.nx = cfg.nx;
    this.ny = cfg.ny;
    this.nz = cfg.nz;
    this.N = cfg.nx * cfg.ny * cfg.nz;

    const fBytes = 19 * this.N * 4;
    const mk = (size: number, usage: number, label: string) =>
      device.createBuffer({ size, usage, label });
    const S = GPUBufferUsage.STORAGE, CS = GPUBufferUsage.COPY_SRC, CD = GPUBufferUsage.COPY_DST;
    this.fA = mk(fBytes, S | CS | CD, 'lbm.fA');
    this.fB = mk(fBytes, S | CS | CD, 'lbm.fB');
    this.flags = mk(this.N * 4, S | CS | CD, 'lbm.flags');
    this.mac = mk(this.N * 16, S | CS | CD, 'lbm.mac');
    this.forces = mk(16, S | CS | CD, 'lbm.forces');
    this.forceStage = mk(16, GPUBufferUsage.MAP_READ | CD, 'lbm.forceStage');
    this.params = mk(96, GPUBufferUsage.UNIFORM | CD, 'lbm.params');

    const module = device.createShaderModule({ code: WGSL, label: 'lbm' });
    this.initPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'init' } });
    this.stepPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'step' } });

    this.classifyPipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'classify' } });
    const bg = (pipe: GPUComputePipeline, fi: GPUBuffer, fo: GPUBuffer) =>
      device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.params } },
          { binding: 1, resource: { buffer: fi } },
          { binding: 2, resource: { buffer: fo } },
          { binding: 3, resource: { buffer: this.flags } },
          { binding: 4, resource: { buffer: this.mac } },
          { binding: 5, resource: { buffer: this.forces } },
        ],
      });
    this.bgAB = bg(this.stepPipe, this.fA, this.fB);
    this.bgBA = bg(this.stepPipe, this.fB, this.fA);
    this.bgClassify = device.createBindGroup({
      layout: this.classifyPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 3, resource: { buffer: this.flags } },
      ],
    });
    // init only uses params, fout, flags, mac (auto layout drops unused bindings)
    this.bgInit = device.createBindGroup({
      layout: this.initPipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.params } },
        { binding: 2, resource: { buffer: this.fA } },
        { binding: 3, resource: { buffer: this.flags } },
        { binding: 4, resource: { buffer: this.mac } },
      ],
    });

    this.writeParams();
    this.reset();
  }

  private writeParams() {
    const c = this.cfg;
    const buf = new ArrayBuffer(96);
    const u = new Uint32Array(buf);
    const f = new Float32Array(buf);
    u.set([this.nx, this.ny, this.nz, this.N], 0);
    u.set([c.faces[0], c.faces[1], c.faces[2], c.faces[3]], 4);
    u.set([c.faces[4], c.faces[5], 0, 0], 8);
    f.set([c.uIn[0], c.uIn[1], c.uIn[2], c.tau0], 12);
    f.set([c.uWall[0], c.uWall[1], c.uWall[2], c.cs], 16);
    f.set([c.gravity[0], c.gravity[1], c.gravity[2], 0], 20);
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  /** Change run-time parameters without reallocating (wind speed, viscosity...). */
  update(patch: Partial<Pick<LBMConfig, 'tau0' | 'cs' | 'uIn' | 'uWall' | 'gravity'>>) {
    Object.assign(this.cfg, patch);
    this.writeParams();
  }

  private dispatchSize(): [number, number, number] {
    return [Math.ceil(this.nx / 16), Math.ceil(this.ny / 4), Math.ceil(this.nz / 4)];
  }

  /** Fill everything with uniform flow (uIn) at rho = 1. */
  reset() {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.initPipe);
    pass.setBindGroup(0, this.bgInit);
    pass.dispatchWorkgroups(...this.dispatchSize());
    pass.end();
    enc.clearBuffer(this.forces);
    this.device.queue.submit([enc.finish()]);
    this.cur = 0;
    this.stepsSinceForceRead = 0;
    this.totalSteps = 0;
  }

  /** Upload solid mask (1 = solid). */
  setSolid(mask: Uint32Array) {
    this.device.queue.writeBuffer(this.flags, 0, mask.buffer, mask.byteOffset, mask.byteLength);
    this.classify();
  }

  /** Recompute the "fluid next to solid" tags. Call after the mask changes on the GPU. */
  classify() {
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.classifyPipe);
    pass.setBindGroup(0, this.bgClassify);
    pass.dispatchWorkgroups(...this.dispatchSize());
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Upload a full population field (SoA, length 19*N) as the current state. */
  setPopulations(f: Float32Array) {
    this.device.queue.writeBuffer(this.cur === 0 ? this.fA : this.fB, 0, f.buffer, f.byteOffset, f.byteLength);
  }

  /** Run n lattice steps. */
  step(n: number, encoder?: GPUCommandEncoder) {
    const own = !encoder;
    const enc = encoder ?? this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.stepPipe);
    const size = this.dispatchSize();
    for (let i = 0; i < n; i++) {
      pass.setBindGroup(0, this.cur === 0 ? this.bgAB : this.bgBA);
      pass.dispatchWorkgroups(...size);
      this.cur ^= 1;
    }
    pass.end();
    if (own) this.device.queue.submit([enc.finish()]);
    this.stepsSinceForceRead += n;
    this.totalSteps += n;
  }

  /** Run n steps in modest GPU submissions so the browser's GPU watchdog is never tripped. */
  async run(n: number, chunk = 40) {
    for (let done = 0; done < n; done += chunk) {
      this.step(Math.min(chunk, n - done));
      await this.device.queue.onSubmittedWorkDone();
    }
  }

  /** Buffer currently holding the latest post-collision populations. */
  get current(): GPUBuffer {
    return this.cur === 0 ? this.fA : this.fB;
  }

  /**
   * Mean force on solid cells since the previous call, in lattice units
   * (momentum per time step). Resets the accumulator.
   */
  async readForces(): Promise<{ fx: number; fy: number; fz: number; steps: number }> {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.forces, 0, this.forceStage, 0, 16);
    enc.clearBuffer(this.forces);
    this.device.queue.submit([enc.finish()]);
    const steps = this.stepsSinceForceRead;
    this.stepsSinceForceRead = 0;
    await this.forceStage.mapAsync(GPUMapMode.READ);
    const v = new Int32Array(this.forceStage.getMappedRange().slice(0));
    this.forceStage.unmap();
    const k = steps > 0 ? 1 / (FORCE_SCALE * steps) : 0;
    return { fx: v[0] * k, fy: v[1] * k, fz: v[2] * k, steps };
  }

  /** Read back (u.xyz, rho) for every cell. Meant for tests, not for the render loop. */
  async readMacro(): Promise<Float32Array> {
    const size = this.N * 16;
    const stage = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.mac, 0, stage, 0, size);
    this.device.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    stage.destroy();
    return out;
  }

  async idle() {
    await this.device.queue.onSubmittedWorkDone();
  }

  destroy() {
    for (const b of [this.fA, this.fB, this.flags, this.mac, this.forces, this.forceStage, this.params]) b.destroy();
  }
}

export async function requestGPU(): Promise<GPUDevice> {
  if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No GPU adapter found');
  const want: GPUFeatureName[] = [];
  for (const f of ['timestamp-query', 'shader-f16'] as GPUFeatureName[]) if (adapter.features.has(f)) want.push(f);
  return adapter.requestDevice({
    requiredFeatures: want,
    requiredLimits: {
      maxBufferSize: Math.min(adapter.limits.maxBufferSize, 2 ** 31),
      maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 2 ** 31),
    },
  });
}
