import type { LBM } from './lbm';

/**
 * Packs the solver's per-cell (u, rho) into two 3D textures the renderer samples with hardware
 * trilinear filtering:
 *   t0 = (ux, uy, uz, Cp)          Cp = (p - p_inf) / (1/2 rho U^2), p = rho/3 in lattice units, p_inf measured from the upstream air
 *   t1 = (Q, |u|/U, |vorticity|, solid)
 * Q is the Q-criterion, Q = 1/2 (|Omega|^2 - |S|^2): positive where rotation beats strain, i.e. inside a vortex core.
 */
const WGSL = /* wgsl */ `
struct VP {
  dims: vec4<u32>,   // nx, ny, nz, N
  p: vec4<f32>,      // uLat, qNorm, vortNorm, rhoRef (free-stream density)
}
@group(0) @binding(0) var<uniform> V: VP;
@group(0) @binding(1) var<storage, read> mac: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> flags: array<u32>;
@group(0) @binding(3) var t0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(4) var t1: texture_storage_3d<rgba16float, write>;

fn idxOf(p: vec3<i32>) -> u32 {
  let d = vec3<i32>(i32(V.dims.x), i32(V.dims.y), i32(V.dims.z));
  let c = clamp(p, vec3<i32>(0), d - 1);
  return u32(c.x) + V.dims.x * (u32(c.y) + V.dims.y * u32(c.z));
}
fn vel(p: vec3<i32>) -> vec3<f32> {
  let i = idxOf(p);
  if (flags[i] == 1u) { return vec3<f32>(0.0); }
  return mac[i].xyz;
}

@compute @workgroup_size(8, 8, 4)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= V.dims.x || g.y >= V.dims.y || g.z >= V.dims.z) { return; }
  let p = vec3<i32>(g);
  let i = idxOf(p);
  let m = mac[i];
  let solid = flags[i] == 1u;
  let U = V.p.x;
  var cp = 0.0;
  var q = 0.0;
  var vort = 0.0;
  var sp = 0.0;
  var u = vec3<f32>(0.0);
  if (!solid) {
    u = m.xyz;
    cp = ((m.w - V.p.w) / 3.0) / (0.5 * U * U);
    sp = length(u) / U;
    // velocity gradient tensor by central differences
    let dx = 0.5 * (vel(p + vec3<i32>(1, 0, 0)) - vel(p - vec3<i32>(1, 0, 0)));
    let dy = 0.5 * (vel(p + vec3<i32>(0, 1, 0)) - vel(p - vec3<i32>(0, 1, 0)));
    let dz = 0.5 * (vel(p + vec3<i32>(0, 0, 1)) - vel(p - vec3<i32>(0, 0, 1)));
    // J[i][j] = d u_i / d x_j
    let j00 = dx.x; let j01 = dy.x; let j02 = dz.x;
    let j10 = dx.y; let j11 = dy.y; let j12 = dz.y;
    let j20 = dx.z; let j21 = dy.z; let j22 = dz.z;
    let s01 = 0.5 * (j01 + j10); let s02 = 0.5 * (j02 + j20); let s12 = 0.5 * (j12 + j21);
    let o01 = 0.5 * (j01 - j10); let o02 = 0.5 * (j02 - j20); let o12 = 0.5 * (j12 - j21);
    let S2 = j00 * j00 + j11 * j11 + j22 * j22 + 2.0 * (s01 * s01 + s02 * s02 + s12 * s12);
    let O2 = 2.0 * (o01 * o01 + o02 * o02 + o12 * o12);
    q = 0.5 * (O2 - S2) / V.p.y;
    let w = vec3<f32>(j21 - j12, j02 - j20, j10 - j01);
    vort = length(w) / V.p.z;
  }
  textureStore(t0, p, vec4<f32>(u, cp));
  textureStore(t1, p, vec4<f32>(q, sp, vort, select(0.0, 1.0, solid)));
}
`;

export class Viz {
  readonly t0: GPUTexture;
  readonly t1: GPUTexture;
  private pipe: GPUComputePipeline;
  private bg: GPUBindGroup;
  private uni: GPUBuffer;

  constructor(private device: GPUDevice, private lbm: LBM) {
    const size: [number, number, number] = [lbm.nx, lbm.ny, lbm.nz];
    const usage = GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING;
    this.t0 = device.createTexture({ size, dimension: '3d', format: 'rgba16float', usage, label: 'viz.t0' });
    this.t1 = device.createTexture({ size, dimension: '3d', format: 'rgba16float', usage, label: 'viz.t1' });
    this.uni = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const module = device.createShaderModule({ code: WGSL, label: 'viz' });
    this.pipe = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
    this.bg = device.createBindGroup({
      layout: this.pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: lbm.mac } },
        { binding: 2, resource: { buffer: lbm.flags } },
        { binding: 3, resource: this.t0.createView({ dimension: '3d' }) },
        { binding: 4, resource: this.t1.createView({ dimension: '3d' }) },
      ],
    });
  }

  /** @param qNorm normaliser for Q, vortNorm for vorticity (both in lattice units). */
  update(enc: GPUCommandEncoder, uLat: number, qNorm: number, vortNorm: number, rhoRef = 1) {
    const buf = new ArrayBuffer(32);
    new Uint32Array(buf).set([this.lbm.nx, this.lbm.ny, this.lbm.nz, this.lbm.N], 0);
    new Float32Array(buf).set([uLat, qNorm, vortNorm, rhoRef], 4);
    this.device.queue.writeBuffer(this.uni, 0, buf);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.bg);
    pass.dispatchWorkgroups(Math.ceil(this.lbm.nx / 8), Math.ceil(this.lbm.ny / 8), Math.ceil(this.lbm.nz / 4));
    pass.end();
  }

  destroy() {
    this.t0.destroy();
    this.t1.destroy();
    this.uni.destroy();
  }
}
