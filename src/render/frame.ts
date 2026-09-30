import type { Mat4 } from './math';

/** WGSL shared by every render pass: the per-frame uniform block, palettes, studio lighting. */
export const FRAME_WGSL = /* wgsl */ `
struct Frame {
  viewProj: mat4x4<f32>,
  invViewProj: mat4x4<f32>,
  eye: vec4<f32>,        // xyz camera position, w time (s)
  screen: vec4<f32>,     // w, h, 1/w, 1/h
  dom: vec4<f32>,        // nx, ny, nz, lattice inflow speed
  layers: vec4<f32>,     // pressure paint amount, slice opacity, vortex opacity, streamline opacity
  misc: vec4<f32>,       // cpScale, slice position (0..1), vortex threshold, exposure
  misc2: vec4<f32>,      // slice mode (0 speed,1 pressure), slice axis (0 x,1 y,2 z), ground on, particle width px
  model0: vec4<f32>,     // body -> cell transform, rows
  model1: vec4<f32>,
  model2: vec4<f32>,
}

fn turbo(t0: f32) -> vec3<f32> {
  let t = clamp(t0, 0.0, 1.0);
  let r = 0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  let g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  let b = 0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0));
}

// blue (suction) - light grey - red (stagnation) diverging map for pressure. The square-root stretch
// keeps mild pressures visible instead of letting them wash out to grey.
fn coolwarm(x: f32) -> vec3<f32> {
  let e = sign(x) * pow(min(abs(x), 1.0), 0.62);
  let t = clamp(e * 0.5 + 0.5, 0.0, 1.0);
  let cold = vec3<f32>(0.03, 0.22, 1.00);
  let mid = vec3<f32>(0.66, 0.70, 0.80);
  let hot = vec3<f32>(1.00, 0.10, 0.04);
  return select(mix(cold, mid, t * 2.0), mix(mid, hot, t * 2.0 - 1.0), t > 0.5);
}

// Same idea for slices, on a dark neutral so a translucent plane does not wash the scene out.
fn coolwarmDark(x: f32) -> vec3<f32> {
  let e = sign(x) * pow(min(abs(x), 1.0), 0.62);
  let t = clamp(e * 0.5 + 0.5, 0.0, 1.0);
  let cold = vec3<f32>(0.04, 0.32, 1.00);
  let mid = vec3<f32>(0.055, 0.075, 0.12);
  let hot = vec3<f32>(1.00, 0.22, 0.06);
  return select(mix(cold, mid, t * 2.0), mix(mid, hot, t * 2.0 - 1.0), t > 0.5);
}

// speed palette shared by streamlines and slices: calm air cool blue, accelerated air warm
fn speedColor(sp: f32) -> vec3<f32> {
  return turbo(clamp(0.27 + (sp - 1.0) * 0.95, 0.02, 0.98));
}

fn hash11(n: u32) -> f32 {
  var x = n * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
  x = (x >> 22u) ^ x;
  return f32(x) / 4294967295.0;
}

// Procedural studio used for reflections: dark dome, cool horizon and three soft boxes.
fn studio(d: vec3<f32>, rough: f32) -> vec3<f32> {
  let up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(vec3<f32>(0.010, 0.012, 0.020), vec3<f32>(0.05, 0.075, 0.13), pow(up, 1.3));
  col += vec3<f32>(0.06, 0.11, 0.20) * exp(-abs(d.y) * (7.0 - rough * 5.0)) * 0.5;
  let w = 0.07 + rough * 0.30;
  let l1 = normalize(vec3<f32>(-0.35, 0.90, 0.25));
  let l2 = normalize(vec3<f32>(0.85, 0.30, -0.45));
  let l3 = normalize(vec3<f32>(-0.75, 0.25, -0.60));
  col += vec3<f32>(2.6, 2.7, 3.0) * smoothstep(1.0 - w, 1.0 - w * 0.25, dot(d, l1));
  col += vec3<f32>(0.9, 1.15, 1.8) * smoothstep(1.0 - w * 0.8, 1.0 - w * 0.2, dot(d, l2));
  col += vec3<f32>(1.5, 1.0, 0.7) * smoothstep(1.0 - w * 0.6, 1.0 - w * 0.15, dot(d, l3));
  return col;
}

// Soft diffuse lighting: cool hemisphere ambient plus a warm key from above-left.
fn diffuseLight(n: vec3<f32>) -> vec3<f32> {
  let amb = mix(vec3<f32>(0.030, 0.034, 0.048), vec3<f32>(0.16, 0.19, 0.27), n.y * 0.5 + 0.5);
  let key = max(dot(n, normalize(vec3<f32>(-0.4, 0.85, 0.35))), 0.0);
  let fill = max(dot(n, normalize(vec3<f32>(0.7, 0.2, -0.6))), 0.0);
  return amb + vec3<f32>(0.95, 0.92, 0.86) * key * 0.85 + vec3<f32>(0.25, 0.40, 0.65) * fill * 0.45;
}

fn backdrop(d: vec3<f32>) -> vec3<f32> {
  let up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(vec3<f32>(0.006, 0.008, 0.014), vec3<f32>(0.020, 0.032, 0.062), pow(up, 0.9));
  col += vec3<f32>(0.020, 0.045, 0.085) * exp(-abs(d.y) * 9.0);
  return col;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
  let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn bodyToCell(F: Frame, p: vec3<f32>) -> vec3<f32> {
  let q = vec4<f32>(p, 1.0);
  return vec3<f32>(dot(F.model0, q), dot(F.model1, q), dot(F.model2, q));
}
fn bodyToCellDir(F: Frame, n: vec3<f32>) -> vec3<f32> {
  return normalize(vec3<f32>(dot(F.model0.xyz, n), dot(F.model1.xyz, n), dot(F.model2.xyz, n)));
}
`;

export class FrameUniform {
  readonly data = new Float32Array(72);
  readonly buffer: GPUBuffer;
  constructor(private device: GPUDevice) {
    this.buffer = device.createBuffer({ size: 288, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, label: 'frame' });
  }
  setMatrices(viewProj: Mat4, invViewProj: Mat4) {
    this.data.set(viewProj, 0);
    this.data.set(invViewProj, 16);
  }
  setEye(x: number, y: number, z: number, t: number) { this.data.set([x, y, z, t], 32); }
  setScreen(w: number, h: number) { this.data.set([w, h, 1 / w, 1 / h], 36); }
  setDomain(nx: number, ny: number, nz: number, uLat: number) { this.data.set([nx, ny, nz, uLat], 40); }
  setLayers(paint: number, slice: number, vortex: number, streams: number) { this.data.set([paint, slice, vortex, streams], 44); }
  setMisc(cpScale: number, slicePos: number, vortexThr: number, exposure: number) { this.data.set([cpScale, slicePos, vortexThr, exposure], 48); }
  setMisc2(sliceMode: number, sliceAxis: number, ground: number, particleWidth: number) { this.data.set([sliceMode, sliceAxis, ground, particleWidth], 52); }
  /** 3x4 row-major body->cell matrix as produced by bodyToCell(). */
  setModel(m: number[]) { this.data.set(m, 56); }
  upload() { this.device.queue.writeBuffer(this.buffer, 0, this.data); }
}
