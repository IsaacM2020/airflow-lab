import { FRAME_WGSL } from './frame';
import { DEPTH_FORMAT, HDR_FORMAT, type Scene } from './scene';

// =====================================================================================
// Smoke streamlines: GPU particles advected through the live velocity field, drawn as
// glowing fading ribbons (a ring buffer of the last K positions per particle).
// =====================================================================================
const PARTICLE_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var viz0: texture_3d<f32>;
@group(0) @binding(2) var viz1: texture_3d<f32>;
@group(0) @binding(3) var vizS: sampler;

struct PartParams {
  counts: vec4<u32>,   // N, K, slot (newest write), frame
  seed: vec4<f32>,     // y0, y1, z0, z1 of the smoke rake
  adv: vec4<f32>,      // advect dt (lattice steps per frame), inlet x, life min, life max
}
@group(1) @binding(0) var<uniform> PP: PartParams;
@group(1) @binding(1) var<storage, read_write> trailW: array<vec4<f32>>;
@group(1) @binding(2) var<storage, read_write> life: array<f32>;

fn vel(p: vec3<f32>) -> vec3<f32> {
  return textureSampleLevel(viz0, vizS, p / F.dom.xyz, 0.0).xyz;
}

@compute @workgroup_size(64)
fn advect(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= PP.counts.x) { return; }
  let K = PP.counts.y;
  let slot = PP.counts.z;
  let prev = (slot + K - 1u) % K;
  let head = trailW[i * K + prev];
  var pos = head.xyz;
  var age = head.w;
  var l = life[i];
  var respawn = l <= 0.0;
  if (!respawn) {
    let dt = PP.adv.x;
    let u1 = vel(pos);
    let u2 = vel(pos + 0.5 * dt * u1);
    pos = pos + dt * u2;
    age += 1.0;
    l -= 1.0;
    if (any(pos < vec3<f32>(0.5)) || any(pos > F.dom.xyz - vec3<f32>(1.5))) {
      respawn = true;
    } else if (textureSampleLevel(viz1, vizS, pos / F.dom.xyz, 0.0).a > 0.5) {
      respawn = true;
    }
  }
  if (respawn) {
    let base = i * 7919u + PP.counts.w * 104729u;
    let r1 = hash11(base + 1u);
    let r2 = hash11(base + 2u);
    let r3 = hash11(base + 3u);
    let r4 = hash11(base + 4u);
    pos = vec3<f32>(PP.adv.y + r4 * r4 * 46.0, mix(PP.seed.x, PP.seed.y, r1), mix(PP.seed.z, PP.seed.w, r2));
    age = 0.0;
    l = mix(PP.adv.z, PP.adv.w, r3);
  }
  trailW[i * K + slot] = vec4<f32>(pos, age);
  life[i] = l;
}

// ---- ribbon rendering
@group(1) @binding(1) var<storage, read> trail: array<vec4<f32>>;

struct RO {
  @builtin(position) p: vec4<f32>,
  @location(0) col: vec4<f32>,
  @location(1) side: f32,
}

@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> RO {
  var o: RO;
  o.p = vec4<f32>(2.0, 2.0, 2.0, 1.0);
  o.col = vec4<f32>(0.0);
  o.side = 0.0;
  let K = PP.counts.y;
  let i = ii / (K - 1u);
  let j = ii % (K - 1u);
  let slot = PP.counts.z;
  let sNew = (slot + K - j) % K;
  let sOld = (slot + K - j - 1u) % K;
  let a = trail[i * K + sOld];
  let b = trail[i * K + sNew];
  // a valid segment joins consecutive frames of one particle life
  if (b.w < 0.5 || abs(a.w - (b.w - 1.0)) > 0.01) { return o; }
  let ca = F.viewProj * vec4<f32>(a.xyz, 1.0);
  let cb = F.viewProj * vec4<f32>(b.xyz, 1.0);
  if (ca.w <= 0.1 || cb.w <= 0.1) { return o; }
  let half = F.screen.xy * 0.5;
  let sa = ca.xy / ca.w * half;
  let sb = cb.xy / cb.w * half;
  var dir = sb - sa;
  let len = length(dir);
  dir = select(vec2<f32>(1.0, 0.0), dir / len, len > 1e-4);
  let perp = vec2<f32>(-dir.y, dir.x);
  // quad corners: 0:(a,-) 1:(a,+) 2:(b,-) 3:(b,-) 4:(a,+) 5:(b,+)
  var endB = array<f32, 6>(0.0, 0.0, 1.0, 1.0, 0.0, 1.0);
  var sideS = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
  let eb = endB[vi];
  let sd = sideS[vi];
  let taper = 1.0 - 0.55 * f32(j) / f32(K - 1u);
  let widthPx = F.misc2.w * taper;
  var clip = select(ca, cb, eb > 0.5);
  clip.x += perp.x * sd * widthPx / half.x * clip.w;
  clip.y += perp.y * sd * widthPx / half.y * clip.w;
  o.p = clip;
  o.side = sd;
  let u = textureSampleLevel(viz0, vizS, b.xyz / F.dom.xyz, 0.0).xyz;
  let sp = length(u) / F.dom.w;
  // calm air reads cool blue, accelerated air warms through green to orange, the slow wake goes deep blue
  let col = speedColor(sp);
  let fade = pow(1.0 - f32(j) / f32(K - 1u), 1.4)
           * smoothstep(0.0, 8.0, b.w)
           * (1.0 - smoothstep(0.80, 0.98, b.x / F.dom.x));
  o.col = vec4<f32>(col * 1.5, fade * 0.26 * F.layers.w);
  return o;
}

@fragment fn fs(i: RO) -> @location(0) vec4<f32> {
  let e = 1.0 - i.side * i.side;
  return vec4<f32>(i.col.rgb, i.col.a * (0.25 + 0.75 * e));
}
`;

export interface StreamOptions {
  count: number;
  trail: number;
}

export class Streamlines {
  private computePipe: GPUComputePipeline;
  private drawPipe: GPURenderPipeline;
  private uni: GPUBuffer;
  private trail: GPUBuffer;
  private life: GPUBuffer;
  private bgCompute: GPUBindGroup;
  private bgDraw: GPUBindGroup;
  private slot = 0;
  private frame = 0;
  readonly count: number;
  readonly K: number;
  seed: [number, number, number, number] = [0, 1, 0, 1];
  dt = 6;
  lifeRange: [number, number] = [260, 520];

  constructor(private device: GPUDevice, private scene: Scene, opts: StreamOptions, nx: number) {
    this.count = opts.count;
    this.K = opts.trail;
    const module = device.createShaderModule({ code: PARTICLE_WGSL, label: 'streamlines' });
    const S = GPUShaderStage.COMPUTE, V = GPUShaderStage.VERTEX;
    const layoutC = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: S | V, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S, buffer: { type: 'storage' } },
        { binding: 2, visibility: S, buffer: { type: 'storage' } },
      ],
    });
    const layoutD = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: S | V, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.computePipe = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [scene.frameLayout, layoutC] }),
      compute: { module, entryPoint: 'advect' },
    });
    this.drawPipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [scene.frameLayout, layoutD] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module, entryPoint: 'fs',
        targets: [{
          format: HDR_FORMAT,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one', operation: 'add' },
            alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
          },
        }],
      },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
    });
    this.uni = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.trail = device.createBuffer({ size: this.count * this.K * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.life = device.createBuffer({ size: this.count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bgCompute = device.createBindGroup({
      layout: layoutC,
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.trail } },
        { binding: 2, resource: { buffer: this.life } },
      ],
    });
    this.bgDraw = device.createBindGroup({
      layout: layoutD,
      entries: [
        { binding: 0, resource: { buffer: this.uni } },
        { binding: 1, resource: { buffer: this.trail } },
      ],
    });
    this.nx = nx;
  }
  private nx: number;

  /** Scatter particles along the domain so the first frame is already full of smoke. */
  reset(nx: number, dom: [number, number, number]) {
    const head = new Float32Array(this.count * this.K * 4);
    const life = new Float32Array(this.count);
    const [y0, y1, z0, z1] = this.seed;
    for (let i = 0; i < this.count; i++) {
      // slot K-1 holds the "previous" head for the very first advect (slot = 0)
      const o = (i * this.K + (this.K - 1)) * 4;
      head[o] = 2 + Math.random() * (nx * 0.55);
      head[o + 1] = y0 + Math.random() * (y1 - y0);
      head[o + 2] = z0 + Math.random() * (z1 - z0);
      head[o + 3] = 12;
      life[i] = 30 + Math.random() * (this.lifeRange[1] - 30);
    }
    void dom;
    this.device.queue.writeBuffer(this.trail, 0, head);
    this.device.queue.writeBuffer(this.life, 0, life);
    this.slot = 0;
    this.frame = 0;
  }

  private writeUniform() {
    const buf = new ArrayBuffer(48);
    new Uint32Array(buf).set([this.count, this.K, this.slot, this.frame], 0);
    new Float32Array(buf).set([...this.seed, this.dt, 2, this.lifeRange[0], this.lifeRange[1]], 4);
    this.device.queue.writeBuffer(this.uni, 0, buf);
  }

  /** Advance every particle one step. Call once per rendered frame while the flow is running. */
  advance(enc: GPUCommandEncoder) {
    this.slot = (this.slot + 1) % this.K;
    this.frame++;
    this.writeUniform();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.computePipe);
    pass.setBindGroup(0, this.scene.frameBG);
    pass.setBindGroup(1, this.bgCompute);
    pass.dispatchWorkgroups(Math.ceil(this.count / 64));
    pass.end();
  }

  draw(pass: GPURenderPassEncoder) {
    this.writeUniform();
    pass.setPipeline(this.drawPipe);
    pass.setBindGroup(0, this.scene.frameBG);
    pass.setBindGroup(1, this.bgDraw);
    pass.draw(6, this.count * (this.K - 1));
  }

  destroy() {
    this.trail.destroy();
    this.life.destroy();
    this.uni.destroy();
  }
}

// =====================================================================================
// Slice plane through the flow, coloured by speed or pressure.
// =====================================================================================
const SLICE_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var viz0: texture_3d<f32>;
@group(0) @binding(2) var viz1: texture_3d<f32>;
@group(0) @binding(3) var vizS: sampler;

struct SO { @builtin(position) p: vec4<f32>, @location(0) uvw: vec3<f32>, @location(1) uv: vec2<f32> }

@vertex fn vs(@builtin(vertex_index) i: u32) -> SO {
  var c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  let uv = c[i];
  let f = clamp(F.misc.y, 0.001, 0.999);
  let axis = i32(F.misc2.y + 0.5);
  var uvw: vec3<f32>;
  if (axis == 0) { uvw = vec3<f32>(f, uv.x, uv.y); }
  else if (axis == 1) { uvw = vec3<f32>(uv.x, f, uv.y); }
  else { uvw = vec3<f32>(uv.x, uv.y, f); }
  var o: SO;
  o.p = F.viewProj * vec4<f32>(uvw * F.dom.xyz, 1.0);
  o.uvw = uvw;
  o.uv = uv;
  return o;
}

@fragment fn fs(i: SO) -> @location(0) vec4<f32> {
  let a = textureSampleLevel(viz0, vizS, i.uvw, 0.0);
  let b = textureSampleLevel(viz1, vizS, i.uvw, 0.0);
  if (b.a > 0.5) { discard; }
  var col: vec3<f32>;
  if (F.misc2.x < 0.5) {
    col = speedColor(b.g);
  } else {
    col = coolwarmDark(a.a / F.misc.x);
  }
  let e = min(min(i.uv.x, 1.0 - i.uv.x), min(i.uv.y, 1.0 - i.uv.y));
  let border = 1.0 - smoothstep(0.0, 0.006, e);
  col = mix(col * 0.95, vec3<f32>(0.6, 0.85, 1.0), border * 0.7);
  return vec4<f32>(col * 0.95, F.layers.y * (0.82 + 0.18 * border));
}
`;

export class SliceLayer {
  private pipe: GPURenderPipeline;
  constructor(device: GPUDevice, private scene: Scene) {
    const module = device.createShaderModule({ code: SLICE_WGSL, label: 'slice' });
    this.pipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [scene.frameLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module, entryPoint: 'fs',
        targets: [{
          format: HDR_FORMAT,
          blend: {
            color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
      primitive: { cullMode: 'none' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
    });
  }
  draw(pass: GPURenderPassEncoder) {
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.scene.frameBG);
    pass.draw(6);
  }
}

// =====================================================================================
// Vortex tubes: ray-march the Q-criterion volume and shade the first iso-surface.
// =====================================================================================
const VORTEX_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var viz0: texture_3d<f32>;
@group(0) @binding(2) var viz1: texture_3d<f32>;
@group(0) @binding(3) var vizS: sampler;
@group(1) @binding(0) var depthT: texture_depth_2d;

struct O { @builtin(position) p: vec4<f32>, @location(0) ndc: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> O {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: O; let p = pos[i];
  o.p = vec4(p, 0.0, 1.0); o.ndc = p; return o;
}

fn qAt(p: vec3<f32>) -> f32 { return textureSampleLevel(viz1, vizS, p / F.dom.xyz, 0.0).r; }

@fragment fn fs(i: O) -> @location(0) vec4<f32> {
  let px = vec2<i32>(i.p.xy);
  let depth = textureLoad(depthT, px, 0);
  let a = F.invViewProj * vec4(i.ndc, 0.0, 1.0);
  let b = F.invViewProj * vec4(i.ndc, 1.0, 1.0);
  let eye = F.eye.xyz;
  let rd = normalize(b.xyz / b.w - a.xyz / a.w);
  var tScene = 1.0e9;
  if (depth < 1.0) {
    let sp = F.invViewProj * vec4(i.ndc, depth, 1.0);
    tScene = length(sp.xyz / sp.w - eye);
  }
  let inv = 1.0 / rd;
  let t0 = (vec3<f32>(0.0) - eye) * inv;
  let t1 = (F.dom.xyz - eye) * inv;
  let tlo = min(t0, t1);
  let thi = max(t0, t1);
  let tmin = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
  let tmax = min(min(thi.x, thi.y), thi.z);
  let tEnd = min(tmax, tScene);
  if (tmin >= tEnd) { discard; }

  let thr = F.misc.z;
  let step = 1.15;
  var t = tmin;
  var prev = qAt(eye + rd * t);
  var hit = false;
  var th = 0.0;
  for (var k = 0; k < 380; k++) {
    t += step;
    if (t > tEnd) { break; }
    let q = qAt(eye + rd * t);
    if (q >= thr && prev < thr) {
      var lo = t - step;
      var hi = t;
      for (var r = 0; r < 5; r++) {
        let mid = 0.5 * (lo + hi);
        if (qAt(eye + rd * mid) >= thr) { hi = mid; } else { lo = mid; }
      }
      th = hi;
      hit = true;
      break;
    }
    prev = q;
  }
  if (!hit) { discard; }
  let hp = eye + rd * th;
  let h = 1.0;
  let g = vec3<f32>(
    qAt(hp + vec3(h, 0.0, 0.0)) - qAt(hp - vec3(h, 0.0, 0.0)),
    qAt(hp + vec3(0.0, h, 0.0)) - qAt(hp - vec3(0.0, h, 0.0)),
    qAt(hp + vec3(0.0, 0.0, h)) - qAt(hp - vec3(0.0, 0.0, h)));
  var n = -normalize(g + vec3<f32>(1e-6));
  let v = -rd;
  if (dot(n, v) < 0.0) { n = -n; }
  let ndv = clamp(dot(n, v), 0.0, 1.0);
  let rim = pow(1.0 - ndv, 2.2);
  let s = textureSampleLevel(viz1, vizS, hp / F.dom.xyz, 0.0);
  let core = clamp(s.b / 9.0, 0.0, 1.0);
  let base = mix(vec3<f32>(0.05, 0.42, 1.0), vec3<f32>(1.0, 0.42, 0.06), core * core);
  let l = max(dot(n, normalize(vec3<f32>(-0.3, 0.8, 0.4))), 0.0);
  var col = base * (0.22 + 0.55 * l) + vec3<f32>(0.20, 0.55, 1.0) * rim * 0.55;
  let alpha = F.layers.z * (0.55 + 0.35 * rim);
  return vec4<f32>(col * alpha, alpha);
}
`;

export class VortexLayer {
  private pipe: GPURenderPipeline;
  private depthLayout: GPUBindGroupLayout;
  private depthBG: GPUBindGroup | null = null;
  private depthTex: GPUTexture | null = null;
  constructor(private device: GPUDevice, private scene: Scene) {
    const module = device.createShaderModule({ code: VORTEX_WGSL, label: 'vortex' });
    this.depthLayout = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }],
    });
    this.pipe = device.createRenderPipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [scene.frameLayout, this.depthLayout] }),
      vertex: { module, entryPoint: 'vs' },
      fragment: {
        module, entryPoint: 'fs',
        targets: [{
          format: HDR_FORMAT,
          blend: {
            color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
            alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          },
        }],
      },
    });
  }
  draw(enc: GPUCommandEncoder) {
    if (this.depthTex !== this.scene.depth) {
      this.depthTex = this.scene.depth;
      this.depthBG = this.device.createBindGroup({
        layout: this.depthLayout,
        entries: [{ binding: 0, resource: this.scene.depth.createView({ aspect: 'depth-only' }) }],
      });
    }
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.scene.hdr.createView(), loadOp: 'load', storeOp: 'store' }],
    });
    pass.setPipeline(this.pipe);
    pass.setBindGroup(0, this.scene.frameBG);
    pass.setBindGroup(1, this.depthBG!);
    pass.draw(3);
    pass.end();
  }
}
