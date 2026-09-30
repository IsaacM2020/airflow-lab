import { FRAME_WGSL } from './frame';
import { HDR_FORMAT } from './scene';

const POST_WGSL = /* wgsl */ `
${FRAME_WGSL}
@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var src2: texture_2d<f32>;   // bloom result (composite pass only)

struct O { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> O {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: O; let p = pos[i];
  o.p = vec4(p, 0.0, 1.0); o.uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); return o;
}

fn tap(uv: vec2<f32>) -> vec3<f32> { return textureSampleLevel(src, samp, uv, 0.0).rgb; }

@fragment fn down(i: O) -> @location(0) vec4<f32> {
  let d = vec2<f32>(textureDimensions(src));
  let t = 1.0 / d;
  var c = tap(i.uv + vec2(-1.0, -1.0) * t) + tap(i.uv + vec2(1.0, -1.0) * t)
        + tap(i.uv + vec2(-1.0, 1.0) * t) + tap(i.uv + vec2(1.0, 1.0) * t);
  c = c * 0.25 * 0.5 + tap(i.uv) * 0.5;
  return vec4(c, 1.0);
}
@fragment fn down_first(i: O) -> @location(0) vec4<f32> {
  // soft knee threshold so only bright things (streamlines, highlights) glow
  let d = vec2<f32>(textureDimensions(src));
  let t = 1.0 / d;
  var c = vec3<f32>(0.0);
  var w = 0.0;
  for (var k = 0; k < 4; k++) {
    let o = vec2(f32(k & 1) * 2.0 - 1.0, f32(k >> 1) * 2.0 - 1.0);
    let s = tap(i.uv + o * t);
    let lum = max(max(s.r, s.g), s.b);
    let wt = 1.0 / (1.0 + lum);
    c += s * wt; w += wt;
  }
  c = c / w;
  let lum = max(max(c.r, c.g), c.b);
  let k = clamp((lum - 1.0) / 1.2, 0.0, 1.0);
  return vec4(c * k * k * (3.0 - 2.0 * k), 1.0);
}
@fragment fn up(i: O) -> @location(0) vec4<f32> {
  let d = vec2<f32>(textureDimensions(src));
  let t = 1.0 / d;
  var c = tap(i.uv + vec2(-0.5, -0.5) * t) + tap(i.uv + vec2(0.5, -0.5) * t)
        + tap(i.uv + vec2(-0.5, 0.5) * t) + tap(i.uv + vec2(0.5, 0.5) * t);
  return vec4(c * 0.25, 1.0);
}

fn srgb(c: vec3<f32>) -> vec3<f32> {
  return select(1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3(0.0031308));
}

@fragment fn composite(i: O) -> @location(0) vec4<f32> {
  var col = textureSampleLevel(src, samp, i.uv, 0.0).rgb;
  let glow = textureSampleLevel(src2, samp, i.uv, 0.0).rgb;
  col = (col + glow * 0.6) * F.misc.w;
  col = aces(col);
  let q = i.uv * 2.0 - 1.0;
  col *= 1.0 - 0.32 * dot(q * vec2(0.8, 1.0), q * vec2(0.8, 1.0)) * 0.55;
  col = srgb(col);
  let n = hash11(u32(i.p.x) * 1973u + u32(i.p.y) * 9277u + u32(F.eye.w * 60.0) * 26699u);
  col += (n - 0.5) / 255.0;
  return vec4(col, 1.0);
}

fn luma(c: vec3<f32>) -> f32 { return dot(c, vec3(0.299, 0.587, 0.114)); }
@fragment fn fxaa(i: O) -> @location(0) vec4<f32> {
  let t = F.screen.zw;
  let m = textureSampleLevel(src, samp, i.uv, 0.0).rgb;
  let n = textureSampleLevel(src, samp, i.uv + vec2(0.0, -t.y), 0.0).rgb;
  let s = textureSampleLevel(src, samp, i.uv + vec2(0.0, t.y), 0.0).rgb;
  let e = textureSampleLevel(src, samp, i.uv + vec2(t.x, 0.0), 0.0).rgb;
  let w = textureSampleLevel(src, samp, i.uv + vec2(-t.x, 0.0), 0.0).rgb;
  let lm = luma(m); let ln = luma(n); let ls = luma(s); let le = luma(e); let lw = luma(w);
  let lo = min(lm, min(min(ln, ls), min(le, lw)));
  let hi = max(lm, max(max(ln, ls), max(le, lw)));
  let range = hi - lo;
  if (range < max(0.04, hi * 0.11)) { return vec4(m, 1.0); }
  let dirx = -((ln + ls) - 2.0 * lm);
  let diry = ((le + lw) - 2.0 * lm);
  var dir = vec2(dirx, diry);
  let reduce = max((ln + ls + le + lw) * 0.25 * 0.125, 1.0 / 128.0);
  let rcp = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
  dir = clamp(dir * rcp, vec2(-8.0), vec2(8.0)) * t;
  let a = 0.5 * (textureSampleLevel(src, samp, i.uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb
               + textureSampleLevel(src, samp, i.uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
  let b = a * 0.5 + 0.25 * (textureSampleLevel(src, samp, i.uv + dir * -0.5, 0.0).rgb
                          + textureSampleLevel(src, samp, i.uv + dir * 0.5, 0.0).rgb);
  let lb = luma(b);
  return vec4(select(b, a, lb < lo || lb > hi), 1.0);
}
`;

const LEVELS = 6;

export class Post {
  private module: GPUShaderModule;
  private pipes: Record<string, GPURenderPipeline> = {};
  private sampler: GPUSampler;
  private bloom: GPUTexture[] = [];
  private ldr!: GPUTexture;
  private blank: GPUTexture;
  private layout: GPUBindGroupLayout;
  private w = 0;
  private h = 0;

  constructor(private device: GPUDevice, private frameBuffer: GPUBuffer, private canvasFormat: GPUTextureFormat) {
    this.module = device.createShaderModule({ code: POST_WGSL, label: 'post' });
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.blank = device.createTexture({ size: [1, 1], format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING });
    const F = GPUShaderStage.FRAGMENT;
    this.layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: F, buffer: { type: 'uniform' } },
        { binding: 1, visibility: F, sampler: { type: 'filtering' } },
        { binding: 2, visibility: F, texture: { sampleType: 'float' } },
        { binding: 3, visibility: F, texture: { sampleType: 'float' } },
      ],
    });
    const plLayout = device.createPipelineLayout({ bindGroupLayouts: [this.layout] });
    const mk = (frag: string, format: GPUTextureFormat, additive = false) =>
      device.createRenderPipeline({
        layout: plLayout,
        vertex: { module: this.module, entryPoint: 'vs' },
        fragment: {
          module: this.module, entryPoint: frag,
          targets: [{
            format,
            blend: additive ? { color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' }, alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' } } : undefined,
          }],
        },
      });
    this.pipes.downFirst = mk('down_first', HDR_FORMAT);
    this.pipes.down = mk('down', HDR_FORMAT);
    this.pipes.up = mk('up', HDR_FORMAT, true);
    this.pipes.composite = mk('composite', 'rgba8unorm');
    this.pipes.fxaa = mk('fxaa', canvasFormat);
  }

  resize(w: number, h: number) {
    if (w === this.w && h === this.h) return;
    this.w = w; this.h = h;
    this.bloom.forEach((t) => t.destroy());
    this.bloom = [];
    for (let i = 0; i < LEVELS; i++) {
      this.bloom.push(this.device.createTexture({
        size: [Math.max(1, w >> (i + 1)), Math.max(1, h >> (i + 1))], format: HDR_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }));
    }
    this.ldr?.destroy();
    this.ldr = this.device.createTexture({ size: [w, h], format: 'rgba8unorm', usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
  }

  private bg(pipe: GPURenderPipeline, src: GPUTexture, src2?: GPUTexture) {
    return this.device.createBindGroup({
      layout: this.layout,
      entries: [
        { binding: 0, resource: { buffer: this.frameBuffer } },
        { binding: 1, resource: this.sampler },
        { binding: 2, resource: src.createView() },
        { binding: 3, resource: (src2 ?? this.blank).createView() },
      ],
    });
  }

  private pass(enc: GPUCommandEncoder, pipe: GPURenderPipeline, bg: GPUBindGroup, target: GPUTexture | GPUTextureView, load: GPULoadOp = 'clear') {
    const view = 'createView' in target ? target.createView() : target;
    const p = enc.beginRenderPass({ colorAttachments: [{ view, loadOp: load, storeOp: 'store', clearValue: [0, 0, 0, 1] }] });
    p.setPipeline(pipe);
    p.setBindGroup(0, bg);
    p.draw(3);
    p.end();
  }

  run(enc: GPUCommandEncoder, hdr: GPUTexture, canvasView: GPUTextureView) {
    // bloom: down chain then additive up chain
    this.pass(enc, this.pipes.downFirst, this.bg(this.pipes.downFirst, hdr), this.bloom[0]);
    for (let i = 1; i < LEVELS; i++) this.pass(enc, this.pipes.down, this.bg(this.pipes.down, this.bloom[i - 1]), this.bloom[i]);
    for (let i = LEVELS - 1; i > 0; i--) this.pass(enc, this.pipes.up, this.bg(this.pipes.up, this.bloom[i]), this.bloom[i - 1], 'load');
    this.pass(enc, this.pipes.composite, this.bg(this.pipes.composite, hdr, this.bloom[0]), this.ldr);
    this.pass(enc, this.pipes.fxaa, this.bg(this.pipes.fxaa, this.ldr), canvasView);
  }
}
