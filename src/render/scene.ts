import { FRAME_WGSL, FrameUniform } from './frame';
import type { Model } from './model';

export const HDR_FORMAT: GPUTextureFormat = 'rgba16float';
export const DEPTH_FORMAT: GPUTextureFormat = 'depth32float';

const SCENE_WGSL = /* wgsl */ `
${FRAME_WGSL}

@group(0) @binding(0) var<uniform> F: Frame;
@group(0) @binding(1) var viz0: texture_3d<f32>;   // (u.xyz, Cp)
@group(0) @binding(2) var viz1: texture_3d<f32>;   // (Q, speed, vorticity, solid)
@group(0) @binding(3) var vizS: sampler;

struct Mat { color: vec4<f32>, spec: f32, shin: f32, hasTex: f32, pad: f32 }
@group(1) @binding(0) var<uniform> M: Mat;
@group(1) @binding(1) var albedoT: texture_2d<f32>;
@group(1) @binding(2) var albedoS: sampler;

// ---------------------------------------------------------------- background
struct BgOut { @builtin(position) p: vec4<f32>, @location(0) ndc: vec2<f32> }
@vertex fn bg_vs(@builtin(vertex_index) i: u32) -> BgOut {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: BgOut; let p = pos[i];
  o.p = vec4(p, 1.0, 1.0); o.ndc = p; return o;
}
@fragment fn bg_fs(i: BgOut) -> @location(0) vec4<f32> {
  let a = F.invViewProj * vec4(i.ndc, 0.0, 1.0);
  let b = F.invViewProj * vec4(i.ndc, 1.0, 1.0);
  let d = normalize(b.xyz / b.w - a.xyz / a.w);
  var col = backdrop(d);
  let v = length(i.ndc * vec2(0.85, 1.0));
  col *= 1.0 - 0.5 * smoothstep(0.35, 1.5, v);
  return vec4(col, 1.0);
}

// ---------------------------------------------------------------- model
struct VOut {
  @builtin(position) p: vec4<f32>,
  @location(0) wp: vec3<f32>,
  @location(1) n: vec3<f32>,
  @location(2) uv: vec2<f32>,
}
fn model_vs(pos: vec3<f32>, nrm: vec3<f32>, uv: vec2<f32>, mirror: bool) -> VOut {
  var wp = bodyToCell(F, pos);
  var n = bodyToCellDir(F, nrm);
  if (mirror) { wp.y = -wp.y; n.y = -n.y; }
  var o: VOut;
  o.p = F.viewProj * vec4(wp, 1.0);
  o.wp = wp; o.n = n; o.uv = uv;
  return o;
}
@vertex fn model_main(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
  return model_vs(pos, nrm, uv, false);
}
@vertex fn model_mirror(@location(0) pos: vec3<f32>, @location(1) nrm: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
  return model_vs(pos, nrm, uv, true);
}

fn shade(i: VOut, front: bool, mirrored: bool) -> vec4<f32> {
  var N = normalize(i.n);
  if (!front) { N = -N; }
  var wpos = i.wp;
  if (mirrored) { wpos.y = -wpos.y; }
  let V = normalize(F.eye.xyz - i.wp);
  var albedo = M.color.rgb;
  var alpha = M.color.a;
  if (M.hasTex > 0.5) {
    let t = textureSample(albedoT, albedoS, i.uv);
    albedo = pow(t.rgb, vec3(2.2));
    alpha = alpha * t.a;
  }
  albedo = clamp(albedo, vec3(0.0), vec3(1.0));
  let ndv = clamp(dot(N, V), 0.0, 1.0);
  let rough = mix(0.10, 0.55, 1.0 - clamp(M.shin / 400.0, 0.0, 1.0));
  let R = reflect(-V, N);
  let f0 = 0.04 + M.spec * 0.10;
  let fres = f0 + (1.0 - f0) * pow(1.0 - ndv, 5.0);

  var ao = 1.0;
  if (F.misc2.z > 0.5) { ao = mix(0.35, 1.0, smoothstep(0.0, 5.0, wpos.y)); }
  let diff = albedo * diffuseLight(N) * ao;
  var spec = studio(R, rough) * fres * ao;

  var col = diff + spec;
  if (F.layers.x > 0.001) {
    let uvw = (wpos + N * 1.6) / F.dom.xyz;
    let cp = textureSampleLevel(viz0, vizS, uvw, 0.0).a;
    let pc = coolwarm(cp / F.misc.x);
    col = mix(col, pc * (0.55 + 0.45 * ndv) + spec * 0.5, F.layers.x);
  }
  return vec4(col, alpha);
}
@fragment fn model_fs(i: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
  return shade(i, ff, false);
}
@fragment fn model_fs_mirror(i: VOut, @builtin(front_facing) ff: bool) -> @location(0) vec4<f32> {
  let c = shade(i, ff, true);
  return vec4(c.rgb * 0.9, c.a);
}

// ---------------------------------------------------------------- ground (moving belt)
struct GOut { @builtin(position) p: vec4<f32>, @location(0) w: vec2<f32> }
@vertex fn floor_vs(@builtin(vertex_index) i: u32) -> GOut {
  var c = array<vec2<f32>, 6>(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
  let e = 260.0;
  let x = mix(-e, F.dom.x + e, c[i].x);
  let z = mix(-e, F.dom.z + e, c[i].y);
  var o: GOut;
  o.p = F.viewProj * vec4(x, 0.0, z, 1.0);
  o.w = vec2(x, z);
  return o;
}
@fragment fn floor_fs(i: GOut) -> @location(0) vec4<f32> {
  if (F.misc2.z < 0.5) { discard; }
  let dx = max(max(-i.w.x, i.w.x - F.dom.x), 0.0);
  let dz = max(max(-i.w.y, i.w.y - F.dom.z), 0.0);
  let outside = length(vec2(dx, dz));
  let fade = 1.0 - smoothstep(20.0, 230.0, outside);
  var col = vec3(0.020, 0.024, 0.032);
  // belt: soft stripes rolling with the wind
  let speed = F.dom.w * 60.0;
  let s = i.w.x - F.eye.w * speed * 0.7;
  let stripe = smoothstep(0.46, 0.5, abs(fract(s / 9.0) - 0.5));
  let lat = smoothstep(0.47, 0.5, abs(fract(i.w.y / 9.0) - 0.5));
  let inside = 1.0 - smoothstep(0.0, 6.0, outside);
  col += vec3(0.05, 0.16, 0.24) * (stripe * 0.07 + lat * 0.04) * inside;
  col += vec3(0.006, 0.016, 0.026) * (1.0 - smoothstep(0.0, 150.0, outside));
  return vec4(col, 0.80 * fade);
}

// ---------------------------------------------------------------- domain wireframe
struct LOut { @builtin(position) p: vec4<f32>, @location(0) t: f32 }
@vertex fn box_vs(@builtin(vertex_index) i: u32) -> LOut {
  var a = array<u32, 24>(0u,1u, 1u,3u, 3u,2u, 2u,0u, 4u,5u, 5u,7u, 7u,6u, 6u,4u, 0u,4u, 1u,5u, 2u,6u, 3u,7u);
  let k = a[i];
  let c = vec3<f32>(f32(k & 1u), f32((k >> 1u) & 1u), f32((k >> 2u) & 1u));
  var o: LOut;
  o.p = F.viewProj * vec4(c * F.dom.xyz, 1.0);
  o.t = f32(i & 1u);
  return o;
}
@fragment fn box_fs(i: LOut) -> @location(0) vec4<f32> {
  return vec4(vec3(0.20, 0.55, 0.85) * 0.5, 0.35);
}
`;

export class Scene {
  readonly frame: FrameUniform;
  readonly frameLayout: GPUBindGroupLayout;
  readonly matLayout: GPUBindGroupLayout;
  frameBG!: GPUBindGroup;
  hdr!: GPUTexture;
  depth!: GPUTexture;
  private pipes: Record<string, GPURenderPipeline> = {};
  private dummy3D: GPUTexture;
  private sampler: GPUSampler;
  width = 0;
  height = 0;
  private viz0: GPUTexture;
  private viz1: GPUTexture;

  constructor(private device: GPUDevice) {
    this.frame = new FrameUniform(device);
    const V = GPUShaderStage.VERTEX, Fr = GPUShaderStage.FRAGMENT, C = GPUShaderStage.COMPUTE;
    this.frameLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: V | Fr | C, buffer: { type: 'uniform' } },
        { binding: 1, visibility: V | Fr | C, texture: { viewDimension: '3d', sampleType: 'float' } },
        { binding: 2, visibility: V | Fr | C, texture: { viewDimension: '3d', sampleType: 'float' } },
        { binding: 3, visibility: V | Fr | C, sampler: { type: 'filtering' } },
      ],
    });
    this.matLayout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: Fr, buffer: { type: 'uniform' } },
        { binding: 1, visibility: Fr, texture: { sampleType: 'float' } },
        { binding: 2, visibility: Fr, sampler: { type: 'filtering' } },
      ],
    });
    this.sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
    this.dummy3D = device.createTexture({ size: [1, 1, 1], dimension: '3d', format: HDR_FORMAT, usage: GPUTextureUsage.TEXTURE_BINDING });
    this.viz0 = this.dummy3D;
    this.viz1 = this.dummy3D;
    this.rebuildFrameBG();
    this.buildPipelines();
  }

  /** Bind the flow-visualisation volumes (or the dummy 1x1x1 ones before the solver exists). */
  setViz(t0: GPUTexture | null, t1: GPUTexture | null) {
    this.viz0 = t0 ?? this.dummy3D;
    this.viz1 = t1 ?? this.dummy3D;
    this.rebuildFrameBG();
  }

  private rebuildFrameBG() {
    this.frameBG = this.device.createBindGroup({
      layout: this.frameLayout,
      entries: [
        { binding: 0, resource: { buffer: this.frame.buffer } },
        { binding: 1, resource: this.viz0.createView({ dimension: '3d' }) },
        { binding: 2, resource: this.viz1.createView({ dimension: '3d' }) },
        { binding: 3, resource: this.sampler },
      ],
    });
  }

  private buildPipelines() {
    const d = this.device;
    const module = d.createShaderModule({ code: SCENE_WGSL, label: 'scene' });
    const layout = d.createPipelineLayout({ bindGroupLayouts: [this.frameLayout, this.matLayout] });
    const layout0 = d.createPipelineLayout({ bindGroupLayouts: [this.frameLayout] });
    const vbufs: GPUVertexBufferLayout[] = [
      { arrayStride: 12, attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }] },
      { arrayStride: 12, attributes: [{ shaderLocation: 1, offset: 0, format: 'float32x3' }] },
      { arrayStride: 8, attributes: [{ shaderLocation: 2, offset: 0, format: 'float32x2' }] },
    ];
    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    this.pipes.bg = d.createRenderPipeline({
      layout: layout0,
      vertex: { module, entryPoint: 'bg_vs' },
      fragment: { module, entryPoint: 'bg_fs', targets: [{ format: HDR_FORMAT }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'always' },
    });
    const modelPipe = (frag: string, vert: string, cull: GPUCullMode, alpha: boolean, depthWrite: boolean) =>
      d.createRenderPipeline({
        layout,
        vertex: { module, entryPoint: vert, buffers: vbufs },
        fragment: { module, entryPoint: frag, targets: [{ format: HDR_FORMAT, blend: alpha ? blend : undefined }] },
        primitive: { topology: 'triangle-list', cullMode: cull },
        depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: depthWrite, depthCompare: 'less' },
      });
    this.pipes.model = modelPipe('model_fs', 'model_main', 'none', false, true);
    this.pipes.modelAlpha = modelPipe('model_fs', 'model_main', 'none', true, false);
    this.pipes.mirror = modelPipe('model_fs_mirror', 'model_mirror', 'none', false, true);
    this.pipes.floor = d.createRenderPipeline({
      layout: layout0,
      vertex: { module, entryPoint: 'floor_vs' },
      fragment: { module, entryPoint: 'floor_fs', targets: [{ format: HDR_FORMAT, blend }] },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: true, depthCompare: 'less' },
    });
    this.pipes.box = d.createRenderPipeline({
      layout: layout0,
      vertex: { module, entryPoint: 'box_vs' },
      fragment: { module, entryPoint: 'box_fs', targets: [{ format: HDR_FORMAT, blend }] },
      primitive: { topology: 'line-list' },
      depthStencil: { format: DEPTH_FORMAT, depthWriteEnabled: false, depthCompare: 'less' },
    });
  }

  resize(w: number, h: number) {
    if (w === this.width && h === this.height && this.hdr) return;
    this.width = w; this.height = h;
    this.hdr?.destroy();
    this.depth?.destroy();
    this.hdr = this.device.createTexture({
      size: [w, h], format: HDR_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.depth = this.device.createTexture({
      size: [w, h], format: DEPTH_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
  }

  /** Background, floor with reflection, model and domain box. `extra` draws flow layers into the same pass. */
  draw(enc: GPUCommandEncoder, model: Model, ground: boolean, extra?: (pass: GPURenderPassEncoder) => void, extraAfterModel?: (pass: GPURenderPassEncoder) => void) {
    const pass = enc.beginRenderPass({
      colorAttachments: [{ view: this.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: [0.01, 0.012, 0.02, 1] }],
      depthStencilAttachment: { view: this.depth.createView(), depthClearValue: 1, depthLoadOp: 'clear', depthStoreOp: 'store' },
    });
    pass.setBindGroup(0, this.frameBG);
    pass.setPipeline(this.pipes.bg);
    pass.draw(3);

    const bindModel = () => {
      pass.setVertexBuffer(0, model.pos);
      pass.setVertexBuffer(1, model.nrm);
      pass.setVertexBuffer(2, model.uv);
      pass.setIndexBuffer(model.idx, 'uint32');
    };
    const drawGroups = (transparent: boolean) => {
      for (const g of model.groups) {
        if ((g.alpha < 0.99) !== transparent) continue;
        pass.setBindGroup(1, g.bind);
        pass.drawIndexed(g.count, 1, g.start);
      }
    };

    if (ground) {
      pass.setPipeline(this.pipes.mirror);
      bindModel();
      drawGroups(false);
      pass.setPipeline(this.pipes.floor);
      pass.draw(6);
    }
    pass.setPipeline(this.pipes.model);
    bindModel();
    drawGroups(false);

    pass.setPipeline(this.pipes.box);
    pass.draw(24);

    extra?.(pass);

    pass.setPipeline(this.pipes.modelAlpha);
    pass.setBindGroup(0, this.frameBG);
    bindModel();
    drawGroups(true);
    extraAfterModel?.(pass);
    pass.end();
  }
}
