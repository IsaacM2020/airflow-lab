/** Tiny mip-chain generator (box filter through render passes). */
const SRC = /* wgsl */ `
@group(0) @binding(0) var s: sampler;
@group(0) @binding(1) var t: texture_2d<f32>;
struct O { @builtin(position) p: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> O {
  var pos = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
  var o: O; let p = pos[i];
  o.p = vec4(p, 0.0, 1.0); o.uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); return o;
}
@fragment fn fs(i: O) -> @location(0) vec4<f32> { return textureSample(t, s, i.uv); }
`;
const cache = new WeakMap<GPUDevice, Map<GPUTextureFormat, GPURenderPipeline>>();

export function generateMipmaps(device: GPUDevice, tex: GPUTexture) {
  let byFmt = cache.get(device);
  if (!byFmt) { byFmt = new Map(); cache.set(device, byFmt); }
  let pipe = byFmt.get(tex.format);
  const module = device.createShaderModule({ code: SRC });
  if (!pipe) {
    pipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: tex.format }] },
      primitive: { topology: 'triangle-list' },
    });
    byFmt.set(tex.format, pipe);
  }
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear' });
  const enc = device.createCommandEncoder();
  for (let level = 1; level < tex.mipLevelCount; level++) {
    const bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: sampler },
        { binding: 1, resource: tex.createView({ baseMipLevel: level - 1, mipLevelCount: 1 }) },
      ],
    });
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: tex.createView({ baseMipLevel: level, mipLevelCount: 1 }),
        loadOp: 'clear', storeOp: 'store', clearValue: [0, 0, 0, 0],
      }],
    });
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bg);
    pass.draw(3);
    pass.end();
  }
  device.queue.submit([enc.finish()]);
}
