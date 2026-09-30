import { generateMipmaps } from './mipmaps';

export interface MaterialGroup {
  name: string;
  color: [number, number, number];
  alpha: number;
  tex: number;
  spec: number;
  shin: number;
  start: number;
  count: number;
  bind: GPUBindGroup;
}

export interface Model {
  name: string;
  length: number;
  bbox: { min: number[]; max: number[] };
  vertexCount: number;
  positions: Float32Array<ArrayBuffer>; // CPU copy for the voxeliser
  indices: Uint32Array<ArrayBuffer>;
  pos: GPUBuffer;
  nrm: GPUBuffer;
  uv: GPUBuffer;
  idx: GPUBuffer;
  groups: MaterialGroup[];
}

export async function loadModel(device: GPUDevice, name: string, matLayout: GPUBindGroupLayout, base = '/models'): Promise<Model> {
  const meta = await (await fetch(`${base}/${name}.json`)).json();
  const bin = await (await fetch(`${base}/${name}.bin`)).arrayBuffer();
  const o = meta.offsets;
  const V = meta.vertexCount;
  const positions = new Float32Array(bin.slice(o.pos, o.pos + V * 12));
  const nrm = new Float32Array(bin, o.nrm, V * 3);
  const uv = new Float32Array(bin, o.uv, V * 2);
  const indices = new Uint32Array(bin.slice(o.idx, o.idx + meta.indexCount * 4));

  const mk = (data: ArrayBufferView<ArrayBuffer>, usage: number) => {
    const b = device.createBuffer({ size: Math.ceil(data.byteLength / 4) * 4, usage: usage | GPUBufferUsage.COPY_DST, label: `${name}` });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const posB = mk(positions, GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE);
  const nrmB = mk(new Float32Array(nrm), GPUBufferUsage.VERTEX);
  const uvB = mk(new Float32Array(uv), GPUBufferUsage.VERTEX);
  const idxB = mk(indices, GPUBufferUsage.INDEX);

  // textures
  const white = device.createTexture({ size: [1, 1], format: 'rgba8unorm', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: white }, new Uint8Array([255, 255, 255, 255]), { bytesPerRow: 4 }, [1, 1]);
  const textures: GPUTexture[] = await Promise.all(
    (meta.textures as string[]).map(async (file) => {
      const blob = await (await fetch(`${base}/textures/${file}`)).blob();
      const bmp = await createImageBitmap(blob, { colorSpaceConversion: 'none' });
      const w = Math.min(bmp.width, 2048);
      const h = Math.max(1, Math.round((bmp.height * w) / bmp.width));
      const levels = Math.floor(Math.log2(Math.max(w, h))) + 1;
      const t = device.createTexture({
        size: [w, h], format: 'rgba8unorm', mipLevelCount: levels,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      });
      const scaled = w === bmp.width ? bmp : await createImageBitmap(bmp, { resizeWidth: w, resizeHeight: h, resizeQuality: 'high' });
      device.queue.copyExternalImageToTexture({ source: scaled }, { texture: t }, [w, h]);
      generateMipmaps(device, t);
      return t;
    }),
  );
  const sampler = device.createSampler({ minFilter: 'linear', magFilter: 'linear', mipmapFilter: 'linear', addressModeU: 'repeat', addressModeV: 'repeat', maxAnisotropy: 8 });

  const groups: MaterialGroup[] = (meta.groups as any[]).map((g) => {
    const u = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(u, 0, new Float32Array([g.color[0], g.color[1], g.color[2], g.alpha, g.spec, g.shin, g.tex >= 0 ? 1 : 0, 0]));
    const bind = device.createBindGroup({
      layout: matLayout,
      entries: [
        { binding: 0, resource: { buffer: u } },
        { binding: 1, resource: (g.tex >= 0 ? textures[g.tex] : white).createView() },
        { binding: 2, resource: sampler },
      ],
    });
    return { name: g.name, color: g.color, alpha: g.alpha, tex: g.tex, spec: g.spec, shin: g.shin, start: g.start, count: g.count, bind };
  });
  return { name, length: meta.length, bbox: meta.bbox, vertexCount: V, positions, indices, pos: posB, nrm: nrmB, uv: uvB, idx: idxB, groups };
}
