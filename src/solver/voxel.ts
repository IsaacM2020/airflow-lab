/**
 * GPU voxeliser: triangle mesh -> solid mask for the solver.
 *
 * 1. mark:  every triangle marks every grid cell it touches (exact triangle/box overlap test),
 *           giving a thin "shell". Conservative, so thin wings and panels never vanish.
 * 2. fill:  flood-fill "outside" from the domain boundary through non-shell cells.
 *           Anything not reached from outside is inside the body, so holes in a messy
 *           mesh do not matter as long as the shell has no gap wider than one cell.
 * 3. flags: solid = shell OR not-outside.
 */

export interface Mesh {
  positions: Float32Array<ArrayBuffer>; // xyz, metres, body coordinates
  indices: Uint32Array<ArrayBuffer>;
}

export interface VoxelResult {
  solidCells: number;
  /** Projected frontal area in cells^2 (looking along +x). */
  frontalArea: number;
  /** Projected planform area in cells^2 (looking along y, from above). */
  planformArea: number;
  passes: number;
}

const WGSL = /* wgsl */ `
struct VP {
  m0: vec4<f32>,   // rows of the body->cell transform (3x4)
  m1: vec4<f32>,
  m2: vec4<f32>,
  dims: vec4<u32>, // nx, ny, nz, triCount
}
@group(0) @binding(0) var<uniform> V: VP;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> idx: array<u32>;
@group(0) @binding(3) var<storage, read_write> shell: array<u32>;
@group(0) @binding(4) var<storage, read_write> outside: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> flags: array<u32>;
@group(0) @binding(6) var<storage, read_write> changed: array<atomic<u32>, 1>;

fn toCell(i: u32) -> vec3<f32> {
  let p = vec4<f32>(pos[i * 3u], pos[i * 3u + 1u], pos[i * 3u + 2u], 1.0);
  return vec3<f32>(dot(V.m0, p), dot(V.m1, p), dot(V.m2, p));
}

fn cellIdx(x: u32, y: u32, z: u32) -> u32 { return x + V.dims.x * (y + V.dims.y * z); }

fn triBox(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, h: vec3<f32>) -> bool {
  let mn = min(a, min(b, c));
  let mx = max(a, max(b, c));
  if (any(mn > h) || any(mx < -h)) { return false; }
  let e0 = b - a; let e1 = c - b; let e2 = a - c;
  let n = cross(e0, e1);
  let r = h.x * abs(n.x) + h.y * abs(n.y) + h.z * abs(n.z);
  if (abs(dot(n, a)) > r) { return false; }
  var edges = array<vec3<f32>, 3>(e0, e1, e2);
  var axes = array<vec3<f32>, 3>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      let ax = cross(axes[j], edges[i]);
      if (dot(ax, ax) < 1e-12) { continue; }
      let p0 = dot(ax, a); let p1 = dot(ax, b); let p2 = dot(ax, c);
      let rad = h.x * abs(ax.x) + h.y * abs(ax.y) + h.z * abs(ax.z);
      if (min(p0, min(p1, p2)) > rad || max(p0, max(p1, p2)) < -rad) { return false; }
    }
  }
  return true;
}

@compute @workgroup_size(64)
fn mark(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = g.x + g.y * 65535u * 64u;
  if (t >= V.dims.w) { return; }
  let a = toCell(idx[t * 3u]);
  let b = toCell(idx[t * 3u + 1u]);
  let c = toCell(idx[t * 3u + 2u]);
  let dims = vec3<i32>(vec3<u32>(V.dims.x, V.dims.y, V.dims.z));
  let lo = clamp(vec3<i32>(floor(min(a, min(b, c)))), vec3<i32>(0), dims - 1);
  let hi = clamp(vec3<i32>(floor(max(a, max(b, c)))), vec3<i32>(0), dims - 1);
  let h = vec3<f32>(0.5005);
  for (var z = lo.z; z <= hi.z; z++) {
    for (var y = lo.y; y <= hi.y; y++) {
      for (var x = lo.x; x <= hi.x; x++) {
        let centre = vec3<f32>(f32(x) + 0.5, f32(y) + 0.5, f32(z) + 0.5);
        if (triBox(a - centre, b - centre, c - centre, h)) {
          shell[cellIdx(u32(x), u32(y), u32(z))] = 1u;
        }
      }
    }
  }
}

@compute @workgroup_size(8, 8, 4)
fn seed(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= V.dims.x || g.y >= V.dims.y || g.z >= V.dims.z) { return; }
  let i = cellIdx(g.x, g.y, g.z);
  let onEdge = g.x == 0u || g.y == 0u || g.z == 0u
            || g.x == V.dims.x - 1u || g.y == V.dims.y - 1u || g.z == V.dims.z - 1u;
  atomicStore(&outside[i], select(0u, 1u, onEdge && shell[i] == 0u));
}

@compute @workgroup_size(8, 8, 4)
fn grow(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= V.dims.x || g.y >= V.dims.y || g.z >= V.dims.z) { return; }
  let i = cellIdx(g.x, g.y, g.z);
  if (shell[i] != 0u || atomicLoad(&outside[i]) != 0u) { return; }
  var hit = false;
  if (g.x > 0u)            { hit = hit || atomicLoad(&outside[i - 1u]) != 0u; }
  if (g.x + 1u < V.dims.x) { hit = hit || atomicLoad(&outside[i + 1u]) != 0u; }
  if (g.y > 0u)            { hit = hit || atomicLoad(&outside[i - V.dims.x]) != 0u; }
  if (g.y + 1u < V.dims.y) { hit = hit || atomicLoad(&outside[i + V.dims.x]) != 0u; }
  let sz = V.dims.x * V.dims.y;
  if (g.z > 0u)            { hit = hit || atomicLoad(&outside[i - sz]) != 0u; }
  if (g.z + 1u < V.dims.z) { hit = hit || atomicLoad(&outside[i + sz]) != 0u; }
  if (hit) {
    atomicStore(&outside[i], 1u);
    atomicStore(&changed[0], 1u);
  }
}

@compute @workgroup_size(8, 8, 4)
fn finish(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x >= V.dims.x || g.y >= V.dims.y || g.z >= V.dims.z) { return; }
  let i = cellIdx(g.x, g.y, g.z);
  flags[i] = select(0u, 1u, shell[i] != 0u || atomicLoad(&outside[i]) == 0u);
}
`;

export class Voxeliser {
  private pos: GPUBuffer;
  private idx: GPUBuffer;
  private uni: GPUBuffer;
  private shell: GPUBuffer | null = null;
  private outside: GPUBuffer | null = null;
  private changed: GPUBuffer;
  private changedStage: GPUBuffer;
  private pipes: Record<'mark' | 'seed' | 'grow' | 'finish', GPUComputePipeline>;
  readonly triCount: number;

  constructor(private device: GPUDevice, mesh: Mesh) {
    this.triCount = mesh.indices.length / 3;
    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST;
    this.pos = device.createBuffer({ size: mesh.positions.byteLength, usage: S | D, label: 'vox.pos' });
    this.idx = device.createBuffer({ size: mesh.indices.byteLength, usage: S | D, label: 'vox.idx' });
    device.queue.writeBuffer(this.pos, 0, mesh.positions);
    device.queue.writeBuffer(this.idx, 0, mesh.indices);
    this.uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | D });
    this.changed = device.createBuffer({ size: 16, usage: S | D | GPUBufferUsage.COPY_SRC });
    this.changedStage = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | D });
    const module = device.createShaderModule({ code: WGSL, label: 'voxel' });
    const mk = (entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    this.pipes = { mark: mk('mark'), seed: mk('seed'), grow: mk('grow'), finish: mk('finish') };
  }

  /**
   * @param matrix 3x4 row-major body(metres)->cell transform (12 numbers)
   * @param flags  destination u32 mask buffer, one entry per cell
   */
  async voxelise(dims: [number, number, number], matrix: number[], flags: GPUBuffer): Promise<VoxelResult> {
    const [nx, ny, nz] = dims;
    const N = nx * ny * nz;
    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
    if (!this.shell || this.shell.size !== N * 4) {
      this.shell?.destroy();
      this.outside?.destroy();
      this.shell = this.device.createBuffer({ size: N * 4, usage: S | D, label: 'vox.shell' });
      this.outside = this.device.createBuffer({ size: N * 4, usage: S | D, label: 'vox.outside' });
    }
    const u = new ArrayBuffer(64);
    const f = new Float32Array(u);
    f.set(matrix.slice(0, 4), 0);
    f.set(matrix.slice(4, 8), 4);
    f.set(matrix.slice(8, 12), 8);
    new Uint32Array(u).set([nx, ny, nz, this.triCount], 12);
    this.device.queue.writeBuffer(this.uni, 0, u);

    const bind = (pipe: GPUComputePipeline) =>
      this.device.createBindGroup({
        layout: pipe.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uni } },
          { binding: 1, resource: { buffer: this.pos } },
          { binding: 2, resource: { buffer: this.idx } },
          { binding: 3, resource: { buffer: this.shell! } },
          { binding: 4, resource: { buffer: this.outside! } },
          { binding: 5, resource: { buffer: flags } },
          { binding: 6, resource: { buffer: this.changed } },
        ].filter((e) => this.usesBinding(pipe, e.binding)),
      });
    const bgMark = bind(this.pipes.mark);
    const bgSeed = bind(this.pipes.seed);
    const bgGrow = bind(this.pipes.grow);
    const bgFinish = bind(this.pipes.finish);

    const cells: [number, number, number] = [Math.ceil(nx / 8), Math.ceil(ny / 8), Math.ceil(nz / 4)];
    {
      const enc = this.device.createCommandEncoder();
      enc.clearBuffer(this.shell);
      const p = enc.beginComputePass();
      p.setPipeline(this.pipes.mark);
      p.setBindGroup(0, bgMark);
      const groups = Math.ceil(this.triCount / 64);
      p.dispatchWorkgroups(Math.min(groups, 65535), Math.ceil(groups / 65535));
      p.setPipeline(this.pipes.seed);
      p.setBindGroup(0, bgSeed);
      p.dispatchWorkgroups(...cells);
      p.end();
      this.device.queue.submit([enc.finish()]);
    }
    let passes = 0;
    for (;;) {
      const enc = this.device.createCommandEncoder();
      enc.clearBuffer(this.changed);
      const p = enc.beginComputePass();
      p.setPipeline(this.pipes.grow);
      p.setBindGroup(0, bgGrow);
      for (let i = 0; i < 48; i++) p.dispatchWorkgroups(...cells);
      p.end();
      enc.copyBufferToBuffer(this.changed, 0, this.changedStage, 0, 16);
      this.device.queue.submit([enc.finish()]);
      passes += 48;
      await this.changedStage.mapAsync(GPUMapMode.READ);
      const any = new Uint32Array(this.changedStage.getMappedRange().slice(0))[0];
      this.changedStage.unmap();
      if (!any || passes > 4000) break;
    }
    {
      const enc = this.device.createCommandEncoder();
      const p = enc.beginComputePass();
      p.setPipeline(this.pipes.finish);
      p.setBindGroup(0, bgFinish);
      p.dispatchWorkgroups(...cells);
      p.end();
      const stage = this.device.createBuffer({ size: N * 4, usage: GPUBufferUsage.MAP_READ | D });
      enc.copyBufferToBuffer(flags, 0, stage, 0, N * 4);
      this.device.queue.submit([enc.finish()]);
      await stage.mapAsync(GPUMapMode.READ);
      const mask = new Uint32Array(stage.getMappedRange());
      let solid = 0;
      const cols = new Uint8Array(ny * nz);
      const colsY = new Uint8Array(nx * nz);
      for (let z = 0; z < nz; z++)
        for (let y = 0; y < ny; y++) {
          const row = nx * (y + ny * z);
          for (let x = 0; x < nx; x++)
            if (mask[row + x]) {
              solid++;
              cols[y + ny * z] = 1;
              colsY[x + nx * z] = 1;
            }
        }
      let area = 0;
      for (let i = 0; i < cols.length; i++) area += cols[i];
      let plan = 0;
      for (let i = 0; i < colsY.length; i++) plan += colsY[i];
      stage.unmap();
      stage.destroy();
      return { solidCells: solid, frontalArea: area, planformArea: plan, passes };
    }
  }

  // With layout: 'auto', a pipeline only has the bindings its entry point actually uses.
  private used = new Map<GPUComputePipeline, Set<number>>();
  private usesBinding(pipe: GPUComputePipeline, binding: number): boolean {
    if (!this.used.size) {
      const map: [GPUComputePipeline, number[]][] = [
        [this.pipes.mark, [0, 1, 2, 3]],
        [this.pipes.seed, [0, 3, 4]],
        [this.pipes.grow, [0, 3, 4, 6]],
        [this.pipes.finish, [0, 3, 4, 5]],
      ];
      for (const [p, b] of map) this.used.set(p, new Set(b));
    }
    return this.used.get(pipe)!.has(binding);
  }
}

/**
 * Build a body->cell 3x4 row-major matrix: yaw about y, then pitch about z, then scale to cells and translate.
 * pitch: radians, rotation about +z (nose up is negative because the nose points to -x).
 * yaw: radians, rotation about +y.
 */
export function bodyToCell(opts: {
  pitch: number;
  yaw?: number;
  pivot: [number, number, number]; // body-space rotation centre, metres
  dx: number; // metres per cell
  place: [number, number, number]; // where the pivot lands, in cells
}): number[] {
  const cp = Math.cos(opts.pitch), sp = Math.sin(opts.pitch);
  const cy = Math.cos(opts.yaw ?? 0), sy = Math.sin(opts.yaw ?? 0);
  const k = 1 / opts.dx;
  // R = Rz(pitch) * Ry(yaw)
  const R = [
    [cp * cy, -sp, cp * sy],
    [sp * cy, cp, sp * sy],
    [-sy, 0, cy],
  ];
  const [px, py, pz] = opts.pivot;
  const out: number[] = [];
  for (let i = 0; i < 3; i++) {
    const r = R[i];
    out.push(r[0] * k, r[1] * k, r[2] * k, opts.place[i] - (r[0] * px + r[1] * py + r[2] * pz) * k);
  }
  return out;
}
