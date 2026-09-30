// Flatten GLB models (glTF 1.0 or 2.0) into one clean world-space mesh in Airflow Lab axes:
//   x = streamwise (nose toward -x, wind blows toward +x), y = up, z = lateral. Units: metres.
// Output: public/models/<name>.bin (pos f32x3, nrm f32x3, uv f32x2, idx u32) + <name>.json + textures.
// Usage: node tools/prepare-models.mjs <in.glb> <name> <realLengthMetres> <sourceUpAxis: y|z> [--ground] [--length-axis=x|z]
import fs from 'node:fs';
import path from 'node:path';

const [, , inFile, name, lengthArg, upAxis = 'y', ...flags] = process.argv;
const realLength = parseFloat(lengthArg);
const ground = flags.includes('--ground');
const outDir = path.resolve('public/models');
fs.mkdirSync(path.join(outDir, 'textures'), { recursive: true });

const buf = fs.readFileSync(inFile);
const version = buf.readUInt32LE(4);
const jsonLen = buf.readUInt32LE(12);
const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
const binStart = version === 1 ? 20 + jsonLen : 20 + jsonLen + 8;
const bin = buf.subarray(binStart);
const isV1 = version === 1;

// ---- helpers: glTF 1.0 uses dicts keyed by id, 2.0 uses arrays keyed by index
const get = (coll, k) => (Array.isArray(coll) ? coll[k] : coll[k]);
const values = (coll) => (Array.isArray(coll) ? coll.map((v, i) => [i, v]) : Object.entries(coll));

const TYPE_N = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const CT = { 5120: [1, 'getInt8'], 5121: [1, 'getUint8'], 5122: [2, 'getInt16'], 5123: [2, 'getUint16'], 5125: [4, 'getUint32'], 5126: [4, 'getFloat32'] };

function readAccessor(id) {
  const a = get(json.accessors, id);
  const bv = get(json.bufferViews, a.bufferView);
  const n = TYPE_N[a.type];
  const [size, fn] = CT[a.componentType];
  const stride = a.byteStride || bv.byteStride || n * size;
  const dv = new DataView(bin.buffer, bin.byteOffset + (bv.byteOffset || 0) + (a.byteOffset || 0));
  const out = a.componentType === 5126 ? new Float32Array(a.count * n) : new Uint32Array(a.count * n);
  for (let i = 0; i < a.count; i++)
    for (let j = 0; j < n; j++) out[i * n + j] = dv[fn](i * stride + j * size, true);
  return out;
}

// ---- 4x4 matrices, column-major
const ident = () => [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
function mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
function trs(t = [0, 0, 0], q = [0, 0, 0, 1], s = [1, 1, 1]) {
  const [x, y, z, w] = q;
  const r = [
    1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
    2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
    2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
    0, 0, 0, 1,
  ];
  r[0] *= s[0]; r[1] *= s[0]; r[2] *= s[0];
  r[4] *= s[1]; r[5] *= s[1]; r[6] *= s[1];
  r[8] *= s[2]; r[9] *= s[2]; r[10] *= s[2];
  r[12] = t[0]; r[13] = t[1]; r[14] = t[2];
  return r;
}
const nodeMatrix = (n) => n.matrix || trs(n.translation, n.rotation, n.scale);
const xform = (m, [x, y, z]) => [
  m[0] * x + m[4] * y + m[8] * z + m[12],
  m[1] * x + m[5] * y + m[9] * z + m[13],
  m[2] * x + m[6] * y + m[10] * z + m[14],
];
const xformDir = (m, [x, y, z]) => [
  m[0] * x + m[4] * y + m[8] * z,
  m[1] * x + m[5] * y + m[9] * z,
  m[2] * x + m[6] * y + m[10] * z,
];

// ---- materials and textures
const textures = []; // output file names
const texIndexCache = new Map();
function exportTexture(texId) {
  if (texIndexCache.has(texId)) return texIndexCache.get(texId);
  const tex = get(json.textures, texId);
  const img = get(json.images, tex.source);
  let bv, mime;
  if (isV1) {
    const ext = img.extensions?.KHR_binary_glTF;
    bv = get(json.bufferViews, ext.bufferView);
    mime = ext.mimeType;
  } else {
    bv = get(json.bufferViews, img.bufferView);
    mime = img.mimeType;
  }
  const ext = mime.includes('png') ? 'png' : 'jpg';
  const file = `${name}_tex${textures.length}.${ext}`;
  fs.writeFileSync(path.join(outDir, 'textures', file), bin.subarray(bv.byteOffset, bv.byteOffset + bv.byteLength));
  textures.push(file);
  texIndexCache.set(texId, textures.length - 1);
  return textures.length - 1;
}
function materialInfo(matId) {
  const m = matId == null ? null : get(json.materials, matId);
  const info = { name: m?.name || 'default', color: [0.7, 0.7, 0.72], alpha: 1, tex: -1, spec: 0.3, shin: 40 };
  if (!m) return info;
  if (isV1) {
    const v = m.values || {};
    if (typeof v.diffuse === 'string') {
      // texture reference by name in the technique, resolve through the "textures" dict
      const texId = Object.keys(json.textures).find((k) => v.diffuse === k) ?? v.diffuse;
      if (json.textures[texId]) info.tex = exportTexture(texId);
    } else if (Array.isArray(v.diffuse)) info.color = v.diffuse.slice(0, 3);
    if (typeof v.transparency === 'number') info.alpha = v.transparency;
    else if (Array.isArray(v.diffuse) && v.diffuse[3] != null) info.alpha = v.diffuse[3];
    if (Array.isArray(v.specular)) info.spec = v.specular[0];
    if (typeof v.shininess === 'number') info.shin = v.shininess;
  } else {
    const p = m.pbrMetallicRoughness || {};
    if (p.baseColorFactor) { info.color = p.baseColorFactor.slice(0, 3); info.alpha = p.baseColorFactor[3]; }
    if (p.baseColorTexture) info.tex = exportTexture(p.baseColorTexture.index);
    info.spec = 0.4;
    info.shin = 60;
    if (m.alphaMode === 'BLEND') info.alpha = info.alpha ?? 0.5;
  }
  return info;
}

// ---- walk the scene graph
const P = [], N = [], UV = [];
const groups = new Map(); // material key -> { info, idx: [] }
function addPrimitive(prim, M) {
  if ((prim.mode ?? 4) !== 4) return;
  const pos = readAccessor(prim.attributes.POSITION);
  const hasN = prim.attributes.NORMAL != null;
  const nrm = hasN ? readAccessor(prim.attributes.NORMAL) : null;
  const hasUV = prim.attributes.TEXCOORD_0 != null;
  const uv = hasUV ? readAccessor(prim.attributes.TEXCOORD_0) : null;
  const idx = prim.indices != null ? readAccessor(prim.indices) : Array.from({ length: pos.length / 3 }, (_, i) => i);
  const base = P.length / 3;
  for (let i = 0; i < pos.length / 3; i++) {
    const p = xform(M, [pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]]);
    P.push(...p);
    const nn = hasN ? xformDir(M, [nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2]]) : [0, 1, 0];
    const l = Math.hypot(...nn) || 1;
    N.push(nn[0] / l, nn[1] / l, nn[2] / l);
    UV.push(hasUV ? uv[i * 2] : 0, hasUV ? uv[i * 2 + 1] : 0);
  }
  const key = String(prim.material ?? 'none');
  if (!groups.has(key)) groups.set(key, { info: materialInfo(prim.material), idx: [] });
  const g = groups.get(key);
  // flip winding if the transform mirrors
  const det = M[0] * (M[5] * M[10] - M[6] * M[9]) - M[4] * (M[1] * M[10] - M[2] * M[9]) + M[8] * (M[1] * M[6] - M[2] * M[5]);
  for (let i = 0; i < idx.length; i += 3) {
    if (det < 0) g.idx.push(base + idx[i], base + idx[i + 2], base + idx[i + 1]);
    else g.idx.push(base + idx[i], base + idx[i + 1], base + idx[i + 2]);
  }
}
function walk(nodeId, parent) {
  const n = get(json.nodes, nodeId);
  const M = mul(parent, nodeMatrix(n));
  const meshIds = isV1 ? n.meshes || [] : n.mesh != null ? [n.mesh] : [];
  for (const mid of meshIds) for (const prim of get(json.meshes, mid).primitives) addPrimitive(prim, M);
  for (const c of n.children || []) walk(c, M);
}
let roots;
if (isV1) roots = json.scenes[json.scene].nodes;
else roots = json.scenes[json.scene ?? 0].nodes;
for (const r of roots) walk(r, ident());

// ---- conversion into Airflow Lab axes. First find the model's own length axis.
// Source axes: FlightGear (A380) is x back, y right, z up; glTF is x, y up, z.
// After building "up-axis normalised" coordinates (X, Y up, Z), find the longest horizontal extent.
const count = P.length / 3;
{
  const mn0 = [1e9, 1e9, 1e9], mx0 = [-1e9, -1e9, -1e9];
  for (let i = 0; i < count; i++) for (let a = 0; a < 3; a++) { mn0[a] = Math.min(mn0[a], P[i * 3 + a]); mx0[a] = Math.max(mx0[a], P[i * 3 + a]); }
  console.log('  raw world extents after node transforms (x,y,z):', mn0.map((v, a) => `${v.toFixed(1)}..${mx0[a].toFixed(1)}`).join('  '));
}
const upIsZ = upAxis === 'z';
const conv = (p) => (upIsZ ? [p[0], p[2], -p[1]] : [p[0], p[1], p[2]]); // proper rotation z-up -> y-up
const V = [];
for (let i = 0; i < count; i++) V.push(conv([P[i * 3], P[i * 3 + 1], P[i * 3 + 2]]));
const convN = [];
for (let i = 0; i < count; i++) convN.push(conv([N[i * 3], N[i * 3 + 1], N[i * 3 + 2]]));

const bb = (arr) => {
  const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (const p of arr) for (let a = 0; a < 3; a++) { mn[a] = Math.min(mn[a], p[a]); mx[a] = Math.max(mx[a], p[a]); }
  return { mn, mx };
};
let { mn, mx } = bb(V);
const ext = [mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]];
// length axis = longest horizontal axis (x or z). If z, rotate 90 degrees about y so it becomes x.
let rot = false;
const lenFlag = flags.find((f) => f.startsWith('--length-axis='));
if (lenFlag) rot = lenFlag.endsWith('z');
else if (ext[2] > ext[0]) rot = true;
const R = (p) => (rot ? [p[2], p[1], -p[0]] : p);
for (let i = 0; i < count; i++) { V[i] = R(V[i]); convN[i] = R(convN[i]); }
({ mn, mx } = bb(V));

// which end is the nose? Rear has taller structure (wing/engine cover, tail fin). Nose must be at -x.
const len = mx[0] - mn[0];
let hFront = 0, hRear = 0;
for (const p of V) {
  if (p[0] < mn[0] + 0.15 * len) hFront = Math.max(hFront, p[1] - mn[1]);
  if (p[0] > mx[0] - 0.15 * len) hRear = Math.max(hRear, p[1] - mn[1]);
}
let flipX = hFront > hRear; // taller end is the rear -> it must sit at +x
if (flipX) {
  for (let i = 0; i < count; i++) {
    V[i] = [-V[i][0], V[i][1], -V[i][2]]; // 180 degree turn about y keeps handedness
    convN[i] = [-convN[i][0], convN[i][1], -convN[i][2]];
  }
  ({ mn, mx } = bb(V));
}

// scale to real length, centre in x/z, put the lowest point at y = 0 (ground) or centre y
const s = realLength / (mx[0] - mn[0]);
const cx = (mn[0] + mx[0]) / 2, cz = (mn[2] + mx[2]) / 2;
const cy = ground ? mn[1] : (mn[1] + mx[1]) / 2;
const outP = new Float32Array(count * 3), outN = new Float32Array(count * 3), outUV = new Float32Array(UV);
for (let i = 0; i < count; i++) {
  outP[i * 3] = (V[i][0] - cx) * s;
  outP[i * 3 + 1] = (V[i][1] - cy) * s;
  outP[i * 3 + 2] = (V[i][2] - cz) * s;
  outN[i * 3] = convN[i][0]; outN[i * 3 + 1] = convN[i][1]; outN[i * 3 + 2] = convN[i][2];
}
const fin = bb(Array.from({ length: count }, (_, i) => [outP[i * 3], outP[i * 3 + 1], outP[i * 3 + 2]]));

// group ordering: opaque first, transparent last
const glist = [...groups.values()].sort((a, b) => (b.info.alpha >= 0.99) - (a.info.alpha >= 0.99));
const allIdx = [];
const meta = [];
for (const g of glist) {
  meta.push({ ...g.info, start: allIdx.length, count: g.idx.length });
  for (let i = 0; i < g.idx.length; i++) allIdx.push(g.idx[i]);
}
const idxArr = new Uint32Array(allIdx);
const header = {
  name, source: path.basename(inFile), vertexCount: count, indexCount: idxArr.length,
  triangles: idxArr.length / 3, length: realLength, scale: s, flippedNose: flipX, rotatedAxis: rot,
  bbox: { min: fin.mn, max: fin.mx }, groups: meta, textures,
  offsets: { pos: 0, nrm: outP.byteLength, uv: outP.byteLength * 2, idx: outP.byteLength * 2 + outUV.byteLength },
};
const out = Buffer.concat([Buffer.from(outP.buffer), Buffer.from(outN.buffer), Buffer.from(outUV.buffer), Buffer.from(idxArr.buffer)]);
fs.writeFileSync(path.join(outDir, `${name}.bin`), out);
fs.writeFileSync(path.join(outDir, `${name}.json`), JSON.stringify(header));
const f = (a) => a.map((x) => x.toFixed(2)).join(', ');
console.log(`${name}: ${count} verts, ${idxArr.length / 3} tris, ${meta.length} groups, ${textures.length} textures`);
console.log(`  bbox min [${f(fin.mn)}]  max [${f(fin.mx)}]  (x streamwise, y up, z lateral) metres`);
console.log(`  rotated length axis: ${rot}, nose flipped: ${flipX}, scale ${s.toFixed(5)}, bin ${(out.length / 1e6).toFixed(1)} MB`);
