import { Face, LBM, requestGPU } from '../solver/lbm';

export async function bench(nx = 256, ny = 96, nz = 96, steps = 300, withBody = true) {
  const device = await requestGPU();
  const lbm = new LBM(device, {
    nx, ny, nz, tau0: 0.503, cs: 0.1,
    faces: [Face.Inlet, Face.Outlet, Face.Wall, Face.Slip, Face.Slip, Face.Slip],
    uIn: [0.08, 0, 0], uWall: [0.08, 0, 0],
  });
  if (withBody) {
    const mask = new Uint32Array(lbm.N);
    for (let z = nz / 2 - 13; z < nz / 2 + 13; z++)
      for (let y = 0; y < 14; y++)
        for (let x = 60; x < 132; x++) mask[x + nx * (y + ny * z)] = 1;
    lbm.setSolid(mask);
    lbm.reset();
  }
  await lbm.run(40, 20); // warm up
  const t0 = performance.now();
  await lbm.run(steps, 100);
  const dt = (performance.now() - t0) / 1000;
  const f = await lbm.readForces();
  const m = await lbm.readMacro();
  let bad = 0, umax = 0;
  for (let i = 0; i < lbm.N; i++) {
    const v = Math.hypot(m[4 * i], m[4 * i + 1], m[4 * i + 2]);
    if (!Number.isFinite(v)) bad++; else umax = Math.max(umax, v);
  }
  const r = {
    cells: lbm.N, steps, seconds: +dt.toFixed(3), stepsPerSec: +(steps / dt).toFixed(1),
    MLUPS: +((lbm.N * steps) / dt / 1e6).toFixed(0), nanCells: bad, umax: +umax.toFixed(4), fx: f.fx,
  };
  lbm.destroy();
  return r;
}
(window as unknown as { __bench: typeof bench }).__bench = bench;
