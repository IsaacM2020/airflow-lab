/**
 * Solver validation. Each test compares the GPU solver against a known analytic
 * result or a published correlation. Results are reported as measured.
 *
 * Exposed on window.__validate(name?) so headless Chrome can run them, and shown
 * on the validate.html page for humans.
 */
import { Face, LBM, feqCPU, requestGPU, type Faces } from '../solver/lbm';

export interface TestResult {
  name: string;
  pass: boolean;
  summary: string;
  data?: Record<string, unknown>;
}

const out = document.getElementById('out');
const log = (s: string) => {
  if (out) out.textContent += '\n' + s;
};

/** Build a full population field from macroscopic fields (SoA, length 19N). */
function populationsFrom(
  N: number,
  field: (i: number) => { rho: number; u: [number, number, number] },
): Float32Array {
  const f = new Float32Array(19 * N);
  for (let i = 0; i < N; i++) {
    const { rho, u } = field(i);
    for (let q = 0; q < 19; q++) f[q * N + i] = feqCPU(q, rho, u[0], u[1], u[2]);
  }
  return f;
}

// ---------------------------------------------------------------------------
// 1. Taylor-Green vortex: kinetic energy decays as exp(-4 nu k^2 t). Checks that the
//    lattice viscosity relation nu = (tau - 1/2)/3 really gives the viscosity of NS.
// ---------------------------------------------------------------------------
async function taylorGreen(device: GPUDevice): Promise<TestResult> {
  const n = 64, nz = 4;
  const tau0 = 0.8;
  const nu = (tau0 - 0.5) / 3;
  const U0 = 0.04;
  const k = (2 * Math.PI) / n;
  const faces: Faces = [0, 0, 0, 0, 0, 0];
  const lbm = new LBM(device, { nx: n, ny: n, nz, tau0, cs: 0, faces });
  const N = lbm.N;
  const f = populationsFrom(N, (i) => {
    const x = i % n, y = Math.floor(i / n) % n;
    const X = x + 0.5, Y = y + 0.5;
    const u: [number, number, number] = [
      U0 * Math.sin(k * X) * Math.cos(k * Y),
      -U0 * Math.cos(k * X) * Math.sin(k * Y),
      0,
    ];
    // pressure p = -(rho0 U0^2 / 4)(cos 2kx + cos 2ky), rho = 1 + 3p
    const p = -(U0 * U0) / 4 * (Math.cos(2 * k * X) + Math.cos(2 * k * Y));
    return { rho: 1 + 3 * p, u };
  });
  lbm.setPopulations(f);

  const energy = (m: Float32Array) => {
    let e = 0;
    for (let i = 0; i < N; i++) e += m[4 * i] ** 2 + m[4 * i + 1] ** 2;
    return e / N;
  };
  const samples: [number, number][] = [];
  const total = 1200, every = 100;
  lbm.step(1);
  samples.push([1, energy(await lbm.readMacro())]);
  for (let t = every; t <= total; t += every) {
    await lbm.run(every - (t === every ? 1 : 0));
    samples.push([lbm.totalSteps, energy(await lbm.readMacro())]);
  }
  // least squares slope of ln E vs t
  const xs = samples.map((s) => s[0]), ys = samples.map((s) => Math.log(s[1]));
  const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
  const slope =
    xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  const measured = -slope;
  const analytic = 4 * nu * k * k;
  const err = Math.abs(measured - analytic) / analytic;
  lbm.destroy();
  return {
    name: 'Taylor-Green vortex decay',
    pass: err < 0.02,
    summary: `energy decay rate: measured ${measured.toExponential(4)}, analytic 4νk² = ${analytic.toExponential(4)}, error ${(err * 100).toFixed(2)}%`,
    data: { measured, analytic, err, nu, tau0 },
  };
}

// ---------------------------------------------------------------------------
// 2. Poiseuille channel flow: steady u(y) = g/(2 nu) (y + 1/2)(H - 1/2 - y)
//    with walls half a cell outside the first and last fluid cell.
// ---------------------------------------------------------------------------
async function poiseuille(device: GPUDevice, tau0: number): Promise<TestResult> {
  const nx = 16, ny = 32, nz = 4;
  const nu = (tau0 - 0.5) / 3;
  const g = 2e-5;
  const faces: Faces = [Face.Periodic, Face.Periodic, Face.Wall, Face.Wall, Face.Periodic, Face.Periodic];
  const lbm = new LBM(device, { nx, ny, nz, tau0, cs: 0, faces, gravity: [g, 0, 0] });
  const steps = Math.ceil((ny * ny) / nu) * 3; // several viscous diffusion times
  await lbm.run(steps);
  const m = await lbm.readMacro();
  let maxErr = 0, maxU = 0;
  const profile: { y: number; sim: number; exact: number }[] = [];
  for (let y = 0; y < ny; y++) {
    const i = 8 + nx * (y + ny * 2);
    const sim = m[4 * i];
    const exact = (g / (2 * nu)) * (y + 0.5) * (ny - 0.5 - y);
    profile.push({ y, sim, exact });
    maxU = Math.max(maxU, exact);
    maxErr = Math.max(maxErr, Math.abs(sim - exact));
  }
  const rel = maxErr / maxU;
  lbm.destroy();
  return {
    name: `Poiseuille channel flow (tau=${tau0})`,
    pass: rel < 0.02,
    summary: `max deviation from analytic parabola: ${(rel * 100).toFixed(2)}% of peak velocity (peak ${maxU.toExponential(3)})`,
    data: { rel, tau0, profile },
  };
}

// ---------------------------------------------------------------------------
// 3. Flow past a sphere: drag coefficient vs the Schiller-Naumann correlation
//    Cd = 24/Re (1 + 0.15 Re^0.687), good for Re < ~800.
// ---------------------------------------------------------------------------
export function schillerNaumann(re: number) {
  return (24 / re) * (1 + 0.15 * Math.pow(re, 0.687));
}

async function sphere(device: GPUDevice, re: number, opts: { D?: number; U?: number; W?: number; up?: number } = {}): Promise<TestResult> {
  const D = opts.D ?? 20;
  const U = opts.U ?? 0.06;
  const W5 = opts.W ?? 5;
  const UP = opts.up ?? 3;
  const nx = D * (UP + 9), ny = Math.round(D * W5), nz = Math.round(D * W5);
  const nu = (U * D) / re;
  const tau0 = 3 * nu + 0.5;
  const faces: Faces = [Face.Inlet, Face.Outlet, Face.Slip, Face.Slip, Face.Slip, Face.Slip];
  const cs = re >= 400 ? 0.1 : 0;
  const lbm = new LBM(device, { nx, ny, nz, tau0, cs, faces, uIn: [U, 0, 0] });
  // voxelised sphere, centre at 3D from inlet
  const mask = new Uint32Array(lbm.N);
  const cx = UP * D, cy = ny / 2, cz = nz / 2, R = D / 2;
  let solid = 0;
  for (let z = 0; z < nz; z++)
    for (let y = 0; y < ny; y++)
      for (let x = 0; x < nx; x++) {
        const d2 = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2 + (z + 0.5 - cz) ** 2;
        if (d2 <= R * R) {
          mask[x + nx * (y + ny * z)] = 1;
          solid++;
        }
      }
  lbm.setSolid(mask);
  lbm.reset();
  // effective diameter from the actual voxel count, and projected area from the mask
  let area = 0;
  for (let y = 0; y < ny; y++)
    for (let z = 0; z < nz; z++) {
      for (let x = 0; x < nx; x++) if (mask[x + nx * (y + ny * z)]) { area++; break; }
    }
  const flowThrough = nx / U;
  const spin = Math.ceil(flowThrough * 1.2);
  await lbm.run(spin);
  await lbm.readForces();
  const measure = Math.ceil(flowThrough * 0.4);
  const cds: number[] = [];
  const chunk = Math.ceil(measure / 8);
  for (let i = 0; i < 8; i++) {
    await lbm.run(chunk);
    const f = await lbm.readForces();
    cds.push(f.fx / (0.5 * 1.0 * U * U * area));
  }
  const cd = cds.reduce((a, b) => a + b) / cds.length;
  const ref = schillerNaumann(re);
  const err = (cd - ref) / ref;
  lbm.destroy();
  return {
    name: `Sphere drag Re=${re} (D=${D}, width=${W5}D, inlet ${UP}D upstream)`,
    pass: Math.abs(err) < 0.1,
    summary: `Cd measured ${cd.toFixed(3)}, Schiller-Naumann ${ref.toFixed(3)}, error ${(err * 100).toFixed(1)}%  (tau ${tau0.toFixed(4)}, ${solid} solid cells, area ${area})`,
    data: { re, cd, ref, err, tau0, area, cds },
  };
}

export async function runValidation(which?: string): Promise<TestResult[]> {
  const device = await requestGPU();
  const results: TestResult[] = [];
  const all: [string, () => Promise<TestResult>][] = [
    ['taylor', () => taylorGreen(device)],
    ['poiseuille1', () => poiseuille(device, 1.0)],
    ['poiseuille2', () => poiseuille(device, 0.7)],
    ['sphere10', () => sphere(device, 10)],
    ['sphere50', () => sphere(device, 50)],
    ['sphere100', () => sphere(device, 100)],
    ['sphere200', () => sphere(device, 200)],
    ['s50_w8', () => sphere(device, 50, { W: 8 })],
    ['s50_D32', () => sphere(device, 50, { D: 32 })],
    ['s10_w8', () => sphere(device, 10, { W: 8 })],
    ['s100_w8', () => sphere(device, 100, { W: 8 })],
    ['s50_open', () => sphere(device, 50, { W: 8, up: 8 })],
    ['s10_open', () => sphere(device, 10, { W: 8, up: 8 })],
  ];
  for (const [key, fn] of all) {
    if (which && which !== 'all' && !which.split(',').includes(key)) continue;
    const t0 = performance.now();
    try {
      const r = await fn();
      results.push(r);
      log(`${r.pass ? 'PASS' : 'FAIL'}  ${r.name}: ${r.summary}  [${((performance.now() - t0) / 1000).toFixed(1)}s]`);
    } catch (e) {
      const r: TestResult = { name: key, pass: false, summary: 'threw: ' + (e as Error).message };
      results.push(r);
      log(`FAIL  ${key}: ${r.summary}`);
    }
  }
  return results;
}

(window as unknown as { __validate: typeof runValidation }).__validate = runValidation;
