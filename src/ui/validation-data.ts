/**
 * Measured validation results, copied from headless runs of src/validation/validate.ts on the M4 GPU
 * (node tools/run-validation.mjs). Reported as measured, including the parts that are not perfect.
 */
export const ANALYTIC = [
  {
    name: 'Taylor–Green vortex',
    what: 'Kinetic energy decay rate, exact solution E(t) ∝ exp(−4νk²t)',
    measured: '3.8567 × 10⁻³',
    reference: '3.8553 × 10⁻³',
    error: '0.04%',
    pass: true,
  },
  {
    name: 'Poiseuille channel flow (τ = 1.0)',
    what: 'Steady velocity profile between two walls, exact parabola',
    measured: 'max deviation 0.03% of peak',
    reference: 'u(y) = g/(2ν)·(y+½)(H−½−y)',
    error: '0.03%',
    pass: true,
  },
  {
    name: 'Poiseuille channel flow (τ = 0.7)',
    what: 'Same, at a different viscosity (wall position error grows away from τ = 1)',
    measured: 'max deviation 0.17% of peak',
    reference: 'exact parabola',
    error: '0.17%',
    pass: true,
  },
];

/** Sphere drag vs the Schiller–Naumann correlation Cd = 24/Re (1 + 0.15 Re^0.687), valid for Re < ~800. */
export const SPHERE = [
  { re: 10, cd: 4.729, ref: 4.151, note: 'D = 20, walls 5D' },
  { re: 50, cd: 1.727, ref: 1.538, note: 'D = 20, walls 5D' },
  { re: 100, cd: 1.201, ref: 1.092, note: 'D = 20, walls 5D' },
  { re: 200, cd: 0.867, ref: 0.806, note: 'D = 20, walls 5D' },
];

/** Re = 50 confinement / resolution study. Published (Schiller–Naumann) value 1.538. */
export const SPHERE_STUDY = [
  { label: 'D = 20, walls 5D, inlet 3D upstream (baseline)', cd: 1.727 },
  { label: 'D = 20, walls 8D (wider tunnel)', cd: 1.691 },
  { label: 'D = 32, walls 5D (finer sphere)', cd: 1.687 },
  { label: 'D = 20, walls 8D, inlet 8D upstream (open domain)', cd: 1.654 },
];

export const schillerNaumann = (re: number) => (24 / re) * (1 + 0.15 * Math.pow(re, 0.687));
