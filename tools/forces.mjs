import { chromium } from 'playwright';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const q = process.argv[2] || 'model=f1&streams=0';
const configs = JSON.parse(process.argv[3] || '[{"rideCm":10}]');
const steps = +(process.argv[4] || 2500);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${q}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 240000 });
await page.evaluate(() => { window.__app.state.paused = true; });
for (const cfg of configs) {
  const r = await page.evaluate(async ([cfg, steps]) => {
    const sim = window.__app.sim;
    if (cfg.rideCm !== undefined) sim.rideCm = cfg.rideCm;
    if (cfg.aoa !== undefined) sim.aoaDeg = cfg.aoa;
    if (cfg.re) { sim.re = cfg.re; const nu = (0.07 * sim.lengthCells) / cfg.re; sim.tau0 = 0.5 + 3 * nu; sim.lbm.update({ tau0: sim.tau0 }); }
    await sim.applyGeometry(); sim.resetFlow();
    await sim.lbm.run(steps - 500, 50); await sim.lbm.readForces();
    const samples = [];
    for (let k = 0; k < 5; k++) { await sim.lbm.run(100, 50); const f = await sim.lbm.readForces(); samples.push(f); }
    const avg = (key) => samples.reduce((a, s) => a + s[key], 0) / samples.length;
    const q = 0.5 * 0.07 * 0.07;
    const v = sim.voxel; const P = sim.preset; const ref = P.refAreaM2 ? P.refAreaM2 / (P.dx * P.dx) : v.frontalArea;
    return { cfg, CdRef: avg('fx') / (q * ref), ClRef: avg('fy') / (q * ref), CyRef: avg('fz') / (q * ref), fx: avg('fx'), fy: avg('fy'), fz: avg('fz'), CdFront: avg('fx') / (q * v.frontalArea), ClFront: avg('fy') / (q * v.frontalArea), CdPlan: avg('fx') / (q * v.planformArea), ClPlan: avg('fy') / (q * v.planformArea), front: v.frontalArea, plan: v.planformArea, solid: v.solidCells };
  }, [cfg, steps]);
  console.log(JSON.stringify(r, (k, v) => (typeof v === 'number' ? +v.toFixed(4) : v)));
}
await browser.close();
