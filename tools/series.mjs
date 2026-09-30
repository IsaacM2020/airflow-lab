import { chromium } from 'playwright';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const q = process.argv[2] || 'model=f1&nohud=1&streams=0';
const n = +(process.argv[3] || 24);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${q}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 240000 });
await page.evaluate(() => { window.__app.state.paused = true; });
const out = await page.evaluate(async (n) => {
  const sim = window.__app.sim; const res = [];
  sim.resetFlow(); await sim.lbm.readForces();
  const qd = 0.5 * 0.07 * 0.07, A = sim.voxel.frontalArea;
  for (let k = 0; k < n; k++) {
    await sim.lbm.run(150, 50);
    const f = await sim.lbm.readForces();
    res.push([sim.lbm.totalSteps, +(f.fx / (qd * A)).toFixed(2), +(f.fy / (qd * A)).toFixed(2)]);
  }
  return res;
}, n);
console.log('steps, Cd, Cl(frontal):'); console.log(out.map((r) => r.join('\t')).join('\n'));
await browser.close();
