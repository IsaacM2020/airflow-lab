import { chromium } from 'playwright';
const q = process.argv[2] || 'model=f1';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const seen = new Set();
page.on('console', (m) => { if (['error','warning'].includes(m.type())) { const t = m.text().slice(0,500); if (!seen.has(t) && !t.includes('404') && seen.size < 6) { seen.add(t); console.log('[browser]', t);} } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${q}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 }).catch(() => console.log('not ready'));
await page.waitForTimeout(+(process.env.WAIT || 5000));
const r = await page.evaluate(async () => {
  if (window.__app.sim.lbm.totalSteps < 100) { /* ensure some steps */ }
  const app = window.__app; const sim = app.sim;
  const m = await sim.lbm.readMacro();
  let umax = 0, rmin = 9, rmax = 0, nan = 0, ux = 0, nfl = 0;
  for (let i = 0; i < sim.lbm.N; i++) { const u = Math.hypot(m[4*i], m[4*i+1], m[4*i+2]); const r = m[4*i+3];
    if (!Number.isFinite(u) || !Number.isFinite(r)) { nan++; continue; }
    if (u > umax) umax = u; if (r < rmin) rmin = r; if (r > rmax) rmax = r; }
  return { stats: app.stats, voxel: sim.voxel, steps: sim.lbm.totalSteps, umax, rmin, rmax, nan, spf: app.stepsPerFrame, tau: sim.tau0, re: sim.re };
});
console.log(JSON.stringify(r, (k, v) => (k === 'history' ? undefined : v)));
await browser.close();
