import { chromium } from 'playwright';
const q = process.argv[2] || 'model=f1&streams=0';
const combos = JSON.parse(process.argv[3] || '[[0.52,0.12],[0.51,0.12],[0.51,0.17],[0.5058,0.17]]');
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${q}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
await page.evaluate(() => { window.__app.state.paused = true; });
for (const [tau, cs] of combos) {
  const r = await page.evaluate(async ([tau, cs]) => {
    const sim = window.__app.sim;
    sim.lbm.update({ tau0: tau, cs }); sim.resetFlow();
    let firstBad = -1, last = null;
    for (let k = 1; k <= 24; k++) {
      await sim.lbm.run(100, 50);
      const m = await sim.lbm.readMacro();
      let nan = 0, umax = 0, rmin = 9, rmax = 0;
      for (let i = 0; i < sim.lbm.N; i++) { const u = Math.hypot(m[4*i], m[4*i+1], m[4*i+2]), r = m[4*i+3];
        if (!Number.isFinite(u) || !Number.isFinite(r)) { nan++; continue; } umax = Math.max(umax, u); rmin = Math.min(rmin, r); rmax = Math.max(rmax, r); }
      last = { steps: k * 100, nan, umax: +umax.toFixed(4), rmin: +rmin.toFixed(3), rmax: +rmax.toFixed(3) };
      if (nan > 0 || umax > 0.4) { firstBad = k * 100; break; }
    }
    return { tau, cs, firstBad, last };
  }, [tau, cs]);
  console.log(JSON.stringify(r));
}
await browser.close();
