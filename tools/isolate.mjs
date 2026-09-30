import { chromium } from 'playwright';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?model=f1&streams=0`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
await page.evaluate(() => { window.__app.state.paused = true; });
const run = (name, maker) => page.evaluate(async ([name, maker]) => {
  const sim = window.__app.sim; const l = sim.lbm; const [nx, ny, nz] = [l.nx, l.ny, l.nz];
  l.update({ tau0: 0.55, cs: 0.12 });
  if (maker === 'carslip') { l.cfg.faces[2] = 3; l.update({}); await sim.applyGeometry(); }
  else if (maker === 'none') { l.setSolid(new Uint32Array(l.N)); }
  else if (maker === 'box') { const m = new Uint32Array(l.N); for (let z = nz/2-13; z < nz/2+13; z++) for (let y = 0; y < 14; y++) for (let x = 60; x < 132; x++) m[x + nx*(y + ny*z)] = 1; l.setSolid(m); }
  else if (maker === 'car') { await sim.applyGeometry(); }
  l.reset();
  let out = [];
  for (let k = 1; k <= 6; k++) {
    await l.run(20, 20);
    const m = await l.readMacro(); let nan = 0, umax = 0, rmax = 0, where = null;
    for (let i = 0; i < l.N; i++) { const u = Math.hypot(m[4*i], m[4*i+1], m[4*i+2]), r = m[4*i+3];
      if (!Number.isFinite(u) || !Number.isFinite(r)) { if (!where) where = [i % nx, Math.floor(i / nx) % ny, Math.floor(i / (nx*ny))]; nan++; continue; }
      umax = Math.max(umax, u); rmax = Math.max(rmax, r); }
    out.push(`${k*20}:nan=${nan} umax=${umax.toFixed(3)} rmax=${rmax.toFixed(2)}${where ? ' firstNaN@' + where : ''}`);
    if (nan) break;
  }
  return name + ' -> ' + out.join(' | ');
}, [name, maker]);
for (const t of ['car']) console.log(await run(t, t));
await browser.close();
