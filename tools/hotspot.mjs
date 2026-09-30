import { chromium } from 'playwright';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const steps = +(process.argv[2] || 1200);
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${process.env.Q || 'model=f1&streams=0'}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });
await page.evaluate(() => { window.__app.state.paused = true; });
const r = await page.evaluate(async (steps) => {
  const sim = window.__app.sim, l = sim.lbm; const { nx, ny, nz } = l;
  l.update({ tau0: 0.52, cs: 0.12 }); sim.resetFlow();
  await l.run(steps, 50);
  const m = await l.readMacro();
  // read flags
  const st = sim.device.createBuffer({ size: l.N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  const enc = sim.device.createCommandEncoder(); enc.copyBufferToBuffer(l.flags, 0, st, 0, l.N * 4); sim.device.queue.submit([enc.finish()]);
  await st.mapAsync(GPUMapMode.READ); const fl = new Uint32Array(st.getMappedRange().slice(0)); st.unmap();
  const top = [];
  for (let i = 0; i < l.N; i++) { const r = m[4*i+3]; if (r > 1.6) top.push([r, i]); }
  top.sort((a, b) => b[0] - a[0]);
  const at = (x, y, z) => (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) ? -1 : fl[x + nx * (y + ny * z)];
  const out = { steps, count: top.length, spots: [] };
  for (const [r, i] of top.slice(0, 6)) {
    const x = i % nx, y = Math.floor(i / nx) % ny, z = Math.floor(i / (nx * ny));
    let nb = ''; for (const [dx, dy, dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]) nb += at(x+dx,y+dy,z+dz) + ' ';
    out.spots.push({ rho: +r.toFixed(2), x, y, z, flag: fl[i], u: [m[4*i], m[4*i+1], m[4*i+2]].map(v => +v.toFixed(3)), faceNb: nb.trim() });
  }
  // histogram of y for high-density cells
  const hy = {}; for (const [, i] of top) { const y = Math.floor(i / nx) % ny; hy[y] = (hy[y] || 0) + 1; }
  out.yHist = hy;
  return out;
}, steps);
console.log(JSON.stringify(r, null, 1));
await browser.close();
