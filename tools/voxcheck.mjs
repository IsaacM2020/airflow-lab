import { chromium } from 'playwright';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 800, height: 450 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?model=f1&nohud=1&streams=0`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 240000 });
const out = await page.evaluate(async () => {
  const sim = window.__app.sim, l = sim.lbm;
  const hash = async () => {
    const st = sim.device.createBuffer({ size: l.N * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = sim.device.createCommandEncoder(); enc.copyBufferToBuffer(l.flags, 0, st, 0, l.N * 4); sim.device.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ); const f = new Uint32Array(st.getMappedRange().slice(0)); st.unmap();
    let h = 0, solid = 0, near = 0, ymin = 999;
    for (let i = 0; i < f.length; i++) { const v = f[i]; if (v === 1) { solid++; const y = Math.floor(i / l.nx) % l.ny; if (y < ymin) ymin = y; } else if (v === 2) near++; h = (h * 31 + v * (i % 977 + 1)) >>> 0; }
    return { h, solid, near, ymin };
  };
  const res = [];
  res.push({ tag: 'initial', ride: sim.rideCm, vox: sim.voxel, ...(await hash()) });
  for (let k = 0; k < 3; k++) { await sim.applyGeometry(); res.push({ tag: 're' + k, ride: sim.rideCm, vox: sim.voxel, ...(await hash()) }); }
  return res;
});
for (const r of out) console.log(JSON.stringify(r));
await browser.close();
