// Usage: node tools/shot.mjs "<query>" out.png [waitMs] [width] [height]
import { chromium } from 'playwright';
const [, , query = '', out = 'shot.png', wait = '2500', w = '1600', h = '900', actionsJson = '[]'] = process.argv;
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({ channel: 'chrome', headless: true, args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: 1 });
const seen = new Set();
page.on('console', (m) => { if (['error','warning'].includes(m.type())) { const t = m.text().slice(0,600); if (!seen.has(t) && !t.includes('404') && seen.size < 8) { seen.add(t); console.log('[browser]', t);} } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/index.html?${query}`);
await page.waitForFunction(() => window.__ready === true, null, { timeout: 240000 }).catch(() => console.log('not ready'));
await page.waitForTimeout(+wait);
for (const a of JSON.parse(actionsJson)) {
  if (a.click) await page.click(a.click);
  if (a.fill) await page.$eval(a.fill[0], (el, v) => { el.value = v; el.dispatchEvent(new Event('input', { bubbles: true })); }, a.fill[1]);
  if (a.scroll) await page.$eval(a.scroll[0], (el, v) => { el.scrollTop = v; }, a.scroll[1]);
  if (a.wait) await page.waitForTimeout(a.wait);
}
await page.screenshot({ path: out });
console.log('saved', out);
await browser.close();
