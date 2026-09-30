// Usage: node tools/run-validation.mjs [taylor,poiseuille1,sphere100 | all]
import { chromium } from 'playwright';
const which = process.argv[2] || 'all';
const base = process.env.BASE || 'http://127.0.0.1:5180';
const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=metal', '--ignore-gpu-blocklist'],
});
const page = await browser.newPage();
const seen = new Set();
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') { const t = m.text().slice(0, 400); if (!seen.has(t) && seen.size < 6) { seen.add(t); console.log('[browser]', t); } } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('framenavigated', (f) => console.log('[nav]', f.url()));
page.on('crash', () => console.log('[page crashed]'));
await page.goto(base + '/validate.html');
await page.waitForFunction(() => typeof window.__validate === 'function', null, { timeout: 30000 });
const res = await page.evaluate((w) => window.__validate(w), which);
console.log(JSON.stringify(res.map(r => ({ name: r.name, pass: r.pass, summary: r.summary })), null, 1));
await browser.close();
process.exit(res.every(r => r.pass) ? 0 : 1);
