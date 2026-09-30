import { chromium } from 'playwright';
const browser = await chromium.launch({
  channel: 'chrome', headless: true,
  args: ['--enable-unsafe-webgpu','--enable-features=Vulkan,WebGPU','--use-angle=metal','--ignore-gpu-blocklist']
});
const page = await browser.newPage();
await page.goto('http://127.0.0.1:5199/probe.html');
const r = await page.evaluate(async () => {
  if (!navigator.gpu) return { error: 'no navigator.gpu' };
  const a = await navigator.gpu.requestAdapter();
  if (!a) return { error: 'no adapter' };
  const info = a.info || {};
  const l = {}; for (const k in a.limits) l[k] = a.limits[k];
  return { vendor: info.vendor, arch: info.architecture, desc: info.description, features: [...a.features], limits: {
    maxBufferSize: l.maxBufferSize, maxStorageBufferBindingSize: l.maxStorageBufferBindingSize,
    maxStorageBuffersPerShaderStage: l.maxStorageBuffersPerShaderStage, maxComputeWorkgroupSizeX: l.maxComputeWorkgroupSizeX,
    maxComputeInvocationsPerWorkgroup: l.maxComputeInvocationsPerWorkgroup, maxTextureDimension3D: l.maxTextureDimension3D,
    maxComputeWorkgroupStorageSize: l.maxComputeWorkgroupStorageSize } };
});
console.log(JSON.stringify(r, null, 1));
await browser.close();
