import { App, type AppState } from './app/app';
import { buildUI } from './ui/ui';
import { showFallback } from './ui/fallback';

async function boot() {
  const q = new URLSearchParams(location.search);
  const num = (k: string) => (q.has(k) ? parseFloat(q.get(k)!) : undefined);
  const flag = (k: string) => (q.has(k) ? q.get(k) === '1' : undefined);
  const init: Partial<AppState> = {};
  const set = <K extends keyof AppState>(k: K, v: AppState[K] | undefined) => { if (v !== undefined) init[k] = v; };
  set('model', (q.get('model') as AppState['model']) ?? undefined);
  set('speedKmh', num('speed'));
  set('aoa', num('aoa'));
  set('yaw', num('cyaw'));
  set('streams', flag('streams'));
  set('pressure', flag('pressure'));
  set('slice', flag('slice'));
  set('vortex', flag('vortex'));
  set('sliceAxis', num('axis') as AppState['sliceAxis'] | undefined);
  set('slicePos', num('pos'));
  set('sliceMode', num('mode') as AppState['sliceMode'] | undefined);
  set('vortexThr', num('thr'));
  set('exposure', num('exp'));

  const canvas = document.getElementById('gpu') as HTMLCanvasElement;
  const app = new App(canvas, init);
  (window as any).__app = app;
  if (!q.has('nohud')) buildUI(app);
  await app.init();
  if (q.has('view')) app.setView(q.get('view') as any);
  if (q.has('yaw')) app.camera.set({ target: [...app.preset.camera.target], distance: app.preset.camera.distance * (num('zoom') ?? 1), yaw: num('yaw')!, pitch: num('pitch') ?? 0.3 }, true);
  if (q.has('warm') && app.sim && parseInt(q.get('warm')!) > 0) { await app.sim.lbm.run(parseInt(q.get('warm')!), 40); app.sim.history = []; await app.sim.lbm.readForces(); }
  (window as any).__ready = true;
}
boot().catch((e) => {
  console.error(e);
  showFallback(e instanceof Error ? e : new Error(String(e)));
});
