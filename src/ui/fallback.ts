import './style.css';

const REPO = 'https://github.com/IsaacM2020/airflow-lab';

/** Friendly full-page message for browsers or GPUs that cannot run the simulation. */
export function showFallback(err: Error) {
  const code = err.message;
  const reason =
    code === 'NO_WEBGPU'
      ? 'This browser does not support WebGPU, which the simulation needs to run its physics on your graphics card.'
      : code === 'NO_ADAPTER'
        ? 'WebGPU is available but no usable graphics card was found (some browsers turn it off for battery saving or remote sessions).'
        : code === 'GPU_LIMITS'
          ? 'This graphics card is too small: the solver needs about 250 MB in a single GPU buffer, and this device caps it lower. Phones and older integrated GPUs usually do.'
          : 'Something went wrong while starting the simulation: ' + code;
  document.body.innerHTML = /* html */ `
  <div style="position:fixed;inset:0;overflow:auto;background:radial-gradient(ellipse at 50% 0%,#10203a 0%,#05070b 65%);color:var(--text);font-family:var(--font-ui);padding:32px 20px">
    <div style="max-width:880px;margin:0 auto">
      <div class="eyebrow" style="margin-bottom:10px">Airflow Lab</div>
      <h1 style="font:700 30px/1.15 var(--font-display);margin:0 0 12px">A live 3D wind tunnel that runs on your GPU</h1>
      <p style="font-size:15px;line-height:1.6;color:#c4cfe4;margin:0 0 6px">${reason}</p>
      <p style="font-size:15px;line-height:1.6;color:#c4cfe4;margin:0 0 20px">To run it, open this page on a laptop or desktop in <b>Chrome or Edge</b> (or Safari 26 or newer). Meanwhile, here is what it looks like:</p>
      <div style="display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))">
        <img src="/img/f1-streamlines.jpg" alt="F1 car with smoke streamlines" style="width:100%;border-radius:14px;border:1px solid var(--line)">
        <img src="/img/a380-streamlines-vortices.jpg" alt="Airbus A380 with streamlines and vortices" style="width:100%;border-radius:14px;border:1px solid var(--line)">
        <img src="/img/f1-pressure.jpg" alt="Pressure painted on the F1 car" style="width:100%;border-radius:14px;border:1px solid var(--line)">
        <img src="/img/validation.jpg" alt="Validation against exact solutions" style="width:100%;border-radius:14px;border:1px solid var(--line)">
      </div>
      <p style="margin:22px 0 0;display:flex;gap:10px;flex-wrap:wrap"><a class="btn accent" href="${REPO}" style="text-decoration:none">See the code and how it works on GitHub</a></p>
    </div>
  </div>`;
}
