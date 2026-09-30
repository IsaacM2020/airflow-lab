import { cross, lookAt, multiply, perspective, invert, type Mat4, type Vec3 } from './math';

/** Orbit camera with damping. Drag = orbit, wheel = zoom, right-drag or shift-drag = pan. */
export class OrbitCamera {
  target: Vec3 = [0, 0, 0];
  distance = 100;
  yaw = -0.6;
  pitch = 0.3;
  fov = (38 * Math.PI) / 180;
  near = 0.5;
  far = 3000;
  private tTarget: Vec3 = [0, 0, 0];
  private tDistance = 100;
  private tYaw = -0.6;
  private tPitch = 0.3;
  /** Set when the user is interacting, so the app can stop auto-orbit. */
  interacting = false;
  autoOrbit = 0;

  constructor(private el: HTMLElement) {
    let mode: 'orbit' | 'pan' | null = null;
    let lx = 0, ly = 0;
    el.addEventListener('pointerdown', (e) => {
      if ((e.target as HTMLElement) !== el) return;
      mode = e.button === 2 || e.shiftKey ? 'pan' : 'orbit';
      lx = e.clientX; ly = e.clientY;
      this.interacting = true;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', (e) => {
      if (!mode) return;
      const dx = e.clientX - lx, dy = e.clientY - ly;
      lx = e.clientX; ly = e.clientY;
      if (mode === 'orbit') {
        this.tYaw -= dx * 0.0055;
        this.tPitch = Math.max(-1.5, Math.min(1.5, this.tPitch + dy * 0.0055));
      } else {
        const k = this.tDistance * 0.0011;
        const r = this.right();
        const u = this.upVec();
        for (let i = 0; i < 3; i++) this.tTarget[i] += (-r[i] * dx + u[i] * dy) * k;
      }
    });
    const end = (e: PointerEvent) => {
      mode = null;
      this.interacting = false;
      if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId);
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.tDistance = Math.max(4, Math.min(1500, this.tDistance * Math.exp(e.deltaY * 0.0012)));
    }, { passive: false });
  }

  private right(): Vec3 { return [Math.cos(this.yaw), 0, -Math.sin(this.yaw)]; }
  private upVec(): Vec3 {
    const cp = Math.cos(this.pitch);
    const z: Vec3 = [cp * Math.sin(this.yaw), Math.sin(this.pitch), cp * Math.cos(this.yaw)]; // eye direction
    return cross(z, this.right());
  }

  set(view: { target: Vec3; distance: number; yaw: number; pitch: number }, instant = false) {
    this.tTarget = [...view.target];
    this.tDistance = view.distance;
    this.tYaw = view.yaw;
    this.tPitch = view.pitch;
    if (instant) {
      this.target = [...view.target]; this.distance = view.distance; this.yaw = view.yaw; this.pitch = view.pitch;
    }
  }

  update(dt: number) {
    if (this.autoOrbit && !this.interacting) this.tYaw += this.autoOrbit * dt;
    const k = 1 - Math.exp(-dt * 9);
    for (let i = 0; i < 3; i++) this.target[i] += (this.tTarget[i] - this.target[i]) * k;
    this.distance += (this.tDistance - this.distance) * k;
    this.yaw += (this.tYaw - this.yaw) * k;
    this.pitch += (this.tPitch - this.pitch) * k;
  }

  eye(): Vec3 {
    const cp = Math.cos(this.pitch);
    return [
      this.target[0] + this.distance * cp * Math.sin(this.yaw),
      this.target[1] + this.distance * Math.sin(this.pitch),
      this.target[2] + this.distance * cp * Math.cos(this.yaw),
    ];
  }

  matrices(aspect: number): { view: Mat4; proj: Mat4; viewProj: Mat4; invViewProj: Mat4; eye: Vec3 } {
    const eye = this.eye();
    const view = lookAt(eye, this.target);
    const proj = perspective(this.fov, aspect, this.near, this.far);
    const viewProj = multiply(proj, view);
    return { view, proj, viewProj, invViewProj: invert(viewProj), eye };
  }
}
