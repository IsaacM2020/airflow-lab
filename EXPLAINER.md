# Airflow Lab: how it works and why

Isaac, this is for you. The goal: after reading it you can explain how the simulation works, why each choice was made, and answer your teacher's hard questions. Write your own exhibition in your own words; this is the understanding behind it.

## 1. The big picture

A live 3D wind tunnel in the browser. It solves the physics of airflow on your GPU, 70 to 85 times a second, around a real A380 model and a real F1 car model, and draws the result.

```
   3D model (.bin)                                        what you see
        |                                                      ^
        v                                                      |
  GPU voxeliser  --->  solid/air grid  --->  LBM solver  --->  viz volumes  --->  streamlines, pressure paint,
  (triangles to        (which cells        (stream +          (velocity, Cp,      slice plane, vortex tubes
   cells)               are the car)        collide, 19        Q-criterion)
                                            directions)             |
                                                 |                  v
                                                 +----->  forces on the body  --->  Cd, CL, CY readouts
```

Everything heavy (solver, voxeliser, particles, vortex ray-marching) runs on the GPU. The CPU only wires things together and handles the sliders.

## 2. Background you need

**Navier-Stokes equations.** Newton's second law for air: acceleration + advection = pressure push + viscous friction. There is no formula that solves them for a car, so we solve them numerically on a grid.

**Reynolds number, Re = U L / nu.** The one number that decides what a flow looks like. Same Re, same flow pattern (that is why scale models work in wind tunnels). The real F1 car is at Re around 3e7. This simulation is at about 4.5e3. That gap is the biggest honest limitation, and section 7 explains what it costs us.

**Lattice Boltzmann method (LBM).** Instead of solving Navier-Stokes directly, keep 19 numbers per cell: how much fluid is "moving" in each of 19 directions (the D3Q19 lattice). Each step has two moves:
- stream: every packet hops one cell along its direction
- collide: packets relax toward an equilibrium distribution (this is where viscosity lives)

Density and velocity are just sums over the 19 numbers. A math result called Chapman-Enskog shows this reproduces Navier-Stokes with `nu = (tau - 1/2) / 3` and `p = rho / 3`. We tested that relation directly (Taylor-Green vortex, 0.04% error).

**Bounce-back.** Where a packet would enter a solid cell, it reverses direction instead. That makes the body a no-slip wall. Summing the momentum change over all those bounces gives the force on the body, with no pressure integration.

**Smagorinsky sub-grid model.** Eddies smaller than a cell can't be resolved, so the solver adds an "eddy viscosity" that grows in violent shear. That is what lets it run at low viscosity without blowing up.

**Pressure coefficient, Bernoulli.** `Cp = (p - p_inf) / (1/2 rho U^2)`. Faster air means lower pressure. Stagnation is Cp = +1 (red), suction is negative (blue).

**Q-criterion.** Splits the velocity gradient into strain and rotation: `Q = 1/2 (|Omega|^2 - |S|^2)`. Where Q is positive, rotation wins: that is a vortex. The purple/blue tubes are surfaces of constant Q.

**WebGPU compute.** Lets JavaScript launch programs (written in WGSL) on the GPU that touch millions of cells in parallel. LBM is perfect for it because every cell does the same independent work.

## 3. File map

```
airflow-lab/
  index.html                    the page (canvas + fonts)
  src/main.ts                   boot: reads URL options, builds the UI, starts the app
  src/app/
    presets.ts                  per-model settings: grid size, cell size, boundaries, camera, Re range
    app.ts                      the frame loop: physics steps, particles, drawing, stats
    sim.ts                      one running wind tunnel: solver + voxeliser + viz + forces
  src/solver/
    lbm.ts                      THE solver: D3Q19 shaders (stream, collide, boundaries, forces)
    voxel.ts                    GPU voxeliser: triangles -> solid cells (mark + flood fill)
    viz.ts                      packs velocity, Cp, Q into 3D textures for drawing
  src/render/
    scene.ts                    background, floor reflection, glossy model shader, domain box
    layers.ts                   streamlines (particles), slice plane, vortex ray-marcher
    post.ts                     bloom (glow), filmic tone mapping, edge smoothing
    camera.ts, math.ts, frame.ts, model.ts, mipmaps.ts   support code
  src/ui/
    ui.ts, style.css            sliders, readouts, dock, modals
    maths.ts                    the "Show the maths" panel (KaTeX equations, live numbers)
    validation-data.ts          measured validation results shown in the app
  src/validation/validate.ts    the three benchmark tests (also runnable headless)
  tools/prepare-models.mjs      flattens the raw model files into clean meshes
  tools/*.mjs                   test and screenshot scripts I used to check everything
  public/models/                the two baked models + textures
```

## 4. One real request, traced end to end

You drag the **angle of attack** slider on the A380 from 3 to 8 degrees.

1. `src/ui/ui.ts`: the `input` event fires, calls `app.setAoa(8)`.
2. `src/app/app.ts` `setAoa`: stores 8, tells the sim, calls `sim.applyGeometry()`.
3. `src/app/sim.ts` `applyGeometry`: builds a new body-to-cell matrix with `bodyToCell({pitch: -8 deg, ...})` (`src/solver/voxel.ts`). Nose up is a negative rotation about z because the nose points to -x.
4. `Voxeliser.voxelise` (GPU, four small programs):
   - `mark`: every one of the 36,000 triangles marks each grid cell it touches, using an exact triangle-box overlap test. This gives a thin shell.
   - `seed`: the domain boundary cells that aren't shell are marked "outside".
   - `grow`: repeatedly, any non-shell cell next to an outside cell becomes outside. Stops when nothing changes (about 100 passes).
   - `finish`: solid = shell OR not-outside. So even a leaky mesh gets filled.
5. `lbm.classify()`: tags fluid cells that touch a solid, so only those pay for solid checks in the fast path.
6. Next frame, `lbm.step` runs. The flow doesn't restart: cells that turned solid are held at rest, cells that opened up start from rest, and the air rearranges around the new attitude over the next few hundred steps.
7. `sim.settled` resets so the readout shows "..." instead of stale numbers, then `sampleForces` reads the momentum-exchange accumulator and computes `CL = Fy / (1/2 U^2 A)`.
8. Every frame: `viz.update` refreshes the velocity/Cp/Q volumes, `streams.advance` moves 7,000 smoke particles through the new velocity field (midpoint Runge-Kutta), and the scene draws.

## 5. Why each choice beat the alternatives

| Choice | Alternative | Why this won |
|---|---|---|
| Lattice Boltzmann | Direct Navier-Stokes (projection method) | LBM is local: each cell reads only its neighbours, so it maps perfectly to a GPU. Direct NS needs a global pressure solve every step, which is slow and awkward in 3D. LBM also gives you the strain rate and force for free. |
| Raw WebGPU | Three.js / WebGL | One GPU device shared by solver and renderer, no copies. WebGL has no compute shaders. |
| Voxel bounce-back | Body-fitted mesh | Any messy free 3D model works with no meshing step; the shape can move live (angle of attack). Cost: staircase surfaces. |
| GPU flood-fill voxeliser | Ray-parity inside test | Parity fails on models with holes (both models have some). Flood fill from outside doesn't care. |
| Free-slip floor for the F1 | Moving belt | A moving belt over a one-cell gap pumped air into dead ends and the density hit 6. Real rolling-road tunnels suck the boundary layer off for the same reason, so free-slip is a fair stand-in. |
| Smagorinsky LES | No turbulence model | Without it, it goes unstable (NaN) at the Re we need. |
| Measure p_inf upstream | Assume rho = 1 | The mean density drifts about 1%, which looks like Cp = -1.4 everywhere. Found by looking at a screenshot that was all blue. |
| Discard the start-up transient | Show forces immediately | The first 1,800 or so steps are a shock wave ringing around the tunnel. The force during that time is meaningless. |
| Spin-up behind the loading screen | Start from still air | Otherwise the wake takes 30 seconds to appear. |
| Adaptive steps per frame | Fixed | Keeps the camera smooth while giving the physics whatever GPU time is left. |

## 6. What was validated, and how well

All measured on your M4, in headless Chrome, reported as they came out (`node tools/run-validation.mjs`):

- **Taylor-Green vortex** (exact solution): decay rate matches to **0.04%**.
- **Poiseuille channel flow** (exact parabola): **0.03%** at tau = 1, 0.17% at tau = 0.7.
- **Sphere drag vs the Schiller-Naumann experimental curve**, Re 10 to 200: solver is **8 to 14% high**, shrinking as Re rises. At Re 50, opening the test box (inlet 8 diameters upstream, walls 8D) cuts the error from 12.3% to **7.5%**; a finer sphere gives 9.7%. Reading: a good part is the box being too tight, the rest is the staircase surface and the correlation's own scatter (it is a fit, good to roughly 5%).
- **F1 drag**: Cd about 1.0 to 1.3 on frontal area. Real F1 cars are around 0.9 to 1.1. Same ballpark.
- **A380 lift vs angle** (final configuration, wing area 845 m2 as reference): CL = -0.03, 0.10, 0.26, 0.46 at 0, 4, 8, 12 degrees. Steady, near-linear rise: the right trend. The slope is about 2.4 per radian; a textbook finite wing of this aspect ratio gives roughly 4.5, so lift is under-predicted by about half.
- **Stability**: ran 2,400+ steps with no NaN and density between 0.985 and 1.014 across the tested settings.

## 7. Honest limits (say these out loud, your teacher will respect it)

- **Re is thousands of times lower than reality** (about 6,000x for the F1, 75,000x for the A380 at default speed). The grid can't resolve thin boundary layers. So the numbers are trends, not wind-tunnel data.
- **The F1 makes almost no downforce here.** Real downforce needs wings and a 4 cm floor gap working at Re 1e7. Here a wing is about 6 cells across at a chord Re of a few hundred. I tested two resolutions (5 cm and 3.5 cm cells): same answer, so Re is the cause, not the grid. That is why the F1 panel shows drag and side force, not downforce. I would rather show that than a fake number.
- **A380 drag is about 15 times too high** (Cd 0.46 to 0.53 measured, roughly 0.03 real), so lift-to-drag is 0.5 to 0.9 instead of about 17. Wings are 1 to 2 cells thick, so they behave like blunt plates at low Re. The lift *trend* is right, the drag is inflated.
- **Incompressible.** Fine for the car, an approximation for an airliner at Mach 0.85.
- **Free-slip ground** for the F1 (see table above).

## 8. Bugs found on the way (good story for the exhibition)

Every one of these came from checking a result that looked wrong:

1. Solver blew up (99% NaN). Cause: static tyres and floor touching a ground moving at wind speed. Fixed by a free-slip floor.
2. Density piled up to 6 under the car. Cause: the belt dragging air into a dead-end pocket. Same fix.
3. Lift coefficient of +12 on an F1 car. Cause: I averaged the force over the start-up shock. Fixed by discarding the transient.
4. Downforce that looked great. Cause: my app was tilting the F1 by the aircraft's 3 degree default so its rear wheels dug into the floor. Fixed, and the "downforce" disappeared, which is how I found the real Re limit.
5. Every pressure picture came out blue. Cause: assumed reference density. Fixed by measuring it.
6. Sphere drag 12% high. Not a bug: a tight test box. Proved by widening it.

## 9. Questions your teacher might ask

- *Why lattice Boltzmann and not Navier-Stokes directly?* Same physics (Chapman-Enskog shows it recovers NS), but every cell only talks to neighbours, so 3 million cells update in parallel on a GPU.
- *How do you know it's right?* Three tests against exact or published results (section 6), plus where it fails and why.
- *What is tau?* The relaxation time. Viscosity is `(tau - 1/2)/3`. It must exceed 1/2.
- *Why is your Reynolds number so low?* Cells are 5 cm and 75 cm; resolving real boundary layers needs micrometres. We model small eddies (Smagorinsky) rather than resolve them.
- *Why does the A380 have such poor L/D?* Thin wings aren't resolved and Re is low, so drag is about 15 times too high; the lift trend with angle is right, absolute drag is not.
- *What would you do with more compute?* Finer grid near the surfaces, wall functions, and a moving-belt floor with boundary-layer suction.

## 10. Running it

Double-click `Airflow Lab.command` in this folder (starts the server and opens Chrome), or:

```
cd ~/Elemental/Water/airflow-lab
npm run dev      # then open http://127.0.0.1:5180 in Chrome
```

Keys: `space` pause, `1-4` layers, `M` maths, `V` validation. Needs a browser with WebGPU (Chrome, or Safari 26+). Credits: F1 model markste-in/c42 (WTFPL), A380 Flightradar24/FlightGear models (GPL-2.0).
