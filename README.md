# Airflow Lab

A live 3D Navier-Stokes wind tunnel in the browser: airflow through an Airbus A380 and an F1 car, solved on the GPU with the lattice Boltzmann method (WebGPU), for an IB Math Exhibition.

- Run: double-click `Airflow Lab.command`, or `npm run dev` and open http://127.0.0.1:5180 in Chrome
- Understand it: read `EXPLAINER.md`
- Design spec: `~/IsaacOS/projects/airflow-lab/design-2026-09-30.md`
- Tests: `node tools/run-validation.mjs` (needs the dev server running), `node tools/e2e.mjs`

![F1](docs/f1-streamlines.png)
![A380](docs/a380-streamlines-vortices.png)

Credits: F1 model markste-in/c42 (WTFPL); A380 model Flightradar24 fr24-3d-models / FlightGear (GPL-2.0).
