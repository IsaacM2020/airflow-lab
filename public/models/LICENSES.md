# Third-party 3D models

## Formula 1 car: `f1.bin`, `f1.json`, `textures/f1_*`
Alfa Romeo C42, from https://github.com/markste-in/c42 (based on the 3D model presented by Alfa Romeo).
Licence: **WTFPL** (Do What The Fuck You Want To Public License, http://www.wtfpl.net).
Changes: parts merged into one mesh, converted to the compact binary format used by this project, scaled to metres and oriented (x streamwise, y up).

## Airbus A380: `a380.bin`, `a380.json`, `textures/a380_*`
From the Flightradar24 3D model repository, https://github.com/flightradar24/fr24-3d-models (`models/a380.glb`),
which distributes aircraft models from the FlightGear project (https://www.flightgear.org).
Licence: **GNU General Public License v2.0**, full text in [`A380-GPL-2.0.txt`](A380-GPL-2.0.txt).
Changes: the glTF 1.0 file (524 parts with node transforms) was flattened into a single world-space mesh, converted to the compact binary format used by this project (`tools/prepare-models.mjs`), scaled to real length (72.7 m) and oriented (x streamwise, y up). The conversion script is in this repository.
These files remain under GPL-2.0. The MIT licence in the repository root covers the source code only.
