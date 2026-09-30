import type { Faces } from '../solver/lbm';
import { Face } from '../solver/lbm';

export interface Preset {
  id: 'f1' | 'a380';
  label: string;
  model: string;
  /** Real model length in metres. */
  length: number;
  /** Domain size in cells (x streamwise, y up, z lateral). */
  dims: [number, number, number];
  /** Metres per cell. */
  dx: number;
  /** Where the body pivot (its centre, or the ground point for the F1) is placed, in cells. */
  place: [number, number, number];
  /** Body-space rotation centre in metres. */
  pivot: [number, number, number];
  ground: boolean;
  faces: Faces;
  /** Real-world speed range for the wind-speed slider, km/h. */
  speed: { min: number; max: number; def: number };
  /** Reynolds number (based on body length in cells) at the slider ends. */
  reSim: { min: number; max: number };
  /** Real Reynolds number at the default speed, for the on-screen honesty note. */
  reReal: (kmh: number) => number;
  camera: { target: [number, number, number]; distance: number; yaw: number; pitch: number };
  /** Inlet smoke rake, in cells: [y0, y1, z0, z1]. */
  seedBox: [number, number, number, number];
  /** Smoke particle count and trail length (frames). */
  smoke: { count: number; trail: number };
  /** Overall smoke brightness, 0..1. */
  smokeOpacity: number;
  /** Pressure coefficient that saturates the pressure colour scale. */
  cpScale: number;
  /** Reference area for the force coefficients: the body's frontal area (F1) or the real wing area (A380). */
  refAreaM2?: number;
  reference: 'frontal' | 'planform';
}

const airNu = 1.5e-5; // m^2/s at sea level
const airNuHigh = 3.5e-5; // m^2/s at cruise altitude

export const PRESETS: Record<'f1' | 'a380', Preset> = {
  f1: {
    id: 'f1',
    label: 'Formula 1 car',
    model: 'f1',
    length: 5.63,
    dx: 0.05,
    dims: [300, 80, 120],
    place: [96, 0, 60],
    pivot: [0, 0, 0],
    ground: true,
    // Free-slip floor: a belt with its boundary layer removed. A dragging belt over a one-cell underfloor gap
    // pumps air into dead ends on a grid this coarse.
    faces: [Face.Inlet, Face.Outlet, Face.Slip, Face.Slip, Face.Slip, Face.Slip],
    speed: { min: 120, max: 340, def: 250 },
    reSim: { min: 2000, max: 8000 },
    reReal: (kmh) => ((kmh / 3.6) * 5.63) / airNu,
    camera: { target: [96, 9, 60], distance: 100, yaw: -0.75, pitch: 0.28 },
    seedBox: [1, 26, 36, 84],
    smoke: { count: 3600, trail: 48 },
    smokeOpacity: 1,
    cpScale: 0.7,
    reference: 'frontal',
  },
  a380: {
    id: 'a380',
    label: 'Airbus A380',
    model: 'a380',
    length: 72.7,
    dx: 0.75,
    dims: [260, 76, 136],
    place: [88, 38, 68],
    pivot: [0, 0, 0],
    ground: false,
    faces: [Face.Inlet, Face.Outlet, Face.Slip, Face.Slip, Face.Slip, Face.Slip],
    speed: { min: 250, max: 950, def: 850 },
    reSim: { min: 2000, max: 8000 },
    reReal: (kmh) => ((kmh / 3.6) * 72.7) / airNuHigh,
    camera: { target: [100, 38, 68], distance: 215, yaw: -0.8, pitch: 0.3 },
    seedBox: [20, 56, 16, 120],
    smoke: { count: 7000, trail: 64 },
    smokeOpacity: 0.6,
    cpScale: 0.42,
    refAreaM2: 845,
    reference: 'planform',
  },
};
