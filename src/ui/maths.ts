/**
 * The "Show the maths" panel. Equations are KaTeX strings in data-tex attributes; ui.ts renders them.
 * <span data-live="..."> placeholders are filled from the running simulation.
 * <button data-show="..."> chips switch on the matching layer in the scene.
 */
export const MATHS_HTML = /* html */ `
<div class="live">
  <div class="eyebrow">Your simulation right now</div>
  <div class="cells">
    <div><span>Relaxation time τ</span><b data-live="tau">–</b></div>
    <div><span>Viscosity ν (lattice)</span><b data-live="nu">–</b></div>
    <div><span>Reynolds number Re</span><b data-live="re">–</b></div>
    <div><span>Mach number</span><b data-live="mach">–</b></div>
    <div><span>Cell size Δx</span><b data-live="dx">–</b></div>
    <div><span>Grid</span><b data-live="grid">–</b></div>
  </div>
</div>

<section class="step">
  <div class="num">STEP 1</div>
  <h3>What we are solving: the Navier–Stokes equations</h3>
  <p>Air is a fluid. Newton's second law for a small blob of air gives the <b>incompressible Navier–Stokes equations</b>:</p>
  <div class="eq" data-tex="\\underbrace{\\frac{\\partial \\mathbf{u}}{\\partial t}}_{\\text{acceleration}} + \\underbrace{(\\mathbf{u}\\cdot\\nabla)\\mathbf{u}}_{\\text{advection}} = \\underbrace{-\\frac{1}{\\rho}\\nabla p}_{\\text{pressure}} + \\underbrace{\\nu\\,\\nabla^{2}\\mathbf{u}}_{\\text{viscosity}}, \\qquad \\nabla\\cdot\\mathbf{u}=0"></div>
  <ul>
    <li><b>Advection</b> (u·∇)u: the flow carries its own momentum along. This term is <b>nonlinear</b>, which is why turbulence exists and why nobody has a formula for airflow past a car.</li>
    <li><b>Pressure</b> −∇p/ρ: air is pushed from high pressure to low pressure.</li>
    <li><b>Viscosity</b> ν∇²u: sticky friction that smooths out differences in velocity.</li>
    <li><b>∇·u = 0</b>: incompressibility. Valid because the flow is far below the speed of sound (Mach ≪ 1).</li>
  </ul>
  <div class="hint">Nobody can solve this on paper for a car, so we solve it numerically: chop space into small cells and step time forward.</div>
</section>

<section class="step">
  <div class="num">STEP 2</div>
  <h3>One number rules the flow: Reynolds number</h3>
  <p>Divide every quantity by a typical speed U and length L and the equation collapses to a single parameter:</p>
  <div class="eq" data-tex="\\frac{\\partial \\mathbf{u}^*}{\\partial t^*} + (\\mathbf{u}^*\\!\\cdot\\!\\nabla^*)\\mathbf{u}^* = -\\nabla^* p^* + \\frac{1}{\\mathrm{Re}}\\nabla^{*2}\\mathbf{u}^*, \\qquad \\mathrm{Re}=\\frac{UL}{\\nu}"></div>
  <p>Two flows with the same Re look the same (this is <b>dynamic similarity</b>, the reason wind tunnels work with scale models). Re is the ratio of inertia to viscosity.</p>
  <div class="hint">Real Formula 1 car: Re ≈ <b data-live="reReal">–</b>. This simulation: Re ≈ <b data-live="re">–</b>. Our grid cannot resolve the tiny eddies of the real Re, so we <b>model</b> them (Step 5). Trends are right; exact numbers are not wind-tunnel grade.</div>
</section>

<section class="step">
  <div class="num">STEP 3</div>
  <h3>The trick: simulate particles, get fluid for free (lattice Boltzmann)</h3>
  <p>Instead of solving Navier–Stokes directly, track how many "packets" of fluid f<sub>i</sub> are moving in each of 19 directions <b>c</b><sub>i</sub> at every cell (the <b>D3Q19</b> lattice). Each time step has two parts: <b>stream</b> (move every packet one cell along its direction) and <b>collide</b> (relax toward equilibrium):</p>
  <div class="eq" data-tex="f_i(\\mathbf{x}+\\mathbf{c}_i\\Delta t,\\;t+\\Delta t) \\;=\\; f_i(\\mathbf{x},t) \\;-\\; \\frac{1}{\\tau}\\Big[f_i(\\mathbf{x},t) - f_i^{\\mathrm{eq}}(\\mathbf{x},t)\\Big]"></div>
  <p>The equilibrium is a Maxwell distribution expanded to second order, with sound speed c<sub>s</sub><sup>2</sup> = 1/3 in lattice units:</p>
  <div class="eq" data-tex="f_i^{\\mathrm{eq}} = w_i\\,\\rho\\left[1 + \\frac{\\mathbf{c}_i\\!\\cdot\\!\\mathbf{u}}{c_s^{2}} + \\frac{(\\mathbf{c}_i\\!\\cdot\\!\\mathbf{u})^{2}}{2c_s^{4}} - \\frac{\\mathbf{u}\\!\\cdot\\!\\mathbf{u}}{2c_s^{2}}\\right],\\qquad w_i=\\tfrac13,\\ \\tfrac1{18},\\ \\tfrac1{36}"></div>
  <p>Density and velocity are just <b>moments</b> (sums) of the packets:</p>
  <div class="eq" data-tex="\\rho=\\sum_{i=0}^{18} f_i,\\qquad \\rho\\,\\mathbf{u}=\\sum_{i=0}^{18} f_i\\,\\mathbf{c}_i"></div>
  <p>The weights are 1/3 for the rest packet, 1/18 for the 6 axis directions and 1/36 for the 12 diagonals: 1/3 + 6/18 + 12/36 = 1. Every cell does this independently, which is why a GPU can update about 4 million cells per step, ~100 times a second.</p>
</section>

<section class="step">
  <div class="num">STEP 4</div>
  <h3>Why that gives Navier–Stokes (Chapman–Enskog)</h3>
  <p>Taylor-expand the streaming step, then expand f<sub>i</sub> = f<sub>i</sub><sup>(0)</sup> + ε f<sub>i</sub><sup>(1)</sup> + … in a small parameter ε and take moments order by order. The first two moments give continuity and momentum:</p>
  <div class="eq" data-tex="\\partial_t\\rho+\\nabla\\!\\cdot(\\rho\\mathbf{u})=0,\\qquad \\rho(\\partial_t\\mathbf{u}+\\mathbf{u}\\!\\cdot\\!\\nabla\\mathbf{u}) = -\\nabla p + \\nabla\\!\\cdot\\!\\big[\\rho\\nu(\\nabla\\mathbf{u}+\\nabla\\mathbf{u}^{T})\\big]"></div>
  <p>with the two links between the lattice and physics:</p>
  <div class="eq" data-tex="p=\\rho\\,c_s^{2}=\\frac{\\rho}{3},\\qquad\\qquad \\nu = c_s^{2}\\Big(\\tau-\\tfrac12\\Big)\\Delta t = \\frac{\\tau-\\tfrac12}{3}"></div>
  <div class="hint">The <b>−½</b> comes from the Taylor expansion: streaming itself adds a bit of numerical viscosity that τ has to cancel. It also means τ must exceed ½ (viscosity can't be negative). This solver was checked against this exact relation: a Taylor–Green vortex decays at the predicted rate to <b>0.04%</b> (see Validation).</div>
</section>

<section class="step">
  <div class="num">STEP 5</div>
  <h3>Turbulence we cannot resolve: the Smagorinsky model</h3>
  <p>At high Re the smallest eddies are far smaller than one 5 cm cell. Large-eddy simulation resolves the big swirls and adds an <b>eddy viscosity</b> for the rest:</p>
  <div class="eq" data-tex="\\nu_t=(C_s\\Delta)^2\\,|S|,\\quad |S|=\\sqrt{2S_{ij}S_{ij}},\\quad S_{ij}=\\tfrac12\\big(\\partial_i u_j+\\partial_j u_i\\big)"></div>
  <p>In lattice Boltzmann the strain rate is already sitting in the "non-equilibrium" part of f, so no derivatives are needed:</p>
  <div class="eq" data-tex="\\Pi^{\\mathrm{neq}}_{ab}=\\sum_i c_{ia}c_{ib}\\,(f_i-f_i^{\\mathrm{eq}}),\\qquad \\tau=\\tfrac12\\Big[\\tau_0+\\sqrt{\\tau_0^{2}+18\\sqrt2\\,C_s^{2}\\,|\\Pi^{\\mathrm{neq}}|/\\rho}\\Big]"></div>
  <p>Here C<sub>s</sub> = 0.14. Where the flow is smooth τ ≈ τ<sub>0</sub>; in violent shear (behind wheels, wing tips) τ grows and stabilises the solution.</p>
</section>

<section class="step">
  <div class="num">STEP 6</div>
  <h3>The body, and how forces come out</h3>
  <p>The 3D model is chopped into a grid of solid cells (<i>voxelised</i> on the GPU). A packet that hits a solid bounces straight back (<b>bounce-back</b>, no-slip wall):</p>
  <div class="eq" data-tex="f_{\\bar i}(\\mathbf{x}_f,\\,t+\\Delta t)=f_i^{*}(\\mathbf{x}_f,\\,t)"></div>
  <p>Each bounce reverses a packet's momentum, and momentum conservation says the body gets the difference. Summing over every fluid–solid link gives the force with <b>no pressure integration needed</b>:</p>
  <div class="eq" data-tex="\\mathbf{F}=\\sum_{\\text{links}} 2\\,f_i^{*}(\\mathbf{x}_f)\\,\\mathbf{c}_i,\\qquad C_D=\\frac{F_x}{\\tfrac12\\rho U^2A},\\quad C_L=\\frac{F_y}{\\tfrac12\\rho U^2A}"></div>
  <p>Reference area A: the real wing area (845 m²) for the A380, the frontal area of the voxelised car for the F1.</p>
</section>

<section class="step">
  <div class="num">STEP 7</div>
  <h3>Pressure, Bernoulli, wings and ground effect</h3>
  <p>Along a streamline outside the boundary layer, Bernoulli's equation says faster air means lower pressure. Written as a pressure coefficient:</p>
  <div class="eq" data-tex="p+\\tfrac12\\rho u^{2}=\\text{const}\\ \\Longrightarrow\\ C_p=\\frac{p-p_\\infty}{\\tfrac12\\rho U^{2}}\\approx 1-\\Big(\\frac{u}{U}\\Big)^{2}"></div>
  <ul>
    <li><b>Stagnation point</b> (air stops): u = 0, C<sub>p</sub> = +1. The red patch on the nose and front wing.</li>
    <li><b>Suction</b> (air accelerates): u &gt; U, C<sub>p</sub> &lt; 0. Blue on the top of the A380 wing and under the F1 floor.</li>
    <li><b>Lift/downforce</b> is the pressure difference between the two sides, integrated over the surface.</li>
    <li><b>Ground effect</b> (real F1): squeeze air through the narrow gap under the floor, it speeds up, pressure drops, the car is sucked to the road. <b>This simulation cannot reproduce it</b>: see the honesty note below.</li>
  </ul>
  <button class="btn showbtn" data-show="pressure">Show pressure paint</button><button class="btn showbtn" data-show="slicep">Show pressure slice</button>
</section>

<section class="step">
  <div class="num">STEP 8</div>
  <h3>Vortices and the price of lift</h3>
  <p>To find swirls, decompose the velocity gradient into strain S and rotation Ω. The <b>Q-criterion</b> highlights where rotation beats strain:</p>
  <div class="eq" data-tex="Q=\\tfrac12\\big(\\lVert\\Omega\\rVert^{2}-\\lVert S\\rVert^{2}\\big),\\qquad \\Omega_{ij}=\\tfrac12(\\partial_i u_j-\\partial_j u_i)"></div>
  <p>The tubes you see are the surface Q = threshold. A lifting wing leaks air from the high-pressure underside round the tip, making <b>wingtip vortices</b>. They cost energy, which shows up as <b>induced drag</b>:</p>
  <div class="eq" data-tex="C_{D,i}=\\frac{C_L^{2}}{\\pi\\,e\\,\\mathrm{AR}}"></div>
  <p>For a fixed wing, more lift (higher angle of attack) means much more induced drag: the square of C<sub>L</sub>. Raise the A380's angle of attack and watch the vortices strengthen.</p>
  <button class="btn showbtn" data-show="vortex">Show vortex tubes</button><button class="btn showbtn" data-show="streams">Show streamlines</button>
</section>

<section class="step">
  <div class="num">HONESTY</div>
  <h3>What this simulation is, and is not</h3>
  <ul>
    <li><b>Is:</b> a real 3D solver of the Navier–Stokes equations (via lattice Boltzmann), checked against exact solutions. The physics trends (lift versus angle, ground effect, wake, vortices) are genuine.</li>
    <li><b>Is not:</b> wind-tunnel accurate. Cells are 5 cm (car) and 75 cm (A380), so thin wings are 1–2 cells thick and the boundary layer is unresolved. Real Re is 10<sup>4</sup> times higher than the simulated one.</li>
    <li><b>No F1 downforce.</b> Real downforce comes from thin wings and a 4 cm floor gap working at Re ≈ 10<sup>7</sup>. Here a wing chord is about 6 cells and the chord Reynolds number is a few hundred, so those surfaces stall and the vertical force comes out near zero. Tested at two resolutions (5 cm and 3.5 cm cells) with the same result, so Re is the limit, not the grid. That is why the F1 panel reports <b>drag and side force</b> (which the solver does capture) and not downforce.</li>
    <li>The F1 floor is <b>free-slip</b> (a rolling road with its boundary layer removed): a moving belt over a one-cell gap pumps air into dead ends on a grid this coarse.</li>
    <li>Incompressible: fine for the car, an approximation for an airliner at Mach 0.85.</li>
  </ul>
</section>
`;
