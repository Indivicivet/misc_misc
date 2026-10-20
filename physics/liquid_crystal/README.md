# Nematic Liquid Crystal Cell & Spatial Light Modulator (SLM) Simulator

A continuum physics simulator for nematic liquid-crystal (LC) cells and Spatial Light Modulators (SLMs/LCoS). Implements the 1D Frank–Oseen continuum model with variable twist, chiral pitch, and tilt, alongside a 2D periodic grating model capturing electrostatic fringing, lateral Frank elasticity, and optical spatial harmonics ($e^{i k x} \to e^{i n k x}$).

---

## Features

- **1D Frank–Oseen Continuum Model** ([`frank_model.py`](frank_model.py)):
  - Splay ($K_{11}$), twist ($K_{22}$), bend ($K_{33}$), and chiral pitch ($q_0 = 2\pi/p_0$).
  - Anisotropic dielectric coupling with fixed RMS voltage drive ($\varepsilon_\perp, \Delta\varepsilon$).
  - Dynamic relaxation (rotational viscosity $\gamma_1$) and static equilibrium via L-BFGS.
  - Analytic Fréedericksz threshold validation.
  - Extended Jones matrix optics (transmissive and reflective LCoS modes).

- **2D Periodic Grating & Spatial Crosstalk** ([`crosstalk.py`](crosstalk.py)):
  - Periodic boundary conditions along $x$ over period $\Lambda$, resolved across $z \in [0, d]$.
  - Electrostatic fringing fields ($E_x, E_z$) computed spectrally via 1D FFT.
  - Lateral Frank continuum elasticity coupling adjacent columns.
  - Optical Fourier harmonic decomposition: $e^{i k x} \to \sum_n c_n e^{i n k x}$ and diffraction efficiencies $\eta_n = |c_n|^2$.
  - Enabled by default (`crosstalk=True`), with an option to toggle off (`crosstalk=False`) to compare against ideal uncoupled pixels.

- **Interactive PySide6 GUI** ([`lc_gui.py`](lc_gui.py)):
  - Live 3D director field rod viewer.
  - 1D diagnostics: director profile $(\theta, \phi)$, Frank distortion energies, quasi-static $V\text{–}T$ curves, and dynamic switching.
  - Grating diagnostics: 2D electric potential contours $V(x, z)$, phase profile $\Phi(x)$, transmission $T(x)$, and diffraction harmonic bar charts comparing crosstalk vs. ideal.

---

## Spatial Crosstalk & Harmonics ($e^{i k x} \to e^{i n k x}$)

In pixelated liquid-crystal devices (such as LCoS microdisplays and phase-only SLMs), applying a spatially varying voltage pattern—such as a linear phase ramp for beam steering—suffers from **spatial crosstalk**. The resulting phase profile is rounded and distorted, causing light directed into a target spatial frequency $e^{i k x}$ to scatter into higher and negative harmonics $e^{i n k x}$.

### Physics of Spatial Crosstalk

The model captures three coupled physical mechanisms:

1. **Electrostatic Fringing Fields (Laplace Depth Attenuation)**  
   For a periodic electrode surface potential $V(x, 0) = \sum_m \hat{V}_m e^{i m k x}$ on the bottom substrate and ground $V(x, d) = 0$ at the top substrate, each spatial harmonic mode $q_m = m k = 2\pi m / \Lambda$ attenuates across thickness $d$ as:
   $$V_m(z) = \hat{V}_m \frac{\sinh(q_m (d - z))}{\sinh(q_m d)}$$
   This produces both longitudinal ($E_z = -\partial_z V$) and transverse ($E_x = -\partial_x V$) fields. Higher spatial harmonics penetrate only to a characteristic depth $\delta_m \sim 1/(m k) = \Lambda / (2\pi m)$, meaning fine electrode features are smoothed out before reaching the bulk liquid crystal.

2. **Frank Continuum Elastic Coupling**  
   Lateral director gradients contribute elastic distortion energy:
   $$f_{\text{lateral}} = \frac{1}{2} K_{\text{eff}} \left| \frac{\partial \mathbf{n}}{\partial x} \right|^2 \approx \frac{1}{2} K_{\text{eff}} \left[ \left(\frac{\partial \theta}{\partial x}\right)^2 + \cos^2\theta \left(\frac{\partial \phi}{\partial x}\right)^2 \right]$$
   This penalizes spatial harmonic $n$ proportionally to $K (n k)^2$, enforcing an elastic coherence length:
   $$\xi \approx \sqrt{\frac{K}{\varepsilon_0 |\Delta\varepsilon| E_0^2}}$$
   Sharp voltage transitions (such as the $2\pi \to 0$ flyback reset in a blazed grating) are smoothed out into a rounded transition of width $w \sim \mathcal{O}(d)$.

3. **Optical Harmonic Generation & Jacobi–Anger Expansion**  
   The optical field transmitted or reflected through the cell is:
   $$E_{\text{out}}(x) = \sqrt{T(x)} e^{i \Phi(x)}$$
   Even for a single harmonic phase modulation $\Phi(x) = \Delta\Phi \cos(k x)$, the optical field contains all spatial harmonics via the Jacobi–Anger relation:
   $$e^{i \Delta\Phi \cos(k x)} = \sum_{n=-\infty}^\infty i^n J_n(\Delta\Phi) e^{i n k x}$$
   When crosstalk distorts the phase into a blunted profile, the discrete Fourier transform of $E_{\text{out}}(x)$ directly yields the diffraction efficiencies $\eta_n = |c_n|^2$:
   - Depletion of the intended target order ($\eta_{+1}$).
   - Residual leakage into the unmodulated zeroth order ($\eta_0$).
   - Spurious ghost orders at negative and higher harmonics ($\eta_{-1}, \eta_{+2}, \dots$).

---

## Real SLM Behaviour: Accuracy and Limitations

### What the Model Captures Accurately
- **Flyback Blunting**: Smooths the discontinuous $2\pi$ phase reset of blazed gratings, reproducing the characteristic drop in first-order steering efficiency and zeroth-order leakage.
- **Voltage-Dependent Elastic Width**: The transition width scales physically with $\xi(V) \propto 1/E$, broadening at lower voltages and sharpening at higher fields.
- **High-Frequency Cutoff**: Higher-order electrode patterns decay exponentially in $z$, accurately predicting modulation loss for fine gratings ($\Lambda \lesssim 4d$).
- **Direct Fourier Decomposition**: Accurately computes far-field diffraction efficiencies $\eta_n$ for arbitrary periodic profiles (blazed, sinusoidal, binary).

### Physical Approximations & Edge Cases
- **Dielectric Anisotropy Back-Reaction**: The electrostatics are computed from the linear/layered potential rather than self-consistently solving the anisotropic equation $\nabla \cdot (\boldsymbol{\varepsilon}(\mathbf{n}) \nabla V) = 0$. For high-$\Delta\varepsilon$ nematics (e.g. E7, $\Delta\varepsilon \approx 13.8$), field lines in real cells bend slightly toward regions of director reorientation.
- **Pretilt Steering Asymmetry**: Alignment layers enforce a substrate pretilt $\theta_0 \approx 1^\circ\text{--}3^\circ$. In real devices, transverse fringing fields $E_x$ reinforce pretilt for one steering direction and oppose it for the reverse direction, making $+k$ and $-k$ steering efficiencies slightly asymmetric.
- **Reverse-Tilt Disclinations**: Strong lateral fields at sharp pixel gaps can induce localized reverse-tilt domains. The vector angle formulation $(\theta, \phi)$ cannot model topological defect cores (which require a Landau–de Gennes $\mathbf{Q}$-tensor solver).
- **Pixel Backplane Fill Factor**: Metal pixel pads on CMOS backplanes have an unaddressed inter-pixel gap (fill factor $\sim 85\text{--}93\%$). This acts as a fixed amplitude mask, adding diffraction orders at the pixel grid frequency $2\pi/p$.
- **Thin-Film Jones Approximation**: Jones calculus assumes light propagates vertically through each column. When grating periods approach the optical wavelength ($\Lambda \lesssim 4\text{--}6\,\mu\text{m}$), internal diffraction within the LC layer requires full wave solvers (RCWA or FDTD).

---

## Usage

### 1. Interactive GUI
Launch the application from the parent `physics` directory:
```bash
python -m liquid_crystal.lc_gui
```
- **"1D Cell Diagnostics" Tab**: Explore tilt/azimuth profiles, Frank distortion energies, dynamic response to square waves, and $V\text{–}T$ curves.
- **"Spatial Crosstalk & Harmonics" Tab**: View 2D potential contours $V(x, z)$, optical phase profiles $\Phi(x)$, and harmonic bar charts ($e^{i n k x}$).
- **Crosstalk Toggle**: Use the "Enable spatial crosstalk" checkbox in the left panel to switch between realistic crosstalk and ideal uncoupled pixels.

### 2. Python API
```python
import numpy as np
from liquid_crystal import crosstalk as ct
from liquid_crystal import frank_model as fm

# Define cell and optics
cell = fm.CellParams(thickness=2.5e-6, pretilt_bottom=np.radians(2.0), twist=0.0)
optics = fm.OpticsParams(mode="reflective", polarizer_angle=np.pi / 4)

# 1. Grating with spatial crosstalk (enabled by default)
grating_xtalk = ct.GratingParams(
    period=10e-6,
    n_x=32,
    profile_type="blazed",
    v_min=0.5,
    v_max=4.5,
    crosstalk=True,
)
res_xtalk = ct.solve_grating(cell, optics, grating_xtalk)

# 2. Grating without crosstalk (ideal uncoupled pixels)
grating_ideal = ct.GratingParams(
    period=10e-6,
    n_x=32,
    profile_type="blazed",
    v_min=0.5,
    v_max=4.5,
    crosstalk=False,
)
res_ideal = ct.solve_grating(cell, optics, grating_ideal)

# Inspect diffraction efficiencies eta_n = |c_n|^2
for order, eta_x, eta_i in zip(res_xtalk.orders, res_xtalk.efficiencies, res_ideal.efficiencies):
    if abs(order) <= 3:
        print(f"Order {order:+2d}: with crosstalk = {eta_x*100:5.2f}%, ideal = {eta_i*100:5.2f}%")
```

### 3. Running Tests
```bash
python -m pytest
```
Runs 14 unit tests validating analytic Fréedericksz thresholds, Mauguin limits, 1D energy gradients, 2D electrostatic boundary conditions, and spatial harmonic conservation.
