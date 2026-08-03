# opensg-rm_dynamic — the Nayak–Shenoi–Moy transient sandwich benchmark

> **Want to run it, not read about it?** → **[HOWTO_RUN.md](HOWTO_RUN.md)** is
> the start-to-finish operator's guide: SG mesh → Abaqus deck → remote run →
> `.dat` extraction → post-processing, naming the script for every step and the
> constants to change for a new plate. This file explains *what the benchmark
> is*; [RESULTS.md](RESULTS.md) reports *what we measured*.

**The benchmark case** (all digits quoted from the paper, PDF in
`...\Reissner-mindlin-transverse_Shear_msg\opensg_rm dynamic\`):

A.K. Nayak, R.A. Shenoi, S.S.J. Moy, *"Transient response of composite
sandwich plates"*, **Composite Structures 64 (2004) 249–267 — Example 5**:

| item | value (paper) |
|---|---|
| plate | square, simply supported, a/h = 10, h = 0.1524 m → a = b = 1.524 m (h from their Example 2, which Example 5 inherits loads/geometry conventions from) |
| layup | **(0/90/0/90/core)s** — eight graphite/epoxy plies symmetric about a PVC foam core; 2h_f/h = 0.05 is BOTH face sheets, so each of the eight plies is (0.05/8) h = 0.00625 h = 0.9525 mm and the core is the remaining 0.95 h = 144.78 mm |
| faces | E_L = 128 GPa, E_T = 11.0 GPa, G_LT = G_13 = 4.48 GPa, G_23 = 1.53 GPa, ν = 0.25, ρ = 1500 kg/m³ (Crawley graphite/epoxy) |
| core | **HEREX C70.130 PVC foam**: E_c = 103.63 MPa, G_c = 50 MPa, ν = 0.32, ρ = 130 kg/m³ |
| load | q(x,y,t) = q₀ F(t) sin(πx/a) sin(πy/b), q₀ = 68.9476 MPa; F(t) = **step** and **explosive blast e^(−330 t)** (their sine/triangular variants trivially added) |
| time | Δt = 50 μs (their accepted step), window 0.02 s |
| their results | center deflection w(t) for the four pulses (their Figs. 12–13) + parametric sweeps (Figs. 14–15) |
| their model | refined third-order (Reddy-type) theory, custom C⁰ assumed-strain FE, Newmark |

Their Example 2 (0/90/0 laminate, a = 5h, same pulse family) carries the
**Khdeir–Reddy analytical solution** — the analytic anchor of the whole
suite; their Example 3 is the static Pagano sandwich check (which our
static archive already covers to 0.4–0.9 %).

## Why this benchmark lets OpenSG-RM prove superiority

1. **Accuracy** — their transient outputs stop at the *deflection*; the
   sandwich through-thickness stresses are never shown because a smeared
   third-order theory cannot deliver reliable interlaminar fields for a
   9-layer soft-core section. OpenSG-RM recovers the full 3-D state
   (face–core σ13/σ23, interface σ33, ply-frame stresses) each time step —
   judged against the **Abaqus 3-D solid** of the same problem, an
   evidence level their paper never had.
2. **Reliability** — no shear-correction factors (as their HSDT), *plus*
   machine-exact face tractions with the load ladders, interface-continuous
   transverse shear by construction, and a σ33 that closes the equilibrium
   integral — each checkable per time step.
3. **Usage** — their approach requires bespoke C⁰ HSDT elements in a custom
   code. OpenSG-RM upgrades a *plain Abaqus S4 model* (one general-section
   line pair from a 1-D SG homogenization) and post-processes standard
   SF/SM output — no special elements anywhere.
4. The double-sine load is the plate's Navier mode, so the recovery
   gradients at the center/edge stations are closed-form — the cleanest
   possible dynamic-recovery demonstration; the deck's patch prints give the
   same information redundantly for cross-checking.

Comparison layers: (a) center w(t) vs their Figs. 12–13 (literature check) —
and vs Khdeir–Reddy analytically on their Example 2; (b) all recovered
through-thickness fields vs the Abaqus solid per instant and as time
histories; (c) DOF/wall-time ratio shell vs solid.

## Files

Per-case layout: everything for the transient Ex.5 lives in `ex5/`
(.py, .yaml, .inp, Abaqus `.dat`, Reddy outputs, all figures) and everything
for the free-vibration Ex.4 in `ex4/`.

| file | content |
|---|---|
| `ex5/1d_sg.py` → `ex5/sandwich_sg.yaml/.png` | the 9-layer 1-D SG (5-noded elements, one per layer incl. the core); prints the 8×8 (D₁₁ = 3.0005e6, G₁₁ = 7.7010e6) and section mass 30.2514 kg/m² |
| `ex5/make_abaqus_dyn.py` | writes the four decks below from the SG |
| `ex5/sandwich_RM_step.inp`, `ex5/sandwich_RM_blast.inp` | OpenSG-RM S4 plate (20×20), SS-1, double-sine load, per-increment center-U + patch SF/SM/COORD prints |
| `ex5/sandwich_SOLID_step.inp`, `ex5/sandwich_SOLID_blast.inp` | 3-D benchmark: 20×20×16 C3D8I (one element per face ply, 8 through the core), same everything; center U(t) + column stresses at 0.5 ms cadence |
| `ex5/Abaqus_results/*.dat` | the four job results copied back from the Abaqus machine |

## Running (on the Abaqus machine)

```
abaqus job=sandwich_RM_step     interactive
abaqus job=sandwich_SOLID_step  interactive
abaqus job=sandwich_RM_blast    interactive
abaqus job=sandwich_SOLID_blast interactive
```

Copy the four job `.dat` files back to `Abaqus_results/` (exact commands:
`ABAQUS_RUN_COMMANDS.md`), then run the post-processing chain — **all
results and the full narrative are in [RESULTS.md](RESULTS.md)**:

| script | output |
|---|---|
| `ex5/recover_dyn.py` | individual three-way figures, each with its own legend: `w_history_*`, `profile_{s13,s23,s33,s11}_*`, `iface_s13_*` (+ `dyn_*.dat` numbers) |
| `ex2/reddy_hsdt_navier.py` (moved from `ex5/`) | the ANALYTICAL Reddy-TSDT (= Nayak's theory) + RM Navier responses and the Ex.2 Table-3 anchor check. Call `set_case('ex2')` first — the module's default state is the Ex.5 sandwich. `main()` prints; it writes nothing to disk |
| `ex4/make_abaqus_freq.py` | the three SG yamls + the three `ex4_*_freq.inp` Abaqus `*FREQUENCY` decks (S4 + MSG ABDG general section, clamped cantilever) |
| `ex4/collect_freq.py` | `ex4_freq_table.dat` — Table-5 literature data (Crawley experiment/FEM, Nayak Reddy-HSDT FE) + the OpenSG-RM column and two %-error columns. Table only; it does not draw figures |

## Supporting literature (downloaded alongside the benchmark)

- Nezami & Akhtar, *Arch. Appl. Mech.* (2026): higher-order zigzag theory
  benchmarked against 3-D elasticity — the current competitor-method
  standard our comparison format matches.
- D. Li, *Arch. Comput. Methods Eng.* (2021): layerwise-theory review —
  the survey against which "accuracy of layerwise, cost of ESL" is framed.
- Giffin et al., *IJNME* (2024): layered solid FE with interlaminar
  enhancement — the state-of-the-art solid-side alternative (cost
  comparison point).
- Wanchoo et al., *J. Sandwich Struct. Mater.* (2024): naval sandwich
  blast/implosion review — the realistic-loading context.
