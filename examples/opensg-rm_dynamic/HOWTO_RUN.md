# How to run the dynamic examples, end to end

This is the **operator's guide**: start with nothing, finish with figures. For
*what the benchmark is and why it matters* read [README.md](README.md); for
*the numbers we obtained* read [RESULTS.md](RESULTS.md). This file is only
about the mechanics — which script, in which order, on which machine, and
which line to edit when you point it at a different plate.

Every dynamic example follows the **same five stages**:

| # | stage | where it runs |
|---|---|---|
| 1 | Build the through-thickness 1-D SG and homogenize it to the MSG 8×8 plate law | Linux compute server |
| 2 | Write the Abaqus `.inp` decks (RM shell / conventional FSDT shell / 3-D solid) | Linux compute server |
| 3 | Run the jobs, copy the `.dat` back | Windows Abaqus box, over RDP |
| 4 | Parse the `.dat` (and `.rpt` for full-field work) | Linux compute server |
| 5 | Dehomogenize, compare, plot | Linux compute server |

Stage 6 (optional, ex5 only) adds the full 3-D field VTK comparison and needs
ParaView on a machine with a GPU.

---

## 0. Which example should I copy?

| you want | copy | why |
|---|---|---|
| a transient plate, one model, vs an analytic solution | **ex1** | smallest complete case — one script builds SG *and* deck, one script plots |
| a transient laminate, shell vs solid vs conventional FSDT | **ex2** | the four-way comparison template, single file for all six decks |
| a free-vibration / frequency study | **ex4** | `*FREQUENCY` decks + eigenvalue table + mode images |
| a transient sandwich with **per-time-step 3-D stress recovery** | **ex5** | the full pipeline including field VTKs — this is the reference implementation |

`ex3/` also exists (static Pagano sandwich) but is outside the dynamic scope
and its post-processor has never been run — see §9.

---

## 1. Prerequisites

**Compute server.** `ssh bagla0@msg.ecn.purdue.edu`, repo at
`~/OpenSG-TW-claude`. Always call the environment python explicitly — even the
pure-text post-processors import JAX transitively:

```bash
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex5/1d_sg.py
```

NumPy ≥ 2.0 is required (`np.trapezoid`); the env has 2.3.5.

**Abaqus machine.** RDP to `10.165.18.74` (Windows, Abaqus 2024, license
`iacmi-vlm.ecn.purdue.edu`). Work in `C:\Temp\opensg_dyn`.

**One filesystem, three names.** These are the *same directory*:

```
~/OpenSG-TW-claude          (Linux)
Y:\OpenSG-TW-claude         (your Windows drive letter)
\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude   (UNC, reachable from the RDP session)
```

That is why a deck written on Linux needs no upload, and why a `.dat` copied
over UNC lands straight in the git working tree. (The SMB directory cache can
lag — if `Y:` shows a stale listing, re-check over `ssh`.)

**ParaView** (stage 6 only): the user's *local Windows* pvpython —
`C:\Program Files\ParaView-5.12.0-...\bin\pvpython.exe`. The Linux server has
no headless GL.

---

## 2. Stage 1 — the through-thickness SG and the 8×8 plate law

This is the step that makes the whole thing OpenSG-RM rather than ordinary
Abaqus. The laminate is meshed *through its thickness* (one 5-noded quartic
element per ply) and homogenized into an 8×8 Reissner–Mindlin plate law.

| example | script | writes |
|---|---|---|
| ex1 | `ex1/make_fig5_rm.py` | `ex1_sg.yaml`, `ex1_sg.png` (then the deck too — §3) |
| ex2 | `ex2/make_abaqus_ex2.py` | `ex2_sg.yaml`, `ex2_sg.png` (then all six decks) |
| ex4 | `ex4/make_abaqus_freq.py` | `ex4_<slug>_sg.yaml/.png` ×3 (then the RM decks) |
| ex5 | `ex5/1d_sg.py` | `sandwich_sg.yaml`, `sandwich_sg.png` |

**ex5 is the only one with a separate SG step** — the others fold stages 1 and 2
into one script.

### What to edit for a new laminate

Everything lives in module-level constants at the top of the script:

```python
MATERIAL_DB = {"ge": {"E": [E1, E2, E3], "G": [G12, G13, G23],
                      "nu": [nu12, nu13, nu23], "rho": ...}}
layup = {"mat_names": [...], "thick": [...], "angles": [...]}   # bottom ply FIRST
plate_sg_yaml(yml, layup, MATERIAL_DB, fraction=0.5)
```

All three `layup` lists must be the same length and ordered **bottom to top**.

`fraction` is the single most important argument: it is where the shell
reference surface lives, as a fraction of thickness — **0 = bottom/OML face,
0.5 = mid-surface, 1 = top**. All four dynamic examples use `0.5`, because the
Abaqus S4 mesh sits on the geometric mid-plane. Every depth `z` you later pass
to the recovery is measured from that same plane.

Two library defaults disagree on purpose, and this is a real trap:
`plate_sg_yaml` defaults to `fraction=0.5` but `rm_plate_msg` defaults to
`fraction=0.0`. **Always forward it explicitly**, and always as a *keyword* —
the fourth positional argument of both functions is `n_per_layer`, not
`fraction`.

### What comes out

`rm_plate_msg(...)` returns a dict; the piece you need is `r["ABDG"]`, an 8×8
whose rows and columns are ordered
`e11, e22, g12, k11, k22, k12, 2·g13, 2·g23`:

- `ABDG[:6, :6]` — the classical A/B/D → Abaqus `*SHELL GENERAL SECTION`
- `ABDG[6:, 6:]` — the 2×2 MSG transverse shear → `*TRANSVERSE SHEAR STIFFNESS`
- the off-diagonal 6×2 blocks are identically zero

Check `r["G_msg"] is not None` before indexing `ABDG` — it comes back `None`
when the least-squares-fitted shear compliance is not positive definite, and
nothing raises.

**Sanity check the shear term before going further.** For the ex5 sandwich the
script prints `G11 = 7.7010e6 N/m`. A naive parallel integral ∫G(z)dz would give
4.1377e7 — **5.4× stiffer**. If your number looks like the parallel integral,
the homogenization did not do what you think it did. (Handy cross-check on this
laminate: because G12 = G13 for the faces and the core is isotropic, that
parallel integral equals A66, which the printed 8×8 shows at `g12` — so you can
read both numbers off the same table.)

Section mass is *not* computed by the library. Each example does it inline:

```python
rho_h = sum(material_db[m]["rho"] * t for m, t in zip(mat_names, thick))
```

---

## 3. Stage 2 — write the Abaqus decks

| example | script | writes |
|---|---|---|
| ex1 | `ex1/make_fig5_rm.py` | `ex1_RM_fig5.inp` |
| ex2 | `ex2/make_abaqus_ex2.py` | `ex2_{RM,FSDT,SOLID}_{step,blast}.inp` (6) |
| ex4 | `ex4/make_abaqus_freq.py` → `make_fsdt_freq.py` → `make_solid_freq.py` | `ex4_<slug>_{freq,fsdtfreq,solidfreq}.inp` (9) |
| ex5 | `ex5/make_abaqus_dyn.py` | `sandwich_{RM,SOLID,FSDT}_{step,blast}.inp` (6) |

Run them in the order shown — for ex4 the FSDT and solid writers *import*
`make_abaqus_freq` for their geometry and materials, so it must exist and be
importable first.

### The knobs, by category

**Geometry and mesh**

| symbol | meaning | ex5 value |
|---|---|---|
| `A` (`AX`,`BY`) | plate side(s) [m] | 1.524 |
| `NX` (`NEX`,`NEY`) | in-plane elements per side | 20 |
| `NZC` / `NZP` | solid elements through the core / per ply | 8 / — |

**Load**

| symbol | meaning | ex5 value |
|---|---|---|
| `Q0` | peak pressure [Pa] | 68.9476e6 |
| `sinsin(i,j)` | the *spatial* shape, evaluated at each element centre | sin(πx/a)·sin(πy/b) |
| `CBLAST` | blast decay c in e^(−ct) [1/s] | 330 |
| `T1` | step cut-off time [s] (ex2 only; ex5's step is held) | 0.006 |

To change the load *footprint* — uniform, patch, point — replace the
`sinsin` function (ex1 instead uses a centroid-in-patch test in its `*DLOAD`
loop). To change the load *history*, edit the `*AMPLITUDE` table.

**Time**

| symbol | meaning | ex5 value |
|---|---|---|
| `DT` | fixed implicit increment [s] | 5.0e-5 |
| `TTOT` | window [s] | 0.02 |
| `*STEP INC` | increment cap, `int(2*TTOT/DT)` | 800 |

**Output requests — these define what the post-processors can see**

- `*NODE PRINT, NSET=NCEN, FREQUENCY=1` — the centre-node deflection, every increment
- `*EL PRINT, ELSET=PATCH*, FREQUENCY=1` — `SF, SM` and (a **separate** block) `COORD` on the 2×2 recovery patches
- solid: `*EL PRINT, ELSET=COL*, FREQUENCY=10, POSITION=CENTROIDAL` — stress columns every 10 increments

The `SF`/`COORD` split into two `*EL PRINT` blocks is load-bearing: the parser
keys them as two separate tables. Merging them breaks the reshape.

### Deck traps that produce wrong numbers with no error

- **The 21 general-section terms are the UPPER TRIANGLE, COLUMN BY COLUMN** —
  `[AB[i,j] for j in range(6) for i in range(j+1)]`. A row-major flatten still
  runs and silently describes a different laminate.
- **`DENSITY` on `*SHELL GENERAL SECTION` is mass per unit AREA** (ρh), not ρ.
  Passing ρ would put every frequency out by ~22×.
- **`*TRANSVERSE SHEAR STIFFNESS` takes K11, K22, K12** — that is
  `G2[0,0], G2[1,1], G2[0,1]`, not a row of the matrix. For ±45 stacks K12 is
  genuinely nonzero and dropping it changes the answer.
- **`NALL, 6, 6`** (drilling fixed) is *required* with a general section on a
  flat plate — the section carries no drilling stiffness, so leaving DOF 6 free
  injects near-zero-energy modes.
- **Solid pressure is `P2` on the top layer, which acts in −z, while shell `P`
  acts +z.** Every downstream solid reader therefore negates U and S. If you
  change the solid load face you must remove those negations.
- **Use `C3D8I`, not `C3D8`,** for the solid: with one element per thin ply the
  aspect ratio reaches ~48:1 and plain C3D8 shear-locks.
- **ex2 only:** `write_fsdt()` and `write_solid()` hard-code the ply angles
  `(0, 90, 0)` and thickness `H/3` instead of reading `LAYUP`. Changing `LAYUP`
  alone changes only the OpenSG-RM deck — the benchmarks silently keep the old
  stack. Change all three places together.
- **ex4 only:** `make_solid_freq.py` re-declares `NEX, NEY` locally. Refining
  the shell mesh leaves the solid at 24×12 and you compare different meshes.

---

## 4. Stage 3 — run Abaqus on the Windows box

The practised pattern is a **one-shot batch file written on the Linux side and
executed on the Windows box over the same share**. Write it into
`~/claude_tmp/` (= `Y:\claude_tmp`) and it is immediately visible as
`\\roger.ecn.purdue.edu\bagla0\claude_tmp\`.

### The template

```bat
cd /d C:\Temp\opensg_dyn
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\sandwich_*.inp" .
for %%j in (sandwich_RM_step sandwich_RM_blast sandwich_SOLID_step sandwich_SOLID_blast) do call abaqus job=%%j cpus=4 interactive ask_delete=OFF
copy /Y sandwich_*.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\Abaqus_results\"
dir sandwich_*.dat
```

Four sections: `cd`, copy the decks in, run, copy the results back. The trailing
`dir` is the success check — it prints byte counts and timestamps of what
actually came back.

Then, on the Windows box in a plain **Command Prompt** (not PowerShell, not CAE):

```
\\roger.ecn.purdue.edu\bagla0\claude_tmp\myjob.bat
```

### The percent-sign rule

- **inside a `.bat`**: `for %%j in (...) do call abaqus job=%%j ...`
- **typed at an interactive prompt**: `for %j in (...) do call abaqus job=%j ...`

The single-percent form pasted into a `.bat` expands to nothing and the loop
runs zero jobs. (`ABAQUS_RUN_COMMANDS.md` quotes the interactive form
throughout — correct for what it documents, wrong if you paste it into a file.)

`call` is mandatory in both forms: `abaqus.bat` is itself a batch file, so
without `call` the loop stops after the first job. `ask_delete=OFF` overwrites
old results without the y/n prompt, which is required when the box is driven
non-interactively.

### If the batch also needs a CAE post-step

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\odb_rpt.py" .
call abaqus cae noGUI=odb_rpt.py
timeout /t 120 /nobreak
copy /Y *.rpt "\\roger...\ex5\Abaqus_results\"
```

The `timeout` is a deliberate flush wait before the copy — don't delete it. Note
`abaqus cae noGUI` runs **detached** on Windows and prints nothing to the
console; check for output files, not console text.

### Confirm and collect

Each job must print `Abaqus JOB <name> COMPLETED`. Observed at `cpus=4`:
RM/FSDT shells ~40 s, 20×20×16 C3D8I solids ~5.5 min, ex4 `*FREQUENCY` shells
~10 s.

Only lightweight artifacts travel back — the `.dat` history prints, the `.rpt`
field reports, and CAE-rendered PNGs. **The `.odb` stays on the Windows box**,
which is why mode rendering and any odb re-query can only be done there.

Create `Abaqus_results\` on the share first if it doesn't exist — otherwise
`copy` writes a *file* by that name.

### Rules learned the hard way

- **Never run Abaqus inside a OneDrive folder** — sync locks the `.lck`/scratch
  files and the job dies mid-solve. `C:\Temp\opensg_dyn` is a plain local folder
  on purpose.
- **UNC to roger works from inside the RDP session; `\\tsclient\...` does not**
  (the session has no local-drive redirection).
- **RDP paste race:** if a typed command doesn't appear, click the cmd window
  *title bar* first and retype. A swallowed trailing Enter is common — press
  Enter again if a command sits unexecuted.
- CAE scripts use bare `try/except` and print `FAIL` while still exiting
  cleanly. **A zero exit code proves nothing** — read the log, or check the
  files.

---

## 5. Stage 4 — what the parsers expect from the `.dat`

For everything except the full-field work, **the `.dat` is all you need** — all
history prints are in it.

The shared parser is `read_elprint_tables` in
`examples/yu2003/recover_6p2.py` (not in this folder; the scripts `sys.path`
insert it). Things it and the local readers care about:

- **Time alignment is done from the END.** A `*NODE PRINT` table that Abaqus
  reports as "ALL VALUES IN THIS TABLE ARE ZERO" contributes no row, so the code
  uses `t = t_all[len(t_all) - len(rows):]`. Aligning from the front silently
  shifts every history.
- **Solid times are parsed from `STEP TIME COMPLETED`,** never assumed as k·Δt —
  the solid job runs automatic increments (~430, non-uniform) unless you force
  `*DYNAMIC,DIRECT`.
- **Abaqus always prints one extra final-increment table.** `make_curves.py`
  and `make_curves9.py` handle it; `recover_dyn.py` does not (harmless there).
- **Columns are selected by LABEL,** never by position — Abaqus column order is
  version-dependent.
- **`SF1..SF5, SM1..SM3` = N11, N22, N12, Q1, Q2, M11, M22, M12.** So
  `[0,1,2,5,6,7]` is the 6-vector driving the plate strains and `[3,4]` are the
  transverse shears.

For the `.rpt` files (stage 6 only), two column traps:

- the **S report has 16 columns** — element, int-pt, then **ten invariants**,
  then the six components at `row[10:16]`
- the **U report has 5 columns** — node, then **`U.Magnitude`**, then `row[2:5]`

Both mis-shifted the first parses. And `writeFieldReport` **inherits the
viewport coordinate transformation**, so the report is only in the global frame
because `odb_rpt.py` sets a `DatumCsysByThreePoints` + `USER_SPECIFIED` CSYS
first. Skip that and you get ply-local stresses with no warning. (There is no
`transformationType=GLOBAL` constant.) `vp.setValues(displayedObject=odb)` is
also required before the call or it fails outright.

---

## 6. Stage 5 — post-process and plot

| example | run, in this order | produces |
|---|---|---|
| ex1 | `plot_fig5.py` | `ex1_fig5_opensg_rm.png` (RM vs the analytic Mindlin modal solution, computed in-script) |
| ex2 | `compare_ex2.py` (optionally `reddy_hsdt_navier.py` first as an anchor check) | `ex2_w_history_{step,blast}.png`, `ex2_sx_history_*.png`, `ex2_results.dat` |
| ex4 | `collect_freq.py`, then `~/claude_tmp/finish_modes.sh` for the mode composites | `ex4_freq_table.dat`, `modes/compare_<slug>_m<i>.png`, `ex4_modes_table.dat` |
| ex5 | `recover_dyn.py` → `make_curves.py` → `make_curves9.py` | histories, profiles, `dyn_<kind>.dat`, `curves/` |

### The health checks — read these before trusting anything

`recover_dyn.py` prints, per pulse:

- **σ33 top-face closure** (target 1.0): 1.0092 step / 1.0303 blast. This is the
  through-thickness equilibrium integral closing on the applied pressure — the
  single best per-time-step reliability readout.
- **dynamic shear amplification** Q_measured/Q_quasistatic ≈ 1.39 at the peaks
  (1.0 in the static limit).

### The two dynamic corrections that must not be "simplified away"

1. **σ_a3 rescale.** The strain-gradient chain reproduces only the *quasi-static*
   shear (Q = M,x). Dynamically, the SG through-thickness distribution is
   rescaled to carry the **measured** Q1/Q2 (`SF4`/`SF5`).
2. **σ33 by the dynamic momentum integral.**
   `dσ33/dz = ρ(z)·ẅ − σ13,1 − σ23,2`. The inertia term is 40–100 % of q₀ here;
   drop it and the closure check fails immediately.

### Editing the plots

`curves/README.md` documents the ex5 curve set specifically. In general:

- **legend text** — the literal strings `"Abaqus 3-D solid"`, `"OpenSG-RM"`,
  `"Abaqus FSDT"` in the plotting loops
- **axis labels** — the fourth entry of each `SPEC` row (raw LaTeX)
- **styles** — `STY_SOL` (dashed black ○), `STY_RM` (orange —□),
  `STY_FS` (green -.△), `STY_NAY` (blue ★, ex2). `markevery=(offset, stride)`
  staggers markers so curves don't read as polluted.
- **time window** — `ax.set_xlim(...)` in `one_plot()`
- house rule: **no figure titles** — the LaTeX caption is the title; the legend
  sits outside the axes on the right.

### Station and depth conventions

Both models must be read at the **same depth**, and the solid only offers its
integration points. So the "top surface" curves are actually sampled at the
top-ply centroid `h/2 − t_ply/2` in *both* models even where the axis label
reads `h/2`. Comparing a layer *surface* (where σ_a3 = 0 exactly) against a
solid Gauss depth is what made the first field renders look "completely
different".

Likewise the "x-edge" station is really at `x = dx/2` — `PATCHX`/`COLX` are the
first cells off the edge — worth 0.3 % on the cosine.

### Component-order trap

The recovery returns 3-D **Voigt order (11, 22, 33, 23, 13, 12)**, so
`σ13` is index **4** and `σ23` is index **3**. The Abaqus `.rpt` keeps
**Abaqus order (11, 22, 33, 12, 13, 23)** where `S12 = 3, S13 = 4, S23 = 5`.
Copying index constants between `rm_field_vtk.py` and `rpt_to_vtk.py` swaps
σ12 and σ23. Index by *name*, not by number.

---

## 7. Stage 6 (optional, ex5) — the full 3-D field comparison

```
make_field_deck.py                       # derive the two snapshot decks
  → [Abaqus] run sandwich_RM_field, sandwich_SOLID_field
  → [Abaqus CAE] odb_rpt.py              # writes the S and U .rpt
rm_field_vtk.py                          # RM .dat → sandwich_rm_field.vtk
rpt_to_vtk.py                            # solid .rpt → sandwich_solid_field.vtk
  → [pvpython] render_fields.py <vtk> <outdir> <title> ex5_step_t2p85ms
  → montage the pairs into field_images/compare/ BY HAND
```

Three things make this work, and each was a bug first:

1. **Identical snapshot time.** `make_field_deck.py` converts the solid job to
   `*DYNAMIC,DIRECT` with fixed 50 μs increments and puts field frames every 57
   increments, so frame 1 lands *exactly* on 2.85 ms. On automatic increments
   the nearest frame was 3.52 ms — wrong amplitude *and* phase everywhere.
2. **Identical sampling lattice.** Both converters build the same 40×40×32
   Gauss lattice at **ply-resolved** depths (`TLAY = [t_ply]×4 + [t_core/8]×8 +
   [t_ply]×4`). A uniform h/16 lattice put the outer station in the wrong ply
   (σ11 3.2 vs 27 GPa).
3. **The `.rpt` column layouts** of §5.

The dump cadence `57` is duplicated in `make_field_deck.py` *and*
`rm_field_vtk.py` (`tstar = 57*DT*(kd+1)`); change one and you must change the
other. Worse, both `*DYNAMIC`/`*OUTPUT` edits in `make_field_deck.py` are
**literal string replacements** against `%g`-formatted numbers — change `DT` or
`TTOT` upstream and the replace silently does nothing.

`render_fields.py` rescales each component to **its own** data range, so the two
models' images are *not* on a common colour scale. Read the colorbar numbers,
not the colours.

---

## 8. Adapting to a new plate: the duplication checklist

Case constants are deliberately repeated across scripts rather than imported.
Change one and the others go stale **silently**. Walk this list:

**ex5** — `H`, `TPLY`, `TCORE`, `MATERIAL_DB`, `layup` in `1d_sg.py`;
then `A`, `NX`, `Q0`, `DT`, `TTOT`, `CBLAST`, `NZC` in `make_abaqus_dyn.py`;
then the *same* `A`, `NX`, `Q0`, `CBLAST`, `NZC`, `P = π/A` again in
`recover_dyn.py`; `A`/`NX`/`Q0`/`DT` again in `rm_field_vtk.py` (where `DT` is
assigned **twice**); `NX`/`NZT`/`A`/`H` again in `rpt_to_vtk.py`. Also hard-coded
in `make_curves9.py`: the densities `1500`/`130` and core layer index `4` in
`rho_int`, and the literal `5.0e-5` time step in the acceleration difference.

**ex2** — everything in `make_abaqus_ex2.py`, **plus** the duplicated
`(0, 90, 0)` and `H/3` inside `write_fsdt()`/`write_solid()`, **plus** the same
constants again in `compare_ex2.py`, **plus** `A`/`H`/`NX`/`NZT` in both
`*_field_vtk` converters. `nzt = 3*NZP` hard-codes a three-ply stack.

**ex4** — `make_abaqus_freq.py` is the source of truth (`AX`, `BY`, `TPLY`,
`TAL`, `NEX`, `NEY`, `MATERIAL_DB`, `LAYUPS`, `SLUGS`, `TABLE5`); the FSDT
writer imports from it, but `make_solid_freq.py` **re-declares `NEX, NEY`**.
`odb_modes.py` keeps its own hard-coded `SLUGS` list.

**ex1** — `make_fig5_rm.py` and `plot_fig5.py` duplicate every geometry and
material constant, and `nu`/patch-size appear a *third* time inside
`mindlin_analytic()`. Also: `NEY` is hard-coupled to the hand-written 6+8+6
non-uniform `YL` node-line split — change one and you must change the other.

---

## 9. Known-broken and known-missing steps

Stated plainly so nobody loses an afternoon:

1. **`ABAQUS_RUN_COMMANDS.md` §4 is broken.** It calls
   `ex5/reddy_hsdt_navier.py`, which was **moved to `ex2/`**. It also references
   `ex4/ex4_free_vibration.py`, which does not exist. Same for
   `~/claude_tmp/run_recover_dyn.sh`.
2. **`README.md` overstates `collect_freq.py`** — it is table-only now, it does
   not write `ex4_freq_*.png`.
3. **ex2's field phase is not comparable yet.** The `*OUTPUT` string replace in
   `make_field_deck_ex2.py` is a **no-op** (`write_solid()` never emits an
   `*OUTPUT` block), so the solid odb frames land every 0.5 ms while
   `odb_rpt_ex2.py` targets 0.75 ms — the two snapshots are at **different
   instants**. Fix by adding `*OUTPUT, FIELD, FREQUENCY=15` / `U, S` before
   `*END STEP` in `write_solid()` before comparing anything. Neither
   `ex2_solid_field.vtk` nor `ex2_rm_field.vtk` has ever been generated.
4. **`ex5/curves/make_curves.py` and `make_curves9.py` are non-runnable
   duplicates** kept for provenance. Both resolve paths from `__file__`, so run
   from `curves/` they look for `curves/Abaqus_results/` and fail. **Edit and
   run the `ex5/` originals.**
5. **No script builds `field_images/compare/*.png`.** Those side-by-side
   montages — which `report_ex5.tex` hard-codes by name — were stitched ad hoc.
6. **Four scripts that produce committed artifacts live outside the repo**, in
   `~/claude_tmp/`: `odb_modes.py` and `finish_modes.sh` (ex4 mode images and
   composites), `odb_rpt_fsdt.py` (ex5 FSDT report, which never made it back),
   and `composite_fields.sh`. Two warnings about the last two: `finish_modes.sh`
   ends in an **unconditional `git add -A` / commit / push**, and
   `composite_fields.sh` does `rm -rf field_images` then untars an **untracked**
   `field_images.tgz` — so there is currently no reproducible in-repo path to
   `field_images/OpenSG-RM/` and `/3D_FEA_Abaqus/`. Re-render them with
   `render_fields.py` instead.
7. **ex2's `ex2_SOLID_step_*.png` images come from `ex5/odb_images2.py`,** not
   from anything in `ex2/` — its `CASES` list renders both folders' solids.
8. **`compare_ex2.py` no longer plots the analytical Khdeir–Reddy curve** even
   though its docstring and `report_ex2.tex`'s peak table still mention it.
9. **`ex3/` is undocumented and unfinished** — `make_abaqus_ex3.py` and
   `compare_ex3.py` exist and the two `.dat` results are committed, but
   `ex3_table.dat` has never been produced and `ABAQUS_RUN_COMMANDS.md` has no
   ex3 section.

---

## 10. Traps that give wrong numbers without any error

Consolidated, because every one of these cost real debugging time:

| trap | symptom |
|---|---|
| ABD 21-term list flattened row-major | runs fine, different laminate |
| `DENSITY` given as ρ instead of ρh | frequencies ~22× high |
| `fraction` passed positionally | it lands in `n_per_layer` |
| `rm_plate_msg` default `fraction=0.0` vs yaml `0.5` | reference plane at the wrong face |
| Voigt (…,23,13,12) vs Abaqus (…,12,13,23) | σ12 and σ23 swapped |
| solid `P2` (−z) vs shell `P` (+z) | sign-flipped solid results |
| `.rpt` invariant / Magnitude columns | first components read as invariants |
| history aligned from the front, not the end | whole history shifted in time |
| solid Δt assumed uniform | automatic increments are non-uniform |
| ply-frame rotation in `recover_dyn.py` | only 0°/90° handled; any other angle silently unrotated |
| `collect_freq.py` reading `toks[2]` | rad/s instead of Hz (6.283× out) |
| sampling a layer surface vs a Gauss depth | σ_a3 = 0 exactly at surfaces |
| uniform-h/16 through-thickness lattice | outer station lands in the wrong ply |
| `for %j` inside a `.bat` | loop runs zero jobs |
| missing `call` in the loop | only the first job runs |
| Abaqus run inside OneDrive | job dies mid-solve on a locked `.lck` |

---

*Companion documents:* [README.md](README.md) (what the benchmark is) ·
[RESULTS.md](RESULTS.md) (the numbers) ·
[ABAQUS_RUN_COMMANDS.md](ABAQUS_RUN_COMMANDS.md) (the literal commands used —
but see §9.1) · [ex5/curves/README.md](ex5/curves/README.md) (the ex5 curve set).
