# `abaqus_inp.py` — what it writes into the deck, and why

`abaqus_inp.py` turns the homogenized plate law into a plain Abaqus S4 model:

```
layup_db.yaml + 1dsg.yaml  ->  layup_db_abaqus.inp
```

Nothing in the deck knows the laminate is a laminate. There are no plies, no
`*ORIENTATION` cards and no shear correction factor anywhere — the entire
constitutive description is one general-section block. That is the whole point
of the OpenSG-RM route, and it is why the deck can be run by anyone with a
stock Abaqus licence and no special elements.

Run it, then submit the job:

```bash
python abaqus_inp.py
```

```bash
abaqus job=layup_db_abaqus interactive
```

---

## The keywords it emits, in order

### `*HEADING`
One title line naming the source database and the model number, so a deck found
on its own can be traced back to the YAML that produced it.

### `*NODE`
The grid, x fastest:

```
n(i, j) = 1 + i + (NX + 1) * j
```

with coordinates `(i·dx, j·dy, 0)`, `dx = a/NX`, `dy = b/NY`. The plate lies
in the z = 0 plane, which is the **reference surface** — the same plane the SG's
`fraction` put x₃ = 0 on. With `fraction: 0.5` that is the laminate
mid-surface, so the shell sits where the SG says it sits. Change `fraction`
in `layup_db.yaml` and the physical meaning of z = 0 moves with it.

### `*ELEMENT, TYPE=S4, ELSET=EALL`

```
e(i, j) = 1 + i + NX * j
```

Connectivity is `n(i,j), n(i+1,j), n(i+1,j+1), n(i,j+1)` — **counter-clockwise
seen from +z**, so every element normal points +z. That ordering is what fixes
the sign of the pressure load (below), so do not reorder it casually.

`S4` is the fully integrated four-node shell (four integration points). `S4R`
would work structurally but has one integration point, which breaks any
downstream patch-fit recovery that assumes 4 points per element.

### `*NSET`
Five sets: `NX0`, `NXA` (the x = 0 and x = a edges), `NY0`, `NYB` (y = 0 and
y = b), `NALL` (everything, via `GENERATE`), and `NCEN` (the single centre node,
the deflection probe). Edge sets are written 12 ids per line — Abaqus's limit
is 16.

### `*SHELL GENERAL SECTION, ELSET=EALL, DENSITY=<ρh>`
The 6×6 ABD, as **21 numbers, 8 per line**.

> **The single easiest way to get a silently wrong answer.** Abaqus wants the
> upper triangle read **column by column**: D11, D12, D22, D13, D23, D33, D14,
> … The code produces exactly that with
> `[AB[i, j] for j in range(6) for i in range(j + 1)]`.
> A row-major flatten still runs, still converges, and describes a *different
> laminate*. Nothing warns you.

Row/column order is `e11, e22, g12, k11, k22, k12` — engineering shear strain,
matching Abaqus's own `N11, N22, N12, M11, M22, M12` section rows.

> **`DENSITY` is mass per unit AREA** (ρh), not ρ. Here 30.2514 kg/m², summed
> from each material's `rho` in `layup_db.yaml`. Passing ρ instead would be
> ~490× off and every frequency would come out ~22× too high. Abaqus cannot
> infer the mass from a general section — there are no plies for it to add up.

### `*TRANSVERSE SHEAR STIFFNESS` — only when `model: 1`
Three numbers, in the order **K11, K22, K12**:

```
7.700977e+06, 7.542371e+06, -9.739481e-09
```

That is `G2[0,0], G2[1,1], G2[0,1]` from the MSG 2×2 block — Abaqus's argument
order, not a row of the matrix. On an anisotropic stack K12 is genuinely
non-zero and dropping it changes the answer.

With `model: 0` the card is omitted entirely and Abaqus supplies its own
transverse shear. That is what "classical" means at the deck level: you get the
ABD and Abaqus's default shear rather than the MSG series-compliance value.
On this sandwich the difference is large — the MSG G11 is 7.70e6 N/m against a
naive parallel ∫G(z)dz of 4.14e7, a factor of 5.4.

### `*BOUNDARY`
Two different things share this block, and it is worth keeping them apart.

**The support condition (SS-1)** — applied on the four edge node sets only:

| card | meaning |
|---|---|
| `NX0, 2, 3` / `NXA, 2, 3` | on the x-edges, v = w = 0 |
| `NY0, 1, 1` / `NYB, 1, 1` | on the y-edges, u = 0 |
| `NY0, 3, 3` / `NYB, 3, 3` | on the y-edges, w = 0 |

Each card is `nset, first_dof, last_dof` — a **range**, not a list, which is why
`u = 0` and `w = 0` on the y-edges need two separate cards rather than one.

This is *soft* simple support: the edge rotations are free. Fixing the
tangential rotation as well (the hard/Navier condition) stiffens a thick plate
by several percent — that mattered in ex1 and would matter here if the deck
were ever compared against a Navier solution.

**The drilling restraint** — `NALL, 6, 6`, applied to **every node, interior
ones included**:

> This is **not part of the simply-supported condition**. It is a numerical
> regularisation of the flat-shell formulation, and it happens to be written in
> the same `*BOUNDARY` block only because that is where DOF constraints go.

DOF 6 is rotation about the shell *normal* — the "drilling" rotation, a
screwdriver twist in the plane of the plate, as opposed to DOF 4 and 5 which
tilt the normal and are what bending actually uses.

No plate or shell theory assigns strain energy to that rotation. The membrane
strains come from derivatives of u and v, the bending curvatures from w and the
two tilt rotations; θ_z appears in none of them. So the element stiffness has an
empty row and column for DOF 6.

On a **curved or folded** shell this is self-curing: neighbouring facets have
different normals, so what is a drilling rotation for one element is a bending
rotation for its neighbour, and the assembled stiffness is non-singular. **Our
plate is perfectly flat** — every element normal is +z, every node's DOF 6 is
the same unconstrained direction, and nothing anywhere in the mesh restrains it.
Left free, the global stiffness matrix is singular by one DOF per node. In a
static step Abaqus reports numerical singularities; in a `*FREQUENCY` step
(ex4) the zero-energy modes appear as spurious near-zero frequencies mixed into
the real ones.

Fixing it is free, and that is the important part: because θ_z carries no strain
energy and no applied load does work through it, the constraint reactions are
identically zero and the computed response is unchanged. You are removing a
singularity, not adding a support.

The one caveat: this is only harmless *because the plate is flat*. Copy this
deck to a curved or folded shell and `NALL, 6, 6` would suppress a rotation that
genuinely carries bending in the adjacent facets — there you delete the card and
let the geometry do the job.

### `*STEP, NAME=PULSE, INC=<2·TTOT/DT>`
The increment cap is set to twice the nominal count so a cutback cannot abort
the job.

### `*DYNAMIC`
Four numbers: **initial Δt, total time, minimum Δt, maximum Δt**.

```
5e-05, 0.02, 5e-09, 5e-05
```

Setting *maximum* = `DT` is what pins the fixed increment. Without it Abaqus
runs automatic incrementation, the increments come out non-uniform, and any
post-processor that assumes t = k·Δt is silently wrong — you have to parse
`STEP TIME COMPLETED` instead. (That exact bug put a field snapshot at 3.52 ms
instead of 2.85 ms earlier in this suite.)

### `*DLOAD` — the pressure load
One data line per loaded element:

```
<element id>, P, <magnitude>
```

**`P` is not a variable of ours.** It is Abaqus's distributed-load *type label*
for a uniform pressure on a shell element face — the shell counterpart of the
solid's `P1`…`P6` face labels.

How the magnitudes are built:

1. **The load we want** is the smooth double sine
   `q(x,y) = q0 sin(πx/a) sin(πy/b)`, the plate's first Navier mode. That
   choice is deliberate: the mode shape is its own response shape, so the
   strain gradients the 3-D recovery needs are available in closed form.

2. **Abaqus `P` is uniform over an element** — one number per element, not a
   field. So the continuous q(x,y) is applied as a piecewise-constant
   staircase, one value per element.

3. **Sampled at the element centre**, `(i+0.5)dx` by `(j+0.5)dy` — the midpoint
   rule. Because a half-period of a sine is *concave*, the midpoint rule
   **over**-estimates. Measured on the 20×20 deck:

   | | value |
   |---|---|
   | staircase total Σ q·dA | 6.503430e7 N |
   | exact q₀(2a/π)(2b/π) | 6.490069e7 N |
   | error | **+0.21 %** |

   The excess shrinks as the mesh refines. It is acceptable here only because
   every model being compared is built by this same loop, so the excess is
   common to all of them and cancels out of the comparison. Against a
   closed-form solution it would not be — integrate the sine over each element
   instead of sampling it.

   Note also that **no element carries the full q₀**: with `NX` even the plate
   centre falls on a *node*, so the four hottest element centres sit half an
   element off-peak and reach 6.852317e7 Pa = q₀·sin²(π(1 − dx/a)/2).

4. **Sign.** With the counter-clockwise connectivity above the element normal
   is +z, and a positive `P` then produces a positive U3. This is the
   empirically verified behaviour of these decks. Reverse the node ordering, or
   make q negative, and the response inverts. (The 3-D solid decks elsewhere in
   ex5 load face `P2`, which acts along −z, which is why every solid result
   there has to be read back sign-flipped.)

5. **Time variation.** There is no `*AMPLITUDE`, and the default amplitude for
   an Abaqus/Standard `*DYNAMIC` step is STEP — so the pressure is applied
   instantaneously at t = 0 and **held** for the whole step. To get a blast,
   sine or triangular pulse instead, emit an `*AMPLITUDE` table and reference
   it with `*DLOAD, AMPLITUDE=<name>`. (A static step would default to RAMP,
   not STEP — the default differs by procedure.)

### `*NODE PRINT, NSET=NCEN, FREQUENCY=1` / `U`
The centre deflection at **every** increment, into the `.dat`. `FREQUENCY=1` is
what lets a post-processor assume one row per increment; anything else and the
time axis has to be rebuilt from the printed step times.

### `*END STEP`

---

## What to change for a different problem

Everything the deck needs lives in `layup_db.yaml`, nowhere in the script:

```yaml
model: 1        # 1 -> writes *TRANSVERSE SHEAR STIFFNESS; 0 -> ABD only
fraction: 0.5   # where z = 0 sits: 0 = bottom face, 0.5 = mid-surface, 1 = top
plate:
  a: 1.524      # side in x [m]
  b: 1.524      # side in y [m]
  nx: 20        # S4 elements along x
  ny: 20        # along y
  q0: 68.9476e6 # peak pressure [Pa]
  dt: 5.0e-5    # fixed increment [s]
  ttot: 0.02    # total time [s]
```

Beyond those, the two things you would edit in the script itself are the
**load footprint** (replace the `sin·sin` expression in the `*DLOAD` loop — a
patch load becomes a centroid-inside-the-patch test) and the **boundary
conditions** (the `*BOUNDARY` tuple).

---

*Companion documents:* [../HOWTO_RUN.md](../HOWTO_RUN.md) for the whole
five-stage pipeline · [../README.md](../README.md) for what the benchmark is.
