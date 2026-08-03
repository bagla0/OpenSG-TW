# `abaqus_3dfea.py` — the 3-D benchmark deck

```
layup_db.yaml  ->  layup_db_3dfea.inp
```

This is the model the OpenSG-RM shell is *judged against*. Nothing here is
homogenized: every ply is meshed as its own layer of solid bricks carrying its
real 3-D material properties and its own fibre orientation. Where
`abaqus_inp.py` replaces the whole wall with one 8×8 plate law,
`abaqus_3dfea.py` resolves the wall directly.

| | shell (`abaqus_inp.py`) | 3-D FEA (`abaqus_3dfea.py`) |
|---|---|---|
| elements | 400 S4 | 6400 C3D8I |
| nodes | 441 | 7497 |
| constitutive input | one homogenized 6×6 (+2×2) | 9 materials, ply by ply |
| plies in the model | none | all of them, resolved |
| runtime | ~40 s | ~5½ min |

Both read the **same `layup_db.yaml`** and share geometry, load, boundary
conditions and time integration. They differ *only* in how the wall is
represented, which is what makes the comparison mean something.

```bash
python abaqus_3dfea.py
```

```bash
abaqus job=layup_db_3dfea cpus=4 interactive
```

---

## The through-thickness mesh

The one thing this deck needs that the shell does not is how finely to slice
each ply. That comes from an optional `divisions` key on each layup entry
(default 1), read only by this script:

```yaml
- {material: ge,    thickness: 0.0009525, angle:  0.0}
- {material: herex, thickness: 0.1447800, angle:  0.0, divisions: 8}
```

The thin face plies get one element each; the 144.78 mm core is split 8 ways so
the benchmark can actually resolve the shear gradient through it. That gives

```
4 x 0.9525 mm | 8 x 18.1 mm | 4 x 0.9525 mm   =  16 layers, 152.4 mm
```

`tlay` holds those 16 thicknesses and `play` records which ply each came from,
so every solid layer inherits the right material and angle. `1d_sg.py` ignores
`divisions` entirely — the SG's own through-thickness discretisation is the
separate `mesh:` block.

> **Why `C3D8I` and not `C3D8`.** The face-ply elements are 76.2 mm across and
> 0.95 mm thick — an aspect ratio near 80:1. Plain `C3D8` shear-locks badly at
> that shape and would come out far too stiff. `C3D8I` adds incompatible modes
> that restore correct bending, at modest cost. This choice is not optional
> here; it is what makes a ply-resolved solid model usable at all.

---

## The keywords, in order

### `*NODE` / `*ELEMENT, TYPE=C3D8I`
Node and element ids come from formulas, so boundary sets need no geometric
search:

```
n3(i, j, k) = 1 + i + (NX+1)*j + npl*k        npl = (NX+1)*(NY+1)
e3(i, j, k) = 1 + i + NX*j + NX*NY*k
```

Brick connectivity is the **bottom face counter-clockwise, then the top face
counter-clockwise** — the order Abaqus expects, and the order that fixes which
face is `P1`…`P6`.

**z = 0 is the bottom face here**, not the mid-surface. The shell deck lies in
the plane z = 0 because that is its reference surface; the solid spans
z ∈ [0, h]. So the two models are offset by h/2 in absolute z. That is
harmless for deflection and stress, but it matters the moment you overlay
fields or compare coordinates.

### `*ELSET, ELSET=LAY<k>, GENERATE`
One set per solid layer. Because element ids are contiguous within a layer,
each set is a single `first, last, 1` range.

### `*NSET`
`FX0`, `FXA`, `FY0`, `FYB` are the four side **faces** — every node on that
face through the whole thickness, not just an edge line as in the shell deck.
`NCEN3D` is the single node at the plate centre **on the mid-plane**
(`n3(NX//2, NY//2, NZT//2)`), which is the like-for-like counterpart of the
shell's `NCEN`.

`nx` and `ny` must be even, for the same reason as the shell deck: with an odd
count there is no node at the centre and the probe would silently land half an
element off. The script refuses rather than mislead.

### `*MATERIAL` / `*ELASTIC, TYPE=ENGINEERING CONSTANTS`
Nine constants across **two data lines**, in the order

```
E1, E2, E3, nu12, nu13, nu23, G12, G13,
G23
```

> The trailing comma at the end of the first line is **mandatory** — it tells
> Abaqus the data continues. Drop it and Abaqus reads only eight constants and
> aborts.

The isotropic foam is written the same way with all three moduli equal; that is
valid and avoids a second material formalism.

> **PyYAML trap.** `128.0e9` in the YAML is parsed as a **string**, not a float
> — PyYAML's float resolver requires an explicit exponent *sign*
> (`128.0e+9`). Every number is therefore coerced with `float()` when the
> material database is built. Without that the deck writer dies with
> `TypeError: must be real number, not str`, which is at least loud; the
> dangerous version of this bug is a script that silently string-concatenates
> instead.

### `*ORIENTATION` / `*SOLID SECTION`
One orientation per solid layer, `SYSTEM=RECTANGULAR` with axes (1,0,0) and
(0,1,0), then a rotation of that ply's angle about local axis 3 (the plate
normal). Each `LAY<k>` set gets its ply's material and that orientation.

An orientation is emitted even for the isotropic core layers, where the angle
is meaningless. Harmless, and it keeps the layer loop uniform — but do not
assume the orientation count equals the number of *composite* plies.

### `*BOUNDARY`
SS-1 applied on the side faces:

| card | meaning |
|---|---|
| `FX0, 2, 3` / `FXA, 2, 3` | on the x-faces, v = w = 0 |
| `FY0, 1, 1` / `FYB, 1, 1` | on the y-faces, u = 0 |
| `FY0, 3, 3` / `FYB, 3, 3` | on the y-faces, w = 0 |

**There is no drilling constraint here, and there cannot be.** A solid node has
only three translational DOFs — no rotations exist, so the `NALL, 6, 6` card
that the flat-shell deck requires has no meaning and would be an Abaqus error.
For the same reason the soft-versus-hard simple-support question does not arise
in the solid: there is no edge rotation to leave free or to fix. The solid
simply resolves whatever the 3-D constraint implies, which is one more reason
it is the benchmark.

### `*STEP` / `*DYNAMIC`
Identical to the shell deck: `initial Δt, total time, min Δt, max Δt` with
maximum = `DT`, which is what pins the fixed increment. Matching increments in
both models is what lets the two histories be compared point by point instead
of interpolated.

### `*DLOAD` — `P2`, and the sign
The same double-sine pressure, evaluated at each element centre, applied to the
**top face of the top layer**:

```
<element id>, P2, <magnitude>
```

`P2` is the top face (5-6-7-8) of a `C3D8`. Abaqus pressure acts *into* the
element, so a positive `P2` on the top face pushes **downward, along −z**.

> **This is the opposite sign to the shell deck**, where a positive `P` on a
> +z-normal S4 gives +z deflection. The solid is physically loaded on its top
> surface, as it should be, and the consequence is that **every solid result —
> deflection, stress, everything — must be sign-flipped before being compared
> with the shell.** This is not a bug to be tidied away; it is the single most
> common source of silently wrong comparisons in this suite, and it is why the
> existing ex5 post-processors negate the solid columns.

Only the top layer is loaded (400 lines, one per top-layer element), not all
6400.

### `*NODE PRINT, NSET=NCEN3D, FREQUENCY=1` / `U`
Mid-plane centre deflection every increment, matching the shell's print so the
two `.dat` files line up row for row.

---

## What to compare, and the traps in doing it

1. **Flip the sign** of the solid deflection (see `*DLOAD` above).
2. Both models print at every increment with the same fixed Δt, so no
   interpolation is needed — but only because both decks pin max Δt. If either
   falls back to automatic incrementation the times diverge and you have to
   parse `STEP TIME COMPLETED` rather than assume t = k·Δt.
3. The solid probe is on the **mid-plane** (`NZT//2`), the shell node is the
   reference surface. With `fraction: 0.5` those are the same physical plane;
   with any other `fraction` they are not, and the comparison needs care.
4. The solid is offset by h/2 in absolute z (it spans 0…h, the shell sits at
   z = 0). Only matters for overlaying fields, not for scalar histories.

---

*Companion documents:* [abaqus_inp_README.md](abaqus_inp_README.md) for the
shell deck · [../HOWTO_RUN.md](../HOWTO_RUN.md) for the whole pipeline.
