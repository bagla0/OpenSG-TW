# Command-line reference

The whole plate chain runs without writing Python
(`python -m opensg_jax.fe_jax.cli_rm_plate ...` from the repo, or the
`opensg-rm-plate` console script when pip-installed).

## `homo` — the 8×8 plate law

```bash
opensg-rm-plate homo my_sg.yaml [--out PREFIX]
```

Prints the labeled $8\times8$ ABDG and its compliance; `--out` also writes
`PREFIX_8x8.out`.  Exit code 2 (with a message) if the transverse-shear fit
is not SPD.

## `plot` — the SG mesh figure

```bash
opensg-rm-plate plot my_sg.yaml [--png PATH]
```

## `dehom` — 3-D recovery

```bash
opensg-rm-plate dehom my_sg.yaml \
    (--FF N11 N22 N12 M11 M22 M12 Q1 Q2 | --strain e11 e22 g12 k11 k22 k12 2g13 2g23) \
    [--u2d U1 U2 U3] \
    [--qtop q q1 q2 q11 q12 q22] [--qbot q q1 q2 q11 q12 q22] \
    [--n-per-ply 21] [--base PREFIX]
```

- `--FF` takes plate **resultants** (converted through the 8×8); `--strain`
  takes the plate strains directly.  The transverse-shear gradients are built
  from $Q_1, Q_2$ internally (user-guide load case 2).
- `--qtop/--qbot` are the optional face-pressure ladders (load case 4).
- `--u2d` adds the plate displacement to the recovered warping in the `.U`
  output.

Outputs (`PREFIX` defaults to the YAML path without `.yaml`):

| file | content (one row per station: `x3` + 6 or 3 values) |
|---|---|
| `PREFIX.SM` | **ply-frame** (material) stress $[11,22,33,23,13,12]$ |
| `PREFIX.EM` | ply-frame strain |
| `PREFIX.U`  | recovered displacement $U_1, U_2, U_3$ |
| `PREFIX.out` | the run report (inputs echoed, station count) |

Example — a wall carrying pure bending $M_{11} = 10^3$ with transverse shear
$Q_1 = 5\times10^2$:

```bash
opensg-rm-plate dehom my_sg.yaml --FF 0 0 0 1e3 0 0 5e2 0 --base out/wall
```
