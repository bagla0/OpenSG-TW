# Abaqus remote run — exact commands used (for future reruns)

> **Note.** Every `for %j in (...)` loop below is the form you type at an
> **interactive** Command Prompt. Inside a `.bat` file the loop variable must be
> **doubled** — `for %%j in (...) do call abaqus job=%%j ...` — or the loop
> expands to nothing and runs zero jobs. See
> [HOWTO_RUN.md §4](HOWTO_RUN.md) for the batch-file template actually used.

Machine: RDP to **10.165.18.74** (Windows, Abaqus 2024, license server
`iacmi-vlm.ecn.purdue.edu`). All commands below were typed in a plain
**Command Prompt** on that machine. Working directory: `C:\Temp\opensg_dyn`.

## 1. Stage the decks on the Abaqus machine

The four `.inp` decks (generated locally by `make_abaqus_dyn.py`) were placed
in the OneDrive folder, which syncs to the remote machine:

```
C:\Users\bagla0\OneDrive - purdue.edu\202603_PlateRM\07012026_opensg_abaqus_dynamic\
```

On the remote cmd, copy them to a plain local folder first — **never run
Abaqus inside a OneDrive folder** (sync locks the `.lck`/scratch files):

```bat
mkdir C:\Temp\opensg_dyn
cd /d C:\Temp\opensg_dyn
copy "C:\Users\bagla0\OneDrive - purdue.edu\202603_PlateRM\07012026_opensg_abaqus_dynamic\sandwich_*.inp" .
```

(Expect `4 file(s) copied.`)

## 2. Run the four jobs back-to-back

One `for` loop, sequential (`interactive` blocks until each job ends;
`call` is required or the loop stops after the first `abaqus.bat`):

```bat
for %j in (sandwich_RM_step sandwich_RM_blast sandwich_SOLID_step sandwich_SOLID_blast) do call abaqus job=%j cpus=4 interactive ask_delete=OFF
```

- `ask_delete=OFF` overwrites old results without the y/n prompt (needed —
  the cmd is non-interactive when driven remotely).
- Success line per job: `Abaqus JOB <name> COMPLETED`.
- Wall-clock observed (cpus=4): **RM shell jobs ~40 s each**, **3-D solid
  jobs ~5.5 min each** (20×20×16 C3D8I, 400 increments) — the shell-vs-solid
  cost ratio quoted in the comparison.

## 2a. The Ex.2 jobs (RM/FSDT shells ~40 s, solids ~5.5 min each)

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex2\ex2_*.inp" .
for %j in (ex2_RM_step ex2_RM_blast ex2_FSDT_step ex2_FSDT_blast ex2_SOLID_step ex2_SOLID_blast) do call abaqus job=%j cpus=4 interactive ask_delete=OFF
copy /Y ex2_*.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex2\Abaqus_results\"
```

## 2b. The Ex.4 frequency jobs (same pattern, ~10 s each)

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex4\ex4_*_freq.inp" .
for %j in (ex4_04_Al_freq ex4_0_pm45_90_Al_freq ex4_pm45x2_Al_freq) do call abaqus job=%j interactive ask_delete=OFF
copy /Y ex4_*_freq.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex4\Abaqus_results\"
```

## 2c. The Ex.5 FSDT jobs (conventional composite S4, ~40 s each)

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\sandwich_FSDT_*.inp" .
for %j in (sandwich_FSDT_step sandwich_FSDT_blast) do call abaqus job=%j cpus=4 interactive ask_delete=OFF
copy /Y sandwich_FSDT_*.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\Abaqus_results\"
```

## 3. Copy the results back to the local drive

The recovery post-processor only needs the `.dat` files (all history prints
are in them). Copy straight onto the roger network share, which is the same
filesystem as the local `Y:` drive and the compute server home:

```bat
copy /Y sandwich_*.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex5\Abaqus_results\"
```

(Create `Abaqus_results\` on the share first if it does not exist.)

To also archive in OneDrive:

```bat
copy /Y sandwich_*.dat "C:\Users\bagla0\OneDrive - purdue.edu\202603_PlateRM\07012026_opensg_abaqus_dynamic\"
```

## Transfer-path notes (learned the hard way)

- **UNC to roger works; `\\tsclient` does not** — the RDP session does not
  expose local-drive redirection, so `\\tsclient\...` fails to resolve.
- **OneDrive down-sync to the remote machine is unreliable/slow** — fine for
  staging inputs ahead of time, but for results prefer the direct roger UNC
  copy (instant, and lands directly in the git working tree).
- If a typed command does not appear at the prompt (RDP paste race), click
  the cmd **title bar** first, then retype; the trailing Enter of a paste is
  sometimes swallowed — press Enter again if the command sits unexecuted.

## 2d. The Ex.1 job (Fig-5 patch-load plate, ~20 s)

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex1\ex1_RM_fig5.inp" .
call abaqus job=ex1_RM_fig5 cpus=4 interactive ask_delete=OFF
copy /Y ex1_RM_fig5.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex1\Abaqus_results\"
```

## 2e. The Ex.3 jobs (static Pagano sandwich, ~30 s each)

```bat
copy /Y "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex3\ex3_RM_*.inp" .
for %%j in (ex3_RM_S10 ex3_RM_S100) do call abaqus job=%%j interactive ask_delete=OFF
copy /Y ex3_RM_*.dat "\\roger.ecn.purdue.edu\bagla0\OpenSG-TW-claude\examples\opensg-rm_dynamic\ex3\Abaqus_results\"
```

## 4. Post-process (local/compute side)

Use the environment python explicitly — even the pure-text post-processors
import JAX transitively:

```bash
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex5/recover_dyn.py
```

With the `.dat` files in `Abaqus_results/`, per case:

```bash
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex5/recover_dyn.py
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex5/make_curves.py
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex5/make_curves9.py
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex2/compare_ex2.py
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex4/collect_freq.py
~/miniconda3/envs/opensg_2_0/bin/python examples/opensg-rm_dynamic/ex1/plot_fig5.py
```

`recover_dyn.py` also takes `--kind step|blast` to do one pulse only.

The Reddy-TSDT analytical anchor moved to `ex2/reddy_hsdt_navier.py`
(it is **not** in `ex5/` any more) and needs `set_case('ex2')` — its module
default state is the Ex.5 sandwich.
