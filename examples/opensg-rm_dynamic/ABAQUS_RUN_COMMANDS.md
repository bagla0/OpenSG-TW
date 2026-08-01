# Abaqus remote run — exact commands used (for future reruns)

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

## 4. Post-process (local/compute side)

With the four `.dat` files in `Abaqus_results/`:

```bash
python examples/opensg-rm_dynamic/ex5/reddy_hsdt_navier.py   # Reddy curves first
python examples/opensg-rm_dynamic/ex5/recover_dyn.py         # both pulses
python examples/opensg-rm_dynamic/ex5/recover_dyn.py --kind step
```
