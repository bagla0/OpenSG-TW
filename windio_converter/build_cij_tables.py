"""Express the IEA-22 Timoshenko stiffness in Cij notation. Emits the full 6x6 (solid/RM/KL) per station
-> iea22_Cij_matrices.dat, and a diagonal-stiffness table C11..C66 across span -> iea22_Cij_diagonal.dat."""
import os, sys
import numpy as np
CC = r"C:\Users\bagla0\OneDrive - purdue.edu\2026_195\Claude_code"
for p in ("windio_converter", "rm", "opensg_jax", "", os.path.join("mh104_9cells", "scripts")):
    sys.path.insert(0, os.path.join(CC, p))
import jax; jax.config.update("jax_enable_x64", True)
from strip_RM import rm_timoshenko_6x6
from gradient_kirchhoff import gradient_junction_kirchhoff

VAL = os.path.join(CC, "windio_converter", "validation")
STATIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# Cij DOF order: 1=axial(EA) 2=shear-y(GA2) 3=shear-z(GA3) 4=twist(GJ) 5=bend-y(EI2) 6=bend-z(EI3)
DIAG = ["C11", "C22", "C33", "C44", "C55", "C66"]
LEG = "C11=axial(EA)  C22=shear-y(GA2)  C33=shear-z(GA3)  C44=torsion(GJ)  C55=bend-y(EI2)  C66=bend-z(EI3)"


def sym(M):
    M = np.asarray(M); return 0.5 * (M + M.T)


rows = []
for r in STATIONS:
    tag = "r%03d" % round(r * 100)
    shell = os.path.join(VAL, "shell_iea22_%s.yaml" % tag)
    RM = sym(rm_timoshenko_6x6(shell, 0.0, orient=False))
    KL = sym(gradient_junction_kirchhoff(shell, frac=0.0, orient=False)[0])
    sp = os.path.join(VAL, "C6_solid_iea22_%s.txt" % tag)
    S = sym(np.loadtxt(sp)) if os.path.exists(sp) else None
    rows.append((r, S, RM, KL))

# ---- full 6x6 in Cij ----
with open(os.path.join(VAL, "iea22_Cij_matrices.dat"), "w") as f:
    f.write("# IEA-22-280-RWT Timoshenko stiffness 6x6 in Cij notation (OML reference, SI units N, N*m^2)\n")
    f.write("# DOF: 1=axial 2=shear-y 3=shear-z 4=twist 5=bend-y 6=bend-z ;  Cij couples DOF i with DOF j\n")
    f.write("# %s\n" % LEG)
    for (r, S, RM, KL) in rows:
        f.write("\n# ================= station r = %.2f =================\n" % r)
        for name, M in (("2D-SOLID", S), ("RM", RM), ("KL", KL)):
            if M is None:
                f.write("# %s : (solid pending)\n" % name); continue
            f.write("# %s\n" % name)
            f.write("#" + "".join("%15s" % ("C%d*" % (j + 1)) for j in range(6)) + "\n")
            for i in range(6):
                f.write(" " + " ".join("%14.6e" % M[i, j] for j in range(6)) + "   # C%d*\n" % (i + 1))

# ---- diagonal stiffness table C11..C66 ----
def diag_table(mi, name):
    out = ["# IEA-22 diagonal Timoshenko stiffness C11..C66 (%s)  [%s]" % (name, LEG),
           "%-6s %14s %14s %14s %14s %14s %14s" % ("r", *DIAG)]
    for (r, S, RM, KL) in rows:
        M = (S, RM, KL)[mi]
        if M is None:
            out.append("%-6.2f  (pending)" % r); continue
        out.append("%-6.2f " % r + " ".join("%14.6e" % M[i, i] for i in range(6)))
    return "\n".join(out)


with open(os.path.join(VAL, "iea22_Cij_diagonal.dat"), "w") as f:
    f.write(diag_table(1, "RM") + "\n\n" + diag_table(2, "KL") + "\n\n" + diag_table(0, "2D-solid") + "\n")

print("Cij legend:", LEG)
print("\n##### Diagonal stiffness C11..C66 -- RM (corrected tilted-web shell) #####")
print(diag_table(1, "RM"))
print("\n##### Full 6x6 in Cij at r=0.50 (RM) #####")
M = [m for m in rows if abs(m[0] - 0.5) < 1e-9][0][2]
print("      " + "".join("%14s" % ("C%d*" % (j + 1)) for j in range(6)))
for i in range(6):
    print("C%d* " % (i + 1) + " ".join("%13.5e" % M[i, j] for j in range(6)))
print("\nwrote iea22_Cij_matrices.dat, iea22_Cij_diagonal.dat -> validation/")
