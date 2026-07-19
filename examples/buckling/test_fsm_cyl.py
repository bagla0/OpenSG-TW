"""test_fsm_cyl.py -- set the FSM standard on the cylinder: ISOTROPIC and ANISOTROPIC (m45 [+-45]s).
FSM (cross-section-only) vs 3-D shell FEA (shell_buckling) vs classical (iso).  Signature-curve figure."""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BUCK = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, BUCK)
import fsm_buckling as fsm
import shell_buckling as sb

R, t, L = 1.0, 0.02, 2.0
NC, NL = 160, 80
rt = np.sqrt(R * t)
a_list = np.geomspace(0.4 * rt, 8 * rt, 44)          # local regime around sqrt(Rt)


def local_min(aa, lam):
    """local buckling = minimum of the signature curve within the local regime."""
    i = int(np.argmin(lam)); return aa[i], lam[i]


# ---- materials / ABD ----
ISO = dict(E=200e9, nu=0.3)
ABD_iso = fsm.iso_abd(ISO["E"], ISO["nu"], t)
MAT = dict(E1=140e9, E2=10e9, G12=5e9, nu12=0.3)      # carbon/epoxy
ABD_m45 = fsm.clt_abd([(45, t / 4), (-45, t / 4), (-45, t / 4), (45, t / 4)], MAT)   # [+-45]s
# transverse shear (for the 3-D FEA) ~ (5/6) G t
Gs_iso = (5.0 / 6.0) * (ISO["E"] / (2 * (1 + ISO["nu"]))) * t * np.eye(2)
Gs_m45 = (5.0 / 6.0) * MAT["G12"] * t * np.eye(2)


# ---- 3-D shell FEA on the SS3 cylinder for a given ABD ----
def cyl_fea(ABD, Gs, nmodes=6):
    th = np.linspace(0, 2 * np.pi, NC, endpoint=False); xs = np.linspace(0, L, NL + 1)
    nodes = np.array([[xs[i], R * np.cos(th[j]), R * np.sin(th[j])] for i in range(NL + 1) for j in range(NC)])
    idx = lambda i, j: i * NC + (j % NC)
    quads = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for i in range(NL) for j in range(NC)])
    ne = len(quads)
    ABD_e = np.repeat(ABD[None], ne, 0); Gs_e = np.repeat(Gs[None], ne, 0)
    Nvec = np.repeat(np.array([-1.0, 0.0, 0.0])[None], ne, 0)
    fixed = []
    for j in range(NC):
        r0, rL = idx(0, j), idx(NL, j)
        fixed += [6 * r0 + 1, 6 * r0 + 2, 6 * r0 + 0, 6 * rL + 1, 6 * rL + 2]     # SS3
    loads, _ = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nvec, np.unique(fixed), n_modes=nmodes)
    return loads[0]


# ---- FSM ----
ring, strips = fsm.cyl_ring(R, NC)
N_s = [np.array([-1.0, 0.0, 0.0])] * len(strips)
Ncl = ISO["E"] * t**2 / (R * np.sqrt(3 * (1 - ISO["nu"]**2)))

res = {}
for tag, ABD, Gs in [("iso", ABD_iso, Gs_iso), ("m45", ABD_m45, Gs_m45)]:
    t0 = time.time()
    aa, lam, _, _ = fsm.signature_curve(ring, strips, [ABD] * len(strips), N_s, a_list)
    ac, lc = local_min(aa, lam); tfsm = time.time() - t0
    t1 = time.time(); nfea = cyl_fea(ABD, Gs); tfea = time.time() - t1
    res[tag] = dict(aa=aa, lam=lam, ac=ac, lc=lc, fea=nfea, tfsm=tfsm, tfea=tfea)
    print("%s: FSM local N_cr=%.4e (a*=%.3f, %.2fs) | 3D-FEA N_cr=%.4e (%.0fs) | FSM/FEA=%.3f"
          % (tag, lc, ac, tfsm, nfea, tfea, lc / nfea))
print("\niso classical N_cr = %.4e ; FSM/classical=%.3f ; FEA/classical=%.3f"
      % (Ncl, res["iso"]["lc"] / Ncl, res["iso"]["fea"] / Ncl))

# ---- signature-curve figure ----
fig, ax = plt.subplots(figsize=(7, 4.2))
for tag, c in [("iso", "C0"), ("m45", "C1")]:
    r = res[tag]
    ax.plot(r["aa"] / rt, r["lam"], "-o", color=c, ms=3, label="%s FSM (min=%.3e)" % (tag.upper(), r["lc"]))
    ax.axhline(r["fea"], color=c, ls="--", lw=1, label="%s 3D-FEA=%.3e" % (tag.upper(), r["fea"]))
ax.axhline(Ncl, color="k", ls=":", lw=1, label="iso classical=%.3e" % Ncl)
ax.set_xlabel(r"half-wavelength $a/\sqrt{Rt}$"); ax.set_ylabel(r"buckling $N_{cr}$ (N/m)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(os.path.join(BUCK, "fig", "fsm_cyl_signature.png"), dpi=130, bbox_inches="tight")
print("wrote fig/fsm_cyl_signature.png")
