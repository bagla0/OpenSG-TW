"""
Thickness convergence: as h/R -> 0 the thin-shell MSG must converge to the 3D
solid.  For each h we homogenize + strain-drive (pure eps11, pure kappa2) both
the JAX shell and (offline, via solid_conv_metrics.csv) the FEniCS-solid, and
track the top-path sigma11 through-thickness gradient/mean and the EI offset.
"""
import os, sys
import numpy as np
import yaml as _yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "opensg_jax"))
import jax; jax.config.update("jax_enable_x64", True)
from fe_jax import solve_tw_from_yaml, stress_at_points

R, ANG, N = 0.0715, -45.0, 160
OUT = os.path.join(HERE, "..", "outputs", "tube_dehom")
HS = [0.008682, 0.004341, 0.002171, 0.001085]


def gen_yaml(path, H):
    Rg = R + H/2.0
    th = np.array([2*np.pi*k/N for k in range(N)])
    nodes = [[float(Rg*np.cos(t)), float(Rg*np.sin(t)), 0.0] for t in th]
    elements = [[k+1, k+2] for k in range(N-1)] + [[N, 1]]
    thm = np.array([np.pi*(2*k+1)/N for k in range(N)])
    ori = [[0., 0., 1., float(-np.sin(t)), float(np.cos(t)), 0.,
            float(-np.cos(t)), float(-np.sin(t)), 0.] for t in thm]
    data = {"nodes": nodes, "elements": elements,
            "sets": {"element": [{"name": "tube", "labels": list(range(1, N+1))}]},
            "sections": [{"elementSet": "tube", "layup": [["aniso", H, ANG]]}],
            "materials": [{"name": "aniso", "density": 1800.0,
                           "elastic": {"E": [37e9, 9e9, 9e9], "G": [4e9, 4e9, 4e9],
                                       "nu": [0.3, 0.3, 0.3]}}],
            "elementOrientations": ori}
    with open(path, "w") as f:
        _yaml.safe_dump(data, f)


def grad_mean(b, H, st):
    yy = np.linspace(R + H/2, R - H/2, 21)
    coords = np.column_stack([np.zeros_like(yy), yy])
    s11 = np.asarray(stress_at_points(b, coords, beam_strain=st,
                                      frame="material")["stress"])[:, 0]
    return s11[-1] - s11[0], 0.5*(s11[0] + s11[-1])   # IML-OML, endpoint avg (clean)


def main():
    eps = np.array([1., 0, 0, 0, 0, 0]); kap = np.array([0., 0, 0, 0, 1., 0])
    rows = []
    for H in HS:
        yml = os.path.join(OUT, f"tube_h{H:.6f}.yaml"); gen_yaml(yml, H)
        b = solve_tw_from_yaml(yml, frac=0.5)
        EBi = float(np.asarray(b["EB"])[2, 2]); Ti = float(np.asarray(b["Timo"])[4, 4])
        ge, me = grad_mean(b, H, eps); gk, mk = grad_mean(b, H, kap)
        rows.append([H, H/R, EBi, Ti, ge, me, gk, mk])
        print(f"h/R={H/R:.3f}  shell EB EI={EBi:.4e} Timo EI={Ti:.4e}  "
              f"eps11 grad/mean={ge/me*100:+6.2f}%  kap2 grad/mean={gk/mk*100:+6.2f}%")
    rows = np.array(rows)

    sol = np.loadtxt(os.path.join(OUT, "solid_conv_metrics.csv"), delimiter=",",
                     usecols=(0, 3, 5, 6, 7))   # h, EI, S11_OML, S11_IML, mean
    se = sol[0::2]; sk = sol[1::2]               # eps11 rows, kappa2 rows
    s_hr = se[:, 0]/R
    se_gm = (se[:, 3]-se[:, 2])/se[:, 4]*100      # grad/mean % (IML-OML)
    sk_gm = (sk[:, 3]-sk[:, 2])/sk[:, 4]*100
    sEI = se[:, 1]

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    fig.suptitle("Thickness convergence: thin-shell MSG -> 3D solid as h/R -> 0",
                 fontweight="bold")
    ax[0].plot(rows[:, 1], rows[:, 4]/rows[:, 5]*100, "r-o", label="shell")
    ax[0].plot(s_hr, se_gm, "g--^", label="solid")
    ax[0].set_title("sigma11 grad/mean, pure eps11")
    ax[1].plot(rows[:, 1], rows[:, 6]/rows[:, 7]*100, "r-o", label="shell")
    ax[1].plot(s_hr, sk_gm, "g--^", label="solid")
    ax[1].set_title("sigma11 grad/mean, pure kappa2")
    ax[2].plot(rows[:, 1], rows[:, 3]/sEI, "r-o", label="shell Timo / solid Timo")
    ax[2].plot(rows[:, 1], rows[:, 2]/sEI, "b:s", label="shell EB / solid Timo")
    ax[2].axhline(1.0, color="0.6", ls=":")
    ax[2].set_title("EI ratio")
    for a in ax:
        a.set_xlabel("h/R"); a.grid(True, ls=":", alpha=0.6); a.legend(fontsize=9)
    ax[0].set_ylabel("grad/mean (%)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(OUT, "tube_thickness_conv.png"), dpi=150); plt.close(fig)

    print("\n  h/R    shell eps grad%  solid eps grad%  shell kap grad%  solid kap grad%"
          "   shellTimoEI/solidTimoEI")
    for i in range(len(rows)):
        print(f"  {rows[i,1]:.3f}    {rows[i,4]/rows[i,5]*100:+8.2f}      "
              f"{se_gm[i]:+8.2f}        {rows[i,6]/rows[i,7]*100:+8.2f}      "
              f"{sk_gm[i]:+8.2f}        {rows[i,3]/sEI[i]:.4f}")
    print("wrote", os.path.join(OUT, "tube_thickness_conv.png"))


if __name__ == "__main__":
    main()
