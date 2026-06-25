"""Canonical e1/e2/e3 orientation plotter for OpenSG shell/solid SG YAMLs.

COMPULSORY DELIVERABLE: an orientation PNG is emitted on EVERY homogenization run (auto-emitted by
strip_RM.rm_timoshenko_6x6 with orient=True, and by gradient_junction_kirchhoff). Sign conventions:
  e2 = blue  (in-plane tangent / XML flow direction)
  e3 = green when it follows the OML->IML default (points toward the section interior),
       red   when it opposes  (e3 default = OML -> IML)
  e1 = beam axis, out-of-plane (+y1); printed numerically (e1_z mean ~ 1).

API:
  plot_orient(shell_yaml, solid_yaml=None, out_png=None) -> out_png   (1 or 2 stacked panels)
  auto_emit(shell_yaml, solid_yaml=None, out_png=None)               (cached once per path per process,
                                                                       never raises -- safe in compute path)
"""
import os
import numpy as np
import yaml

_EMITTED = set()   # process cache of (shell_abs, solid_abs) already plotted -> emit once per mesh


def _load(path):
    d = yaml.safe_load(open(path))
    nodes = np.array([[float(v) for v in str(r[0]).split()][:2] for r in d["nodes"]])
    elems = [[int(v) - 1 for v in str(r[0]).split()] for r in d["elements"]]
    oris = np.array([[float(v) for v in (r if isinstance(r, (list, tuple)) else [r])]
                     for r in d["elementOrientations"]])
    return nodes, elems, oris


def _panel(ax, nodes, elems, oris, is_solid, title, stride):
    C = nodes.mean(axis=0)
    if is_solid:
        tris = np.array([e[:3] for e in elems])
        ax.triplot(nodes[:, 0], nodes[:, 1], tris, color="0.88", lw=0.12)
    else:
        for e in elems:
            ax.plot(nodes[e[:2], 0], nodes[e[:2], 1], color="0.6", lw=0.6)
    ng = 0
    e1z = []
    for k in range(0, len(elems), stride):
        cen = nodes[elems[k][:3 if is_solid else 2]].mean(axis=0)
        e2 = oris[k, 3:5]
        e3 = oris[k, 6:8]
        oml2iml = np.dot(e3, C - cen) > 0          # toward centroid = OML->IML for the skin
        ng += oml2iml
        if oris.shape[1] >= 3:
            e1z.append(oris[k, 2])
        ax.quiver(cen[0], cen[1], e2[0], e2[1], color="tab:blue", scale=45, width=0.002, alpha=0.6)
        ax.quiver(cen[0], cen[1], e3[0], e3[1], color=("tab:green" if oml2iml else "tab:red"),
                  scale=45, width=0.0032)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(r"$y_2$ (m)")
    ax.set_ylabel(r"$y_3$ (m)")
    return ng, len(range(0, len(elems), stride)), (np.mean(e1z) if e1z else float("nan"))


def plot_orient(shell_yaml, solid_yaml=None, out_png=None):
    """Emit the e1/e2/e3 orientation PNG (shell panel always; solid panel if solid_yaml given)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out_png is None:
        base = os.path.splitext(os.path.abspath(shell_yaml))[0]
        out_png = base + "_orient_e1e2e3.png"
    sn, se, so = _load(shell_yaml)
    rows = 2 if solid_yaml else 1
    fig, axes = plt.subplots(rows, 1, figsize=(12, 4.6 * rows))
    if rows == 1:
        axes = [axes]
    ng, nt, e1z = _panel(axes[0], sn, se, so, False,
                         "SHELL (1-D)  e2=blue (XML flow)   e3 OML->IML=green / opposed=red   [e1=+y1 out-of-plane]", 1)
    print("[orient_plot] SHELL: e3 OML->IML %d/%d   e1_z mean=%.3f" % (ng, nt, e1z))
    if solid_yaml:
        qn, qe, qo = _load(solid_yaml)
        stride = max(1, len(qe) // 450)
        ngq, ntq, e1zq = _panel(axes[1], qn, qe, qo, True,
                                "SOLID (2-D)  e2=blue   e3 OML->IML=green / opposed=red   [e1=+y1 out-of-plane]", stride)
        print("[orient_plot] SOLID: e3 OML->IML %d/%d   e1_z mean=%.3f" % (ngq, ntq, e1zq))
    fig.suptitle("e1/e2/e3 orientation  (solid + shell)  -  e3 default = OML -> IML", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[orient_plot] wrote", out_png)
    return out_png


def auto_emit(shell_yaml, solid_yaml=None, out_png=None):
    """Emit once per (shell, solid) path per process; never raises (safe inside the compute path)."""
    key = (os.path.abspath(shell_yaml), os.path.abspath(solid_yaml) if solid_yaml else None)
    if key in _EMITTED:
        return None
    _EMITTED.add(key)
    try:
        return plot_orient(shell_yaml, solid_yaml, out_png)
    except Exception as e:                       # plotting must NEVER break a homogenization run
        print("[orient_plot] skipped (%s)" % e)
        return None
