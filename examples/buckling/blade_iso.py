"""blade_iso.py -- ISOTROPIC full IEA-22 blade buckling: JAX-FEA vs RM-OpenSG.

Replaces every laminate with an isotropic wall of the SAME local total thickness, for BOTH pathways.
This removes the anisotropic-ABD material-frame sensitivity (the isotropic cylinder already matched to
MAC=1.0), isolating whether the blade pre-buckling-N -> buckling pipeline is itself self-consistent.
Same 1500 Pa flapwise traction, root clamped.  Caches the isotropic homogenization bundles."""
import os, sys, time, pickle
import numpy as np
BUCK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BUCK)
import blade_buckling as bb
import shell_buckling as sb
import dehom_rm
import yaml

E_ISO, NU_ISO = 3.0e10, 0.3
NSTA, Ntot, MPER = bb.NSTA, bb.Ntot, bb.MPER
sec_elems = bb.sec_elems; NSE = len(sec_elems); DATA = bb.DATA
IC = os.path.join(DATA, "iso_cache"); IY = os.path.join(IC, "yaml")
os.makedirs(os.path.join(IY, "abd"), exist_ok=True)
NMODES = 8


def iso_abd(t):
    return sb._iso_ABD(E_ISO, NU_ISO, t)                       # (ABD6, Gs2)


def make_iso_yaml(i):
    dst = os.path.join(IY, "iso_s%02d.yaml" % i)
    if os.path.exists(dst):
        return dst
    d = yaml.safe_load(open(os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)))
    G = E_ISO / (2 * (1 + NU_ISO))
    for s in d["sections"]:
        t = sum(float(p[1]) for p in s["layup"])
        s["layup"] = [["iso", float(t), 0.0]]                  # one iso ply of the total thickness
    d["materials"] = [{"name": "iso", "density": 1000.0,
                       "elastic": {"E": [E_ISO] * 3, "G": [G] * 3, "nu": [NU_ISO] * 3}}]
    yaml.safe_dump(d, open(dst, "w"))
    return dst


def homog_iso(i):
    cf = os.path.join(IC, "bundle_s%02d.pkl" % i)
    if os.path.exists(cf):
        return pickle.load(open(cf, "rb"))
    B = dehom_rm.build_rm_bundle(make_iso_yaml(i))
    pickle.dump(B, open(cf, "wb"))
    return B


def station_iso(i):
    shell = os.path.join(bb.SHELLD, "iea_s%02d_shell.yaml" % i)
    d = yaml.safe_load(open(shell)); r = i / 50.0
    oml = bb.resample(np.asarray(bb.build_cross_section(bb.blade, r=r)["nodes"], float), bb.N)
    P = np.zeros((Ntot, 2)); P[:bb.N] = oml
    for w in range(bb.NWEB):
        Pa, Pb = oml[bb.web_ia[w]], oml[bb.web_ib[w]]
        tl = np.linspace(0, 1, bb.NW)[1:-1]
        P[bb.wnode(w, 0):bb.wnode(w, 0) + bb.NWI] = Pa[None] + tl[:, None] * (Pb - Pa)[None]
    tree, names = bb.station_layup_lookup(shell)
    thick = {s["elementSet"]: sum(float(p[1]) for p in s["layup"]) for s in d["sections"]}
    mids = 0.5 * (P[sec_elems[:, 0]] + P[sec_elems[:, 1]])
    idx = tree.query(mids + (tree.data.mean(0) - mids.mean(0)))[1]   # align conformal frame -> yaml frame
    ABD = np.zeros((NSE, 6, 6)); Gs = np.zeros((NSE, 2, 2))
    for se in range(NSE):
        a, g = iso_abd(thick[names[idx[se]]]); ABD[se] = a; Gs[se] = g
    return P, ABD, Gs


def build():
    Pk = np.zeros((NSTA, Ntot, 2)); Ak = np.zeros((NSTA, NSE, 6, 6)); Gk = np.zeros((NSTA, NSE, 2, 2))
    Rk = np.arange(NSTA) / 50.0 * bb.BLADE_LEN
    for i in range(NSTA):
        Pk[i], Ak[i], Gk[i] = station_iso(i)
    NS = (NSTA - 1) * MPER + 1
    pts = np.zeros((NS * Ntot, 3))
    for p in range(NS):
        rp = p / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        P = (1 - tt) * Pk[kL] + tt * Pk[kL + 1]; X = (1 - tt) * Rk[kL] + tt * Rk[kL + 1]
        pts[p * Ntot:(p + 1) * Ntot, 0] = X
        pts[p * Ntot:(p + 1) * Ntot, 1] = P[:, 0]; pts[p * Ntot:(p + 1) * Ntot, 2] = P[:, 1]
    quads = []; ABD_e = []; Gs_e = []
    for p in range(NS - 1):
        b0, b1 = p * Ntot, (p + 1) * Ntot
        rp = (p + 0.5) / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        for se in range(NSE):
            a, bbb = sec_elems[se]
            quads.append([b0 + bbb, b1 + bbb, b1 + a, b0 + a])          # e1=span (matches blade_buckling)
            ABD_e.append((1 - tt) * Ak[kL, se] + tt * Ak[kL + 1, se])
            Gs_e.append((1 - tt) * Gk[kL, se] + tt * Gk[kL + 1, se])
    root = np.unique([6 * n + k for n in range(Ntot) for k in range(6)])
    return dict(nodes=pts, quads=np.array(quads), ABD_e=np.array(ABD_e), Gs_e=np.array(Gs_e),
                root=root, Pk=Pk, Ak=Ak, Rk=Rk, NS=NS)


def rm_N(bl, FF):
    Pk, Ak = bl["Pk"], bl["Ak"]; Nst = np.zeros((NSTA, NSE, 3)); t0 = time.time()
    for i in range(NSTA):
        B = homog_iso(i)
        st, st_m, aA, aB = dehom_rm._macro_fields(B, beam_force_vabs=FF[i])
        corners = np.asarray(B["corners"]); rc = np.asarray(B["red_cells"])
        mids = 0.5 * (Pk[i][sec_elems[:, 0]] + Pk[i][sec_elems[:, 1]])
        shift = corners.mean(0) - Pk[i].mean(0)               # align conformal OML frame -> 1-D ring frame
        for se in range(NSE):
            e_ring, xi, pr = dehom_rm._project_point(corners, rc, mids[se] + shift)
            s6, _ = dehom_rm._rm_shell_strain(B, e_ring, xi, st_m, aA, aB)
            Nst[i][se] = Ak[i][se][:3, :3] @ s6[:3] + Ak[i][se][:3, 3:] @ s6[3:6]
        if i % 10 == 0:
            print("  iso dehom %d/%d (%.0fs)" % (i, NSTA, time.time() - t0))
    NS = bl["NS"]; Npl = np.zeros((NS, NSE, 3))
    for p in range(NS):
        rp = p / MPER; kL = min(int(np.floor(rp)), NSTA - 2); tt = rp - kL
        Npl[p] = (1 - tt) * Nst[kL] + tt * Nst[kL + 1]
    Ne = np.zeros((len(bl["quads"]), 3)); q = 0
    for p in range(NS - 1):
        for se in range(NSE):
            Ne[q] = 0.5 * (Npl[p, se] + Npl[p + 1, se]); q += 1
    return Ne


if __name__ == "__main__":
    t0 = time.time(); bl = build()
    nodes, quads, ABD_e, Gs_e, root = bl["nodes"], bl["quads"], bl["ABD_e"], bl["Gs_e"], bl["root"]
    print("iso blade mesh: %d nodes %d quads (%.1fs, E=%.1e nu=%.2f)" % (len(nodes), len(quads), time.time() - t0, E_ISO, NU_ISO))
    K = sb.assemble_K(nodes, quads, ABD_e, Gs_e)
    f = bb.traction_load(nodes, quads)
    u = sb.solve_static(nodes, quads, ABD_e, Gs_e, f, root, K=K)
    itip = int(np.argmax(nodes[:, 0]))
    Nf = sb.element_membrane_N(nodes, quads, ABD_e, u)
    lf, mf = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nf, root, n_modes=NMODES)
    print("iso JAX-FEA : tip flap=%.3f m  lambda_1=%.4f  first=%s"
          % (u[6 * itip + 2], lf[0], np.array2string(lf[:NMODES], precision=3)))
    FF = bb.beam_forces_from_traction(nodes, f, bl["Rk"])
    Nr = rm_N(bl, FF)
    lr, mr = sb.solve_buckling(nodes, quads, ABD_e, Gs_e, Nr, root, n_modes=NMODES)
    print("iso RM-OpenSG: lambda_1=%.4f  first=%s" % (lr[0], np.array2string(lr[:NMODES], precision=3)))
    rel = np.linalg.norm(Nr - Nf) / (np.linalg.norm(Nf) + 1e-30)
    corr = np.corrcoef(Nf[:, 0], Nr[:, 0])[0, 1]
    print("\n=== ISO blade: RM/FEA lambda_1=%.3f ; ||RM-FEA||/||FEA||=%.3f ; span-N corr=%.3f ; rms FEA=%.2e RM=%.2e"
          % (lr[0] / lf[0], rel, corr, np.sqrt((Nf[:, 0]**2).mean()), np.sqrt((Nr[:, 0]**2).mean())))
    np.savez(os.path.join(DATA, "blade_iso.npz"), nodes=nodes, quads=quads, loads_fea=lf, modes_fea=mf,
             loads_rm=lr, modes_rm=mr, Nf=Nf, Nr=Nr)
