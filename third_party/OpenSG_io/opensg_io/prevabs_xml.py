"""PreVABS XML  ->  OpenSG cross-section inputs.

A PreVABS cross-section XML already encodes the section: an airfoil baseline (.dat), dividing points,
layups (plies), surface segments (baseline -> layup), and webs. This adapter turns one into:
  - a 1D-shell SG YAML  -- reconstructed directly from the XML midline + layups (no PreVABS run needed);
  - a 2D-solid SG YAML  -- by running PreVABS on the XML (-> .sg -> convert_sg_to_yaml).

So OpenSG_io accepts a PreVABS XML as a first-class input, not only windIO.
"""
import os
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from .converter import arc_param, emit_opensg_yaml, _Flow  # reuse the OpenSG 1D-shell writer


class _MatProvider:
    """Minimal stand-in for a windIO blade so emit_opensg_yaml(cs) can fetch materials by name."""
    def __init__(self, mats):
        self.mats = mats


def _read_materials_xml(path):
    """materials.xml -> {name: {E:[3], G:[3], nu:[3], rho, ply_t}} (+ lamina name -> (material, thickness))."""
    root = ET.parse(path).getroot()
    mats, lam = {}, {}
    for m in root.findall("material"):
        e = m.find("elastic")
        def g(tag, d=0.0):
            x = e.find(tag)
            return float(x.text) if x is not None else d
        mats[m.get("name")] = dict(
            E=[g("e1"), g("e2"), g("e3")], G=[g("g12"), g("g13"), g("g23")],
            nu=[g("nu12"), g("nu13"), g("nu23")],
            rho=float(m.findtext("density", "1.0")))
    for la in root.findall("lamina"):
        lam[la.get("name")] = (la.findtext("material"), float(la.findtext("thickness", "0.001")))
    return mats, lam


def parse_prevabs_xml(xml_path):
    """Parse a PreVABS XML into a cross-section dict compatible with emit_opensg_yaml()."""
    xdir = os.path.dirname(os.path.abspath(xml_path))
    root = ET.parse(xml_path).getroot()
    chord = float(root.find("general/scale").text)

    # materials (the <include><material>NAME</material> -> NAME.xml)
    inc = root.findtext("include/material", "materials")
    mats, lamina = _read_materials_xml(os.path.join(xdir, inc + ".xml"))

    # airfoil contour from the .dat referenced by the airfoil baseline
    bl = root.find("baselines")
    af = bl.find("line[@type='airfoil']")
    datfile = af.find("points").text.strip()
    raw = [l.split() for l in open(os.path.join(xdir, datfile)) if len(l.split()) == 2]
    xy = np.array([[float(a), float(b)] for a, b in raw if _isnum(a)]) * chord
    s_arc = arc_param(xy)
    perim = float(np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))][-1])

    # dividing points: name -> arc position s (mapped from normalised x + top/bottom side)
    dps = {}
    for p in bl.findall("point[@on]"):
        xn = float(p.text) * chord
        dps[p.get("name")] = _x_to_s(xy, s_arc, xn, p.get("which", "top"))

    # layups: name -> [(material, thickness, angle)]
    layups = {}
    for lu in root.findall("layups/layup"):
        plies = []
        for ly in lu.findall("layer"):
            mat, lthk = lamina[ly.get("lamina")]
            ang, cnt = ly.text.split(":")
            plies.append((mat, float(cnt) * lthk, float(ang)))
        layups[lu.get("name")] = tuple(plies)

    # surface segments: ordered (s_a, s_b, layup) along the contour
    seg_specs = []
    for sg in root.findall("component[@name='surface']/segment") or root.findall("component/segment"):
        blname = sg.findtext("baseline"); lname = sg.findtext("layup")
        line = bl.find("line[@name='%s']" % blname)
        pts = line.findtext("points")
        a, b = pts.split(":")
        seg_specs.append((dps[a], dps[b], layups[lname]))

    # build the midline mesh (nodes/elements/element-sets) from the segments
    cs = _mesh_from_segments(xy, s_arc, perim, chord, seg_specs, layups)

    # webs: bl_web_i = point(midpoint, normalised) + angle -> two contour intersections -> node chain
    webs = []
    laminates = cs["laminates"]
    for comp in root.findall("component"):
        if comp.get("name", "").startswith("web"):
            sg = comp.find("segment"); line = bl.find("line[@name='%s']" % sg.findtext("baseline"))
            wp = bl.find("point[@name='%s']" % line.findtext("point"))
            mx, my = [float(t) * chord for t in wp.text.split()]
            ang = float(line.findtext("angle"))
            na, nb = _web_endpoints(cs["nodes"], np.array([mx, my]), ang)
            lam = layups[sg.findtext("layup")]
            if lam not in laminates:
                laminates[lam] = len(laminates)
            webs.append(dict(a=na, b=nb, lam=laminates[lam], name=comp.get("name"),
                             s=0.0, e=0.0))
    cs["webs"] = webs
    cs["blade"] = _MatProvider(mats)
    cs["chord"] = chord
    cs["twist"] = 0.0
    return cs


def to_shell(xml_path, out_yaml, web_mesh=None):
    """PreVABS XML -> OpenSG 1D-shell SG YAML."""
    cs = parse_prevabs_xml(xml_path)
    return emit_opensg_yaml(cs, out_yaml, web_mesh=web_mesh)


def to_solid(xml_path, out_yaml, prevabs, convert_py=None, py=None):
    """PreVABS XML -> 2D-solid SG YAML by running PreVABS (.sg) then convert_sg_to_yaml.py."""
    import sys
    xdir = os.path.dirname(os.path.abspath(xml_path))
    name = os.path.splitext(os.path.basename(xml_path))[0]
    subprocess.run([prevabs, "-i", os.path.basename(xml_path), "--vabs", "--hm"], cwd=xdir, check=True)
    sg = os.path.join(xdir, name + ".sg")
    convert_py = convert_py or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "scripts", "convert_sg_to_yaml.py")
    subprocess.run([py or sys.executable, convert_py, sg, out_yaml], check=True)
    return out_yaml


# ---- helpers -------------------------------------------------------------------------------------
def _isnum(s):
    try:
        float(s); return True
    except ValueError:
        return False


def _x_to_s(xy, s_arc, xtarget, side):
    """Arc position s of the contour point at x=xtarget on the given side (top: s<0.5, bottom: s>0.5)."""
    lo, hi = (0.0, 0.5) if side == "top" else (0.5, 1.0)
    ss = np.linspace(lo, hi, 2000)
    xs = np.interp(ss, s_arc, xy[:, 0])
    return float(ss[int(np.argmin(np.abs(xs - xtarget)))])


def _mesh_from_segments(xy, s_arc, perim, chord, seg_specs, layups, mesh_size=0.01):
    """Discretise the contour into nodes/line-elements; assign each segment's layup as an element set."""
    brks = sorted({0.0, 1.0} | {round(float(v), 8) for (a, b, _) in seg_specs for v in (a, b)})
    laminates, nodes, node_arc, elems, elem_lam, segments = {}, [], [], [], [], []

    def lam_at(smid):
        for (a, b, lam) in seg_specs:
            if min(a, b) - 1e-9 <= smid <= max(a, b) + 1e-9:
                return lam
        return seg_specs[0][2]

    def pt(s):
        return [float(np.interp(s, s_arc, xy[:, 0])), float(np.interp(s, s_arc, xy[:, 1]))]
    prev = None
    for i in range(len(brks) - 1):
        a, b = brks[i], brks[i + 1]
        nsub = max(1, int(round((b - a) * perim / (mesh_size * chord))))
        lam = lam_at(0.5 * (a + b))
        if lam not in laminates:
            laminates[lam] = len(laminates)
        lid = laminates[lam]
        segments.append(dict(s_a=a, s_b=b, set_id=lid))
        ss = np.linspace(a, b, nsub + 1)
        if prev is None:
            nodes.append(pt(ss[0])); node_arc.append(ss[0]); prev = 0
        for s in ss[1:]:
            nodes.append(pt(s)); node_arc.append(s); cur = len(nodes) - 1
            elems.append((prev, cur)); elem_lam.append(lid); prev = cur
    nodes = np.array(nodes)
    if np.linalg.norm(nodes[-1] - nodes[0]) < 1e-9 * chord:
        elems[-1] = (elems[-1][0], 0); nodes = nodes[:-1]
    else:
        elems.append((len(nodes) - 1, 0)); elem_lam.append(elem_lam[-1])
    return dict(nodes=nodes, elems=elems, elem_lam=elem_lam, laminates=laminates,
                segments=segments, xy=xy, s_arc=s_arc, perim=perim)


def _web_endpoints(nodes, mid, ang_deg):
    """Two contour nodes closest to the infinite web line through `mid` at angle ang_deg (90=vertical)."""
    d = np.array([np.cos(np.radians(ang_deg)), np.sin(np.radians(ang_deg))])
    n = np.array([-d[1], d[0]])                       # normal to the web line
    perp = (np.asarray(nodes) - mid) @ n              # signed distance to the line
    along = (np.asarray(nodes) - mid) @ d
    upper = np.where(along > 0)[0]; lower = np.where(along <= 0)[0]
    na = int(upper[np.argmin(np.abs(perp[upper]))]); nb = int(lower[np.argmin(np.abs(perp[lower]))])
    return na, nb
