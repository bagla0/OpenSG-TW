"""render_blade_pv.py -- LOCAL pvpython: render the lofted full-blade hex .vtk to PNGs (rainbow).

TRUE geometry (no exaggeration), isometric, XYZ triad, colour bar close.  The SKIN wall
(blade3d_hex51.vtk) is drawn semi-transparent; the internal SHEAR WEBS (blade3d_webs51.vtk, if present)
are drawn OPAQUE inside it, so the 3-D webbed structure is visible.  Stress uses a symmetric range
(0 = green), displacement its natural range.  Outputs: 6 field views + an inboard webbed cutaway +
the deflected shape (undeformed grey + real-scale WarpByVector on the total RM_disp).

  & "C:\\Program Files\\ParaView-5.12.0-Windows-Python3.10-msvc2017-AMD64\\bin\\pvpython.exe" ^
      render_blade_pv.py <blade3d_hex51.vtk> <outdir>
"""
from paraview.simple import *
import os, sys

vtkfile = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(vtkfile)
webfile = os.path.join(os.path.dirname(vtkfile), "blade3d_webs51.vtk")
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

FIELDS = [("RM_S11", "sigma11 (MPa)", 1), ("RM_S22", "sigma22 (MPa)", 1), ("RM_S12", "sigma12 (MPa)", 1),
          ("RM_u1", "u1 out-of-plane warping (m)", 0), ("RM_u2", "u2 edgewise (m)", 0), ("RM_u3", "u3 flapwise (m)", 0)]

reader = OpenDataFile(vtkfile)
webreader = OpenDataFile(webfile) if os.path.exists(webfile) else None
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1500, 780]
view.UseColorPaletteForBackground = 0
view.Background = [1, 1, 1]
view.OrientationAxesVisibility = 1
view.OrientationAxesLabelColor = [0, 0, 0]
view.CameraParallelProjection = 1
reader.UpdatePipeline()                          # update the SKIN reader (not the last-opened webreader)
b = reader.GetDataInformation().GetBounds()


def symrange(field):
    ai = reader.GetPointDataInformation().GetArray(field)
    r = ai.GetComponentRange(0)
    return max(abs(r[0]), abs(r[1]), 1e-9)


def set_lut(field, symmetric):
    lut = GetColorTransferFunction(field); lut.ApplyPreset("Rainbow Uniform", True)
    if symmetric:
        m = symrange(field); lut.RescaleTransferFunction(-m, m)
    return lut


def colour(rep, field, title, symmetric, opacity=1.0, bar=True, rescale=True):
    ColorBy(rep, ("POINTS", field))
    if rescale:                                    # webs pass rescale=False -> share the skin's LUT range
        rep.RescaleTransferFunctionToDataRange(True, False)
    rep.Opacity = opacity
    lut = set_lut(field, symmetric)
    if not bar:
        rep.SetScalarBarVisibility(view, False); return lut
    rep.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = title; sb.ComponentTitle = ""
    sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
    sb.TitleFontSize = 22; sb.LabelFontSize = 20
    sb.ScalarBarLength = 0.6
    sb.WindowLocation = "Any Location"; sb.Position = [0.86, 0.2]
    return lut


def iso(bounds, zoom=0.60, cdir=(0.55, -0.75, 0.55)):
    c = [0.5 * (bounds[0] + bounds[1]), 0.5 * (bounds[2] + bounds[3]), 0.5 * (bounds[4] + bounds[5])]
    d = ((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2 + (bounds[5] - bounds[4]) ** 2) ** 0.5
    cam = GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(c[0] + cdir[0] * d, c[1] + cdir[1] * d, c[2] + cdir[2] * d)
    cam.SetViewUp(0, 0, 1)
    ResetCamera(view); view.CameraParallelScale = view.CameraParallelScale * zoom


# ---- 6 field views: translucent skin + opaque webs, from-above isometric ----
skin = Show(reader, view); skin.Representation = "Surface"
webs = Show(webreader, view) if webreader else None
if webs:
    webs.Representation = "Surface"
UpdatePipeline()
for f, title, sym in FIELDS:
    colour(skin, f, title, sym, opacity=0.4)
    if webs:
        colour(webs, f, title, sym, opacity=1.0, bar=False, rescale=False)   # webs opaque, share skin's LUT range
    iso(b, 0.58, cdir=(0.30, -0.45, 0.85)); Render(view)
    SaveScreenshot(os.path.join(outdir, "blade_%s.png" % f), view, ImageResolution=[1500, 780], TransparentBackground=0)
    print("wrote blade_%s.png" % f)
    skin.SetScalarBarVisibility(view, False)
Hide(reader, view)
if webs:
    Hide(webreader, view)

# ---- inboard webbed cutaway: clip both skin + webs to a spanwise band (r/R ~0.20-0.34) ----
ymid = 0.5 * (b[2] + b[3]); zmid = 0.5 * (b[4] + b[5])
xa = b[0] + 0.20 * (b[1] - b[0]); xb = b[0] + 0.34 * (b[1] - b[0])


def band_clip(src):
    c1 = Clip(Input=src, ClipType="Plane")
    c1.ClipType.Origin = [xa, ymid, zmid]; c1.ClipType.Normal = [1, 0, 0]; c1.Invert = 0
    c2 = Clip(Input=c1, ClipType="Plane")
    c2.ClipType.Origin = [xb, ymid, zmid]; c2.ClipType.Normal = [1, 0, 0]; c2.Invert = 1
    return c2


ks = band_clip(reader); kshow = Show(ks, view); kshow.Representation = "Surface"
colour(kshow, "RM_S11", "sigma11 (MPa)", 1, opacity=0.45)
if webreader:
    kw = band_clip(webreader); kwshow = Show(kw, view); kwshow.Representation = "Surface"
    colour(kwshow, "RM_S11", "sigma11 (MPa)", 1, opacity=1.0, bar=False, rescale=False)   # opaque webs in the cutaway
ks.UpdatePipeline(); Render(view)                                     # update pipeline before ResetCamera
iso((xa, xb, b[2], b[3], b[4], b[5]), 0.85, cdir=(0.6, -0.7, 0.55))   # explicit band bounds (no stale GetBounds)
Render(view)
SaveScreenshot(os.path.join(outdir, "blade_structure.png"), view, ImageResolution=[1500, 950], TransparentBackground=0)
print("wrote blade_structure.png (webbed cutaway)")
kshow.SetScalarBarVisibility(view, False); Hide(ks, view)
if webreader:
    Hide(kw, view)

# ---- deflected shape: undeformed (grey) + real-scale deformed skin + webs, coloured by |disp| ----
und = Show(reader, view); und.Representation = "Surface"; und.DiffuseColor = [0.72, 0.72, 0.72]; und.Opacity = 0.28
warp = WarpByVector(Input=reader); warp.Vectors = ["POINTS", "RM_disp"]; warp.ScaleFactor = 1.0
dw = Show(warp, view); dw.Representation = "Surface"
colour(dw, "RM_disp", "|disp| (m)", 0, opacity=0.45)
if webreader:
    wwarp = WarpByVector(Input=webreader); wwarp.Vectors = ["POINTS", "RM_disp"]; wwarp.ScaleFactor = 1.0
    dww = Show(wwarp, view); dww.Representation = "Surface"
    colour(dww, "RM_disp", "|disp| (m)", 0, opacity=1.0, bar=False, rescale=False)
UpdatePipeline()
iso(warp.GetDataInformation().GetBounds(), 0.72)
Render(view)
SaveScreenshot(os.path.join(outdir, "blade_deformed.png"), view, ImageResolution=[1500, 780], TransparentBackground=0)
print("wrote blade_deformed.png")
