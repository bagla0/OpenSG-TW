"""render_conformal.py -- LOCAL pvpython: render the single CONFORMAL webbed loft (blade3d_conformal.vtk).

Shows that the webs meet the skin through SHARED nodes (one watertight mesh): an inboard cutaway band and
a front cross-section, both with element edges on and a high-font colour bar (sigma11).  Also the 6 field
views + deformed on the whole conformal blade.
  & pvpython render_conformal.py <blade3d_conformal.vtk> <outdir>
"""
from paraview.simple import *
import os, sys

vtkfile = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(vtkfile)
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

FIELDS = [("RM_S11", "$\sigma_{11}$ (MPa)", 1), ("RM_S22", "$\sigma_{22}$ (MPa)", 1), ("RM_S12", "$\sigma_{12}$ (MPa)", 1),
          ("RM_u1", "$u_1$ out-of-plane (m)", 0), ("RM_u2", "$u_2$ chordwise (m)", 0), ("RM_u3", "$u_3$ flapwise (m)", 0)]

reader = OpenDataFile(vtkfile); reader.UpdatePipeline()
di = reader.GetDataInformation(); b = di.GetBounds()
print("CONFORMAL class=%s npts=%d ncells=%d" % (di.GetDataClassName(), di.GetNumberOfPoints(), di.GetNumberOfCells()))
view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1500, 1000]; view.Background = [1, 1, 1]; view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 1; view.OrientationAxesLabelColor = [0, 0, 0]; view.CameraParallelProjection = 1


def symrange(field):
    ai = reader.GetPointDataInformation().GetArray(field)
    r = ai.GetComponentRange(0)
    return max(abs(r[0]), abs(r[1]), 1e-9)


def colour(rep, field, title, symmetric, bar=True):
    ColorBy(rep, ("POINTS", field))
    rep.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(field); lut.ApplyPreset("Rainbow Uniform", True)
    if symmetric:
        m = symrange(field); lut.RescaleTransferFunction(-m, m)
    if bar:
        rep.SetScalarBarVisibility(view, True)
        sb = GetScalarBar(lut, view)
        sb.Title = title; sb.ComponentTitle = ""
        sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
        sb.TitleFontSize = 26; sb.LabelFontSize = 24
        sb.ScalarBarLength = 0.62; sb.WindowLocation = "Any Location"; sb.Position = [0.85, 0.18]
    return lut


def band(fa, fb):
    xa = b[0] + fa * (b[1] - b[0]); xb = b[0] + fb * (b[1] - b[0])
    c1 = Clip(Input=reader, ClipType="Plane"); c1.ClipType.Origin = [xa, 0, 0]; c1.ClipType.Normal = [1, 0, 0]; c1.Invert = 0
    c2 = Clip(Input=c1, ClipType="Plane"); c2.ClipType.Origin = [xb, 0, 0]; c2.ClipType.Normal = [1, 0, 0]; c2.Invert = 1
    c2.UpdatePipeline(); return c2, xa, xb


def savecam(name, ctr, pos, zoom):
    cam = GetActiveCamera(); cam.SetFocalPoint(*ctr); cam.SetPosition(*pos); cam.SetViewUp(0, 0, 1)
    ResetCamera(view); view.CameraParallelScale = view.CameraParallelScale * zoom
    Render(view); SaveScreenshot(os.path.join(outdir, name), view, ImageResolution=[1500, 1000], TransparentBackground=0)
    print("wrote", name)


# ---- CUTAWAY band, edges on, S11 + high-font colour bar (skin translucent so webs read) ----
c2, xa, xb = band(0.24, 0.29)
s = Show(c2, view); s.Representation = "Surface With Edges"; s.EdgeColor = [0.12, 0.12, 0.12]; s.LineWidth = 1.0; s.Opacity = 0.55
colour(s, "RM_S11", "$\sigma_{11}$ (MPa)", 1)
ctr = [0.5 * (xa + xb), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]; dd = xb - xa
savecam("blade_conformal_iso.png", ctr, [ctr[0] + 2.4 * dd, ctr[1] - 2.8 * dd, ctr[2] + 2.0 * dd], 0.8)
Hide(c2, view)

# ---- FRONT cross-section, edges on (webs meet skin at shared nodes) ----
cf, xa, xb = band(0.03, 0.05)
sf = Show(cf, view); sf.Representation = "Surface With Edges"; sf.EdgeColor = [0, 0, 0]; sf.LineWidth = 1.0
colour(sf, "RM_S11", "$\sigma_{11}$ (MPa)", 1)
ctr = [0.5 * (xa + xb), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
savecam("blade_conformal_front.png", ctr, [ctr[0] - 10, ctr[1], ctr[2]], 0.9)
Hide(cf, view)

# ---- 6 field views + deformed on the whole conformal blade (translucent, webs read through) ----
full = Show(reader, view); full.Representation = "Surface"; full.Opacity = 0.6
for f, title, sym in FIELDS:
    colour(full, f, title, sym)
    savecam("blade_conf_%s.png" % f, [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])],
            [0.5 * (b[0] + b[1]) + 0.30 * (b[1] - b[0]), 0.5 * (b[2] + b[3]) - 0.45 * (b[1] - b[0]),
             0.5 * (b[4] + b[5]) + 0.85 * (b[1] - b[0])], 0.58)
    full.SetScalarBarVisibility(view, False)
Hide(reader, view)

und = Show(reader, view); und.Representation = "Surface"; und.DiffuseColor = [0.72, 0.72, 0.72]; und.Opacity = 0.28
warp = WarpByVector(Input=reader); warp.Vectors = ["POINTS", "RM_disp"]; warp.ScaleFactor = 1.0
dw = Show(warp, view); dw.Representation = "Surface"; dw.Opacity = 0.6
colour(dw, "RM_disp", "$|u|$ (m)", 0)
warp.UpdatePipeline(); wb = warp.GetDataInformation().GetBounds()
savecam("blade_conf_deformed.png", [0.5 * (wb[0] + wb[1]), 0.5 * (wb[2] + wb[3]), 0.5 * (wb[4] + wb[5])],
        [0.5 * (wb[0] + wb[1]) + 0.55 * (wb[1] - wb[0]), 0.5 * (wb[2] + wb[3]) - 0.75 * (wb[1] - wb[0]),
         0.5 * (wb[4] + wb[5]) + 0.55 * (wb[1] - wb[0])], 0.72)

