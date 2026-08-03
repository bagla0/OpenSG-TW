"""render_cutaway_row.py -- LOCAL pvpython (ParaView 5.12): render the inboard conformal
cutaway BAND of the IEA-22 blade, coloured by each of the three in-plane stress
components (RM_S11, RM_S22, RM_S12), each with element edges and its own colour bar, for
the separate 3-component cutaway-segment figure of the CPB paper.

  & "C:\\Program Files\\ParaView-5.12.0-...\\bin\\pvpython.exe" render_cutaway_row.py <conformal.vtk> <outdir> [fa] [fb]

Writes cut_S11.png, cut_S22.png, cut_S12.png (+ prints the r/R span of the band).
The band fractions default to 0.24..0.29 of the span (r/R ~ 0.24-0.29, the inboard
spar-cap region carrying the peak flap-bending stress).
"""
from paraview.simple import *
import os
import sys

vtkfile = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(vtkfile)
FA = float(sys.argv[3]) if len(sys.argv) > 3 else 0.24
FB = float(sys.argv[4]) if len(sys.argv) > 4 else 0.29
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

reader = OpenDataFile(vtkfile)
reader.UpdatePipeline()
di = reader.GetDataInformation()
b = di.GetBounds()
L = b[1] - b[0]
xa = b[0] + FA * L
xb = b[0] + FB * L
print("bounds x=[%.3f, %.3f]  band r/R = %.3f .. %.3f  (x=%.2f..%.2f m)"
      % (b[0], b[1], (xa - b[0]) / L, (xb - b[0]) / L, xa, xb), flush=True)

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1300, 950]
view.Background = [1, 1, 1]
view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 0            # single triad added in the composite
view.CameraParallelProjection = 1


def symrange(field):
    ai = reader.GetPointDataInformation().GetArray(field)
    r = ai.GetComponentRange(0)
    return max(abs(r[0]), abs(r[1]), 1e-9)


# clip the band [xa, xb]
c1 = Clip(Input=reader, ClipType="Plane")
c1.ClipType.Origin = [xa, 0, 0]
c1.ClipType.Normal = [1, 0, 0]
c1.Invert = 0
c2 = Clip(Input=c1, ClipType="Plane")
c2.ClipType.Origin = [xb, 0, 0]
c2.ClipType.Normal = [1, 0, 0]
c2.Invert = 1
c2.UpdatePipeline()
cb = c2.GetDataInformation().GetBounds()
ctr = [0.5 * (cb[0] + cb[1]), 0.5 * (cb[2] + cb[3]), 0.5 * (cb[4] + cb[5])]
dd = cb[1] - cb[0]

rep = Show(c2, view)
rep.Representation = "Surface With Edges"
rep.EdgeColor = [0.12, 0.12, 0.12]
rep.LineWidth = 1.0
rep.Opacity = 0.55

FIELDS = [("RM_S11", r"$\sigma_{11}$ (MPa)"), ("RM_S22", r"$\sigma_{22}$ (MPa)"),
          ("RM_S12", r"$\sigma_{12}$ (MPa)")]
prev_lut = None
for field, title in FIELDS:
    ColorBy(rep, ("POINTS", field))
    if prev_lut is not None:
        HideScalarBarIfNotNeeded(prev_lut, view)      # drop the previous field's bar
    lut = GetColorTransferFunction(field)
    lut.ApplyPreset("Rainbow Uniform", True)
    m = symrange(field)
    lut.RescaleTransferFunction(-m, m)
    rep.SetScalarBarVisibility(view, True)
    sb = GetScalarBar(lut, view)
    sb.Title = title
    sb.ComponentTitle = ""
    sb.TitleColor = [0, 0, 0]
    sb.LabelColor = [0, 0, 0]
    sb.TitleFontSize = 22
    sb.LabelFontSize = 20
    sb.ScalarBarLength = 0.6
    sb.WindowLocation = "Any Location"
    sb.Position = [0.86, 0.2]
    cam = GetActiveCamera()
    cam.SetFocalPoint(*ctr)
    cam.SetPosition(ctr[0] + 2.4 * dd, ctr[1] - 2.8 * dd, ctr[2] + 2.0 * dd)
    cam.SetViewUp(0, 0, 1)
    ResetCamera(view)
    view.CameraParallelScale = view.CameraParallelScale * 0.82
    Render(view)
    out = os.path.join(outdir, "cut_%s.png" % field.replace("RM_", ""))
    SaveScreenshot(out, view, ImageResolution=[1300, 950], TransparentBackground=0)
    print("wrote", out, flush=True)
    prev_lut = lut
