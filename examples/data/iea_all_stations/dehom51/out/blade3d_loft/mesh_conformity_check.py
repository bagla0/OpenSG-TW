"""mesh_conformity_check.py -- LOCAL pvpython: verify + SHOW that the lofted blade hex mesh is conformal.

The skin loft (blade3d_hex51.vtk) is a pyvista StructuredGrid X,Y,Z[N,NT,NS] -> an implicitly-conformal
structured hex mesh (consecutive span planes, incl. the MPER interpolated planes between stations, share
their interface nodes; through-thickness layers likewise).  This prints the structured extent + cell count
and renders two edge-on views so conformality is visible:
  blade_mesh_front.png : look DOWN the span (-X) at a root slice -> the cross-section ring mesh (cut face)
  blade_mesh_iso.png   : zoom on an inboard band, element edges on (skin translucent + webs) -> the hex
                         cells connecting through-thickness AND span (no gaps / shared nodes).
"""
from paraview.simple import *
import os, sys

vtkfile = sys.argv[1]
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(vtkfile)
os.makedirs(outdir, exist_ok=True)
paraview.simple._DisableFirstRenderCameraReset()

reader = OpenDataFile(vtkfile); reader.UpdatePipeline()
di = reader.GetDataInformation()
b = di.GetBounds()
print("SKIN  class=%s  npts=%d  ncells=%d  extent(=N-1,NT-1,NS-1 dirs)=%s"
      % (di.GetDataClassName(), di.GetNumberOfPoints(), di.GetNumberOfCells(), di.GetExtent()))
webfile = os.path.join(os.path.dirname(vtkfile), "blade3d_webs51.vtk")
webreader = OpenDataFile(webfile) if os.path.exists(webfile) else None
if webreader:
    webreader.UpdatePipeline(); wdi = webreader.GetDataInformation()
    print("WEBS  class=%s  npts=%d  ncells=%d" % (wdi.GetDataClassName(), wdi.GetNumberOfPoints(), wdi.GetNumberOfCells()))

view = GetActiveViewOrCreate("RenderView")
view.ViewSize = [1300, 1000]
view.Background = [1, 1, 1]; view.UseColorPaletteForBackground = 0
view.OrientationAxesVisibility = 1; view.OrientationAxesLabelColor = [0, 0, 0]
view.CameraParallelProjection = 1


def symrange(field):
    ai = reader.GetPointDataInformation().GetArray(field)
    r = ai.GetComponentRange(0)
    return max(abs(r[0]), abs(r[1]), 1e-9)


def colorbar_field(rep, field, title, rescale=True):
    ColorBy(rep, ("POINTS", field))
    if rescale:
        rep.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(field); lut.ApplyPreset("Rainbow Uniform", True)
    m = symrange(field); lut.RescaleTransferFunction(-m, m)          # symmetric: 0 = green
    if rescale:                                                      # only the skin toggles the shared bar
        rep.SetScalarBarVisibility(view, True)                       # (webs share the LUT -> must NOT hide it)
        sb = GetScalarBar(lut, view)
        sb.Title = title; sb.ComponentTitle = ""
        sb.TitleColor = [0, 0, 0]; sb.LabelColor = [0, 0, 0]
        sb.TitleFontSize = 26; sb.LabelFontSize = 24                 # high-font colour bar
        sb.ScalarBarLength = 0.62
        sb.WindowLocation = "Any Location"; sb.Position = [0.85, 0.18]
    return lut


def band(src, fa, fb):
    xa = b[0] + fa * (b[1] - b[0]); xb = b[0] + fb * (b[1] - b[0])
    c1 = Clip(Input=src, ClipType="Plane"); c1.ClipType.Origin = [xa, 0, 0]; c1.ClipType.Normal = [1, 0, 0]; c1.Invert = 0
    c2 = Clip(Input=c1, ClipType="Plane"); c2.ClipType.Origin = [xb, 0, 0]; c2.ClipType.Normal = [1, 0, 0]; c2.Invert = 1
    return c2, xa, xb


# ---- FRONT view (look down -X): root slice cut-face = the cross-section ring mesh ----
cf, xa, xb = band(reader, 0.03, 0.05)
cf.UpdatePipeline()
s = Show(cf, view); s.Representation = "Surface With Edges"; s.EdgeColor = [0, 0, 0]; s.LineWidth = 1.0
colorbar_field(s, "RM_S11", "sigma11 (MPa)")
Render(view)
cam = GetActiveCamera(); ctr = [0.5 * (xa + xb), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]
cam.SetFocalPoint(*ctr); cam.SetPosition(ctr[0] - 10, ctr[1], ctr[2]); cam.SetViewUp(0, 0, 1)   # -X = front
ResetCamera(view); view.CameraParallelScale = view.CameraParallelScale * 0.9
Render(view)
SaveScreenshot(os.path.join(outdir, "blade_mesh_front.png"), view, ImageResolution=[1300, 1000], TransparentBackground=0)
print("wrote blade_mesh_front.png"); Hide(cf, view)

# ---- ISO zoom on an inboard band, edges on: skin (translucent) + webs ----
cs, xa2, xb2 = band(reader, 0.24, 0.285)
cs.UpdatePipeline()
sk = Show(cs, view); sk.Representation = "Surface With Edges"; sk.EdgeColor = [0.15, 0.15, 0.15]; sk.LineWidth = 1.0
sk.Opacity = 0.5; colorbar_field(sk, "RM_S11", "sigma11 (MPa)")
if webreader:
    cw, _, _ = band(webreader, 0.24, 0.285); cw.UpdatePipeline()
    wk = Show(cw, view); wk.Representation = "Surface With Edges"; wk.EdgeColor = [0.4, 0.0, 0.0]; wk.LineWidth = 1.0
    colorbar_field(wk, "RM_S11", "sigma11 (MPa)", rescale=False)
Render(view)
ctr2 = [0.5 * (xa2 + xb2), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5])]; dd = (xb2 - xa2)
cam = GetActiveCamera(); cam.SetFocalPoint(*ctr2)
cam.SetPosition(ctr2[0] + 2.4 * dd, ctr2[1] - 2.8 * dd, ctr2[2] + 2.0 * dd); cam.SetViewUp(0, 0, 1)
ResetCamera(view); view.CameraParallelScale = view.CameraParallelScale * 0.8
Render(view)
SaveScreenshot(os.path.join(outdir, "blade_mesh_iso.png"), view, ImageResolution=[1300, 1000], TransparentBackground=0)
print("wrote blade_mesh_iso.png")
