"""Nosepoke front plate as a printable STL, from the Onshape sketch (DXF).

The sketch is a 100.4 x 48.6 mm plate with two ~19.7 mm nosepoke holes, each
with a slot above and below for the IR beam sensor pair, and two M3 (3.4 mm)
mounting holes. Here each nosepoke hole's outline (hole + both slots) is
rotated about the hole centre so the sensor slots sit 'tilt' degrees off
vertical: bedding piles up straight below the hole, so a vertical bottom slot
fills first. The ports are mirrored (left clockwise, right anticlockwise), so
both bottom slots lean toward the plate's outer edges and the beam still
crosses the hole centre. The plate is extruded 'thickness' mm.

Two steps, because ezdxf and manifold3d live in different environments:
  extract (needs ezdxf, e.g. the data_analysis conda env):
    python make_nosepoke_plate.py extract "<sketch>.dxf"   -> nosepoke_plate_outline.json
  build (needs numpy + manifold3d, e.g. the GuideArrayDesigner venv):
    python make_nosepoke_plate.py build [--tilt 30] [--thickness 6]
      -> nosepoke_plate_t<thickness>_tilt<tilt>.stl
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTLINE = HERE / "nosepoke_plate_outline.json"
TOL = 0.01  # mm: endpoints closer than this are the same point (Onshape's export leaves ~1.5 um gaps)


# ---------------------------------------------------------------- extract --- #
def extract(dxf_path):
    import ezdxf

    msp = ezdxf.readfile(dxf_path).modelspace()
    pieces, circles = [], []
    for e in msp:
        kind = e.dxftype()
        if kind == "LINE":
            pieces.append([tuple(e.dxf.start)[:2], tuple(e.dxf.end)[:2]])
        elif kind == "SPLINE":
            pieces.append([tuple(p)[:2] for p in e.flattening(0.005)])
        elif kind == "CIRCLE":
            circles.append(dict(center=list(e.dxf.center)[:2], radius=e.dxf.radius))
        else:
            sys.exit("unexpected entity in the sketch: %s" % kind)
    loops = chain(pieces)
    loops.sort(key=lambda p: -area(p))  # outer outline first
    data = dict(source=str(dxf_path), units="mm", outer=loops[0], holes=loops[1:], circles=circles)
    OUTLINE.write_text(json.dumps(data))
    print("%s: outer outline + %d nosepoke holes + %d round holes" % (OUTLINE.name, len(loops) - 1, len(circles)))


def chain(pieces):
    """Join open polylines end to end into closed loops."""

    def same(a, b):
        return abs(a[0] - b[0]) < TOL and abs(a[1] - b[1]) < TOL

    pieces = [list(p) for p in pieces]
    loops = []
    while pieces:
        loop = pieces.pop(0)
        while not same(loop[0], loop[-1]):
            for i, p in enumerate(pieces):
                if same(loop[-1], p[0]):
                    loop += p[1:]
                elif same(loop[-1], p[-1]):
                    loop += p[::-1][1:]
                else:
                    continue
                pieces.pop(i)
                break
            else:
                sys.exit("sketch outline is not closed near %s" % (loop[-1],))
        loops.append(loop[:-1])
    return loops


def area(p):
    p = np.asarray(p)
    return abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1))) / 2


# ------------------------------------------------------------------ build --- #
def hole_centre(loop):
    """Centre of the round part of a nosepoke hole: least-squares circle through
    the points that are not on the slots (the slots stick out past the circle)."""

    p = np.asarray(loop)
    c = (p.min(0) + p.max(0)) / 2
    r = np.hypot(*(p - c).T)
    round_part = p[np.abs(r - np.median(r)) < 0.5]
    a = np.c_[2 * round_part, np.ones(len(round_part))]
    b = (round_part ** 2).sum(1)
    cx, cy, k = np.linalg.lstsq(a, b, rcond=None)[0]
    return np.array([cx, cy]), math.sqrt(k + cx * cx + cy * cy)


def rotate(loop, centre, deg):
    t = math.radians(deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])
    return (np.asarray(loop) - centre) @ rot.T + centre


def build(tilt, thickness):
    import manifold3d as m3d

    data = json.loads(OUTLINE.read_text())
    outer = np.asarray(data["outer"])
    plate_mid = (outer[:, 0].min() + outer[:, 0].max()) / 2
    cut = []
    for loop in data["holes"]:
        centre, radius = hole_centre(loop)
        # mirrored: left port clockwise (bottom slot leans left), right anticlockwise
        deg = -tilt if centre[0] < plate_mid else tilt
        cut.append(rotate(loop, centre, deg))
        print("nosepoke hole at (%.2f, %.2f) mm, diameter %.2f mm, slots rotated %+g deg"
              % (centre[0], centre[1], 2 * radius, deg))
    for c in data["circles"]:
        n = 96
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        cut.append(np.c_[c["center"][0] + c["radius"] * np.cos(a), c["center"][1] + c["radius"] * np.sin(a)])
        print("mounting hole at (%.2f, %.2f) mm, diameter %.2f mm" % (c["center"][0], c["center"][1], 2 * c["radius"]))

    shape = m3d.CrossSection([outer.tolist()], m3d.FillRule.EvenOdd)
    for loop in cut:
        shape = shape - m3d.CrossSection([np.asarray(loop).tolist()], m3d.FillRule.EvenOdd)
    solid = m3d.Manifold.extrude(shape, thickness)
    # move to the build plate origin: corner at (0, 0, 0)
    lo = np.asarray(solid.bounding_box()[:3])
    solid = solid.translate((-lo).tolist())

    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties)[:, :3]
    tris = np.asarray(mesh.tri_verts)
    out = HERE / ("nosepoke_plate_t%g_tilt%g.stl" % (thickness, tilt))
    write_binary_stl(out, verts[tris])
    bb = solid.bounding_box()
    print("%s: %d triangles, %.1f x %.1f x %.1f mm, volume %.2f cm3, genus %d"
          % (out.name, len(tris), bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2], solid.volume() / 1000, solid.genus()))
    return out


def write_binary_stl(path, faces):
    faces = np.asarray(faces, dtype=np.float32)
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = normals, faces
    with open(path, "wb") as fh:
        fh.write(b"nosepoke plate".ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        fh.write(rec.tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("dxf")
    bd = sub.add_parser("build")
    bd.add_argument("--tilt", type=float, default=30.0, help="sensor slots, degrees off vertical")
    bd.add_argument("--thickness", type=float, default=6.0, help="plate thickness, mm")
    args = ap.parse_args()
    if args.step == "extract":
        extract(args.dxf)
    else:
        build(args.tilt, args.thickness)
