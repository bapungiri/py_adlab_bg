"""Generate a guide-plate calibration coupon matching guide_array_64ch_Lshaped_mPFC_OFC_BG.

Layout (top view, X to the right):
  - A thin plate ('PLATE_T', same as the guide array plates) held in a thicker frame
    ('FRAME_H') so the coupon can be handled.
  - 'len(HOLE_DIAMETERS)' groups of 2 x 4 holes at 'CELL' pitch (0.4 mm in the original
    guide array, 0.47 mm in mod2), smallest diameter on the left.
  - The frame corner at the small-hole end is cut down to plate height as an
    orientation notch.

Same watertight-grid approach as make_hole_coupon.py, extended to cells of
different heights: vertical walls are added wherever a cell is taller than its
neighbour, and every wall is split at all height levels so no T-junctions occur.

Usage:
  python make_plate_coupon.py            -> writes plate_coupon_p<pitch in um>.stl next to this script
"""

from pathlib import Path

import numpy as np

from make_hole_coupon import Mesh, check_watertight, signed_volume, write_stl

# ---------------- parameters (mm) ----------------
HOLE_DIAMETERS = [0.32, 0.34, 0.36, 0.38, 0.40]
GROUP_COLS, GROUP_ROWS = 4, 2  # holes per group (like the 2 x 4 block in the guide array)
CELL = 0.47  # hole pitch; wall = CELL - diameter (0.47 = min pitch of guide array mod2)
PLATE_T = 0.3  # plate thickness (guide array plates are 0.3 mm)
FRAME_H = 2.0  # frame height
FRAME_W = 3  # frame width in cells
MARGIN = 2  # solid plate cells between holes and frame
GAP = 2  # solid plate cells between hole groups
NOTCH = 2  # notch size in cells (must be < FRAME_W, else frame parts touch at a line)
N_SIDE = 8  # points per cell edge -> 4*N_SIDE segments per hole

OUT = Path(__file__).with_name(f"plate_coupon_p{round(CELL * 1000):03d}.stl")


def build_layout():
    """Return (height, diameter) arrays of shape (rows, cols)."""
    n_groups = len(HOLE_DIAMETERS)
    inner_cols = 2 * MARGIN + n_groups * GROUP_COLS + (n_groups - 1) * GAP
    inner_rows = 2 * MARGIN + GROUP_ROWS
    n_cols, n_rows = inner_cols + 2 * FRAME_W, inner_rows + 2 * FRAME_W
    height = np.full((n_rows, n_cols), FRAME_H)
    height[FRAME_W:-FRAME_W, FRAME_W:-FRAME_W] = PLATE_T
    height[:NOTCH, :NOTCH] = PLATE_T
    diam = np.zeros((n_rows, n_cols))
    r0 = FRAME_W + MARGIN
    for i, d in enumerate(HOLE_DIAMETERS):
        c0 = FRAME_W + MARGIN + i * (GROUP_COLS + GAP)
        diam[r0 : r0 + GROUP_ROWS, c0 : c0 + GROUP_COLS] = d
    return height, diam


def edge_points(x0, y0):
    """The 4 CCW edges of a cell, each as N_SIDE+1 points (corners included)."""
    t = np.linspace(0, CELL, N_SIDE + 1)
    return [
        np.c_[x0 + t, np.full_like(t, y0)],  # bottom (neighbour r-1)
        np.c_[np.full_like(t, x0 + CELL), y0 + t],  # right (neighbour c+1)
        np.c_[x0 + CELL - t, np.full_like(t, y0 + CELL)],  # top (neighbour r+1)
        np.c_[np.full_like(t, x0), y0 + CELL - t],  # left (neighbour c-1)
    ]


def build_mesh(height, diam):
    m = Mesh()
    n_rows, n_cols = height.shape
    levels = np.unique(np.r_[0.0, height.ravel()])
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = c * CELL, r * CELL
            cx, cy = x0 + CELL / 2, y0 + CELL / 2
            h, d = height[r, c], diam[r, c]
            edges = edge_points(x0, y0)
            B = np.vstack([e[:-1] for e in edges])
            n = len(B)
            ang = np.arctan2(B[:, 1] - cy, B[:, 0] - cx)
            ring_xy = [(cx + d / 2 * np.cos(a), cy + d / 2 * np.sin(a)) for a in ang]
            for z, flip in ((h, False), (0.0, True)):
                bi = [m.v(px, py, z) for px, py in B]
                if d == 0:
                    ci = m.v(cx, cy, z)
                    for k in range(n):
                        m.tri(ci, bi[k], bi[(k + 1) % n], flip)
                else:
                    ring = [m.v(px, py, z) for px, py in ring_xy]
                    for k in range(n):
                        k1 = (k + 1) % n
                        m.quad(bi[k], bi[k1], ring[k1], ring[k], flip)
            if d > 0:
                for k in range(n):
                    (xa, ya), (xb, yb) = ring_xy[k], ring_xy[(k + 1) % n]
                    m.quad(m.v(xa, ya, 0), m.v(xa, ya, h), m.v(xb, yb, h), m.v(xb, yb, 0))
            # step / outer walls where this cell is taller than its neighbour
            for e, (dr, dc) in zip(edges, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
                rr, cc = r + dr, c + dc
                h_nb = height[rr, cc] if 0 <= rr < n_rows and 0 <= cc < n_cols else 0.0
                if h <= h_nb:
                    continue
                zs = levels[(levels >= h_nb) & (levels <= h)]
                for za, zb in zip(zs[:-1], zs[1:]):
                    for k in range(N_SIDE):
                        (xa, ya), (xb, yb) = e[k], e[k + 1]
                        m.quad(m.v(xa, ya, za), m.v(xb, yb, za), m.v(xb, yb, zb), m.v(xa, ya, zb))
    return np.array(m.verts), np.array(m.tris)


if __name__ == "__main__":
    height, diam = build_layout()
    V, T = build_mesh(height, diam)
    assert check_watertight(T), "mesh is not watertight"
    size = V.max(0) - V.min(0)
    expected = (height * CELL**2).sum() - sum(np.pi * (d / 2) ** 2 * PLATE_T for d in diam.ravel() if d)
    print(f"coupon {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm, {len(T)} triangles")
    print(f"volume {signed_volume(V, T):.3f} mm^3 (analytic ~{expected:.3f})")
    for d in HOLE_DIAMETERS:
        print(f"  hole {d:.3f} mm -> wall {CELL - d:.3f} mm")
    write_stl(OUT, V, T)
    print(f"wrote {OUT}")
