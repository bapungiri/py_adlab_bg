"""Generate a resin-printer hole-size calibration coupon as a watertight binary STL.

Layout (top view, X to the right):
  - 8 columns of test holes, diameters from 'HOLE_DIAMETERS' (smallest on the left).
  - Each column has 3 holes at 'CELL' pitch so the walls between holes are as thin
    as in a real guide array.
  - A single larger marker hole ('MARKER_D') left of the smallest column shows orientation.

The block is built from a grid of square cells. Every cell edge is sampled with the
same points, so neighbouring cells share vertices and the mesh is watertight
without any boolean operations.

Usage:
  python make_hole_coupon.py            -> writes hole_coupon.stl next to this script
"""

from pathlib import Path

import numpy as np

# ---------------- parameters (mm) ----------------
HOLE_DIAMETERS = [0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.45]
HOLES_PER_SIZE = 3
CELL = 0.8  # hole pitch (centre to centre); wall = CELL - diameter
HEIGHT = 1.0  # channel length / block thickness
MARKER_D = 0.6
BORDER = 2  # empty cells around the hole area
N_SIDE = 12  # points per cell edge -> 4*N_SIDE segments per hole

OUT = Path(__file__).with_name("hole_coupon.stl")


def build_layout():
    """Return a (rows, cols) array of hole diameters, 0 = solid cell."""
    n_cols = 2 * BORDER + 2 * len(HOLE_DIAMETERS) - 1
    n_rows = 2 * BORDER + HOLES_PER_SIZE
    grid = np.zeros((n_rows, n_cols))
    for i, d in enumerate(HOLE_DIAMETERS):
        col = BORDER + 2 * i
        grid[BORDER : BORDER + HOLES_PER_SIZE, col] = d
    grid[BORDER + HOLES_PER_SIZE // 2, BORDER - 1] = MARKER_D
    return grid


def cell_boundary(x0, y0):
    """CCW boundary points of a square cell, 4*N_SIDE points starting at (x0, y0)."""
    t = np.arange(N_SIDE) / N_SIDE * CELL
    bottom = np.c_[x0 + t, np.full(N_SIDE, y0)]
    right = np.c_[np.full(N_SIDE, x0 + CELL), y0 + t]
    top = np.c_[x0 + CELL - t, np.full(N_SIDE, y0 + CELL)]
    left = np.c_[np.full(N_SIDE, x0), y0 + CELL - t]
    return np.vstack([bottom, right, top, left])


class Mesh:
    def __init__(self):
        self.verts, self.index, self.tris = [], {}, []

    def v(self, x, y, z):
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key not in self.index:
            self.index[key] = len(self.verts)
            self.verts.append(key)
        return self.index[key]

    def tri(self, a, b, c, flip=False):
        self.tris.append((a, c, b) if flip else (a, b, c))

    def quad(self, a, b, c, d, flip=False):
        self.tri(a, b, c, flip)
        self.tri(a, c, d, flip)


def build_mesh(grid):
    m = Mesh()
    n_rows, n_cols = grid.shape
    for r in range(n_rows):
        for c in range(n_cols):
            x0, y0 = c * CELL, r * CELL
            cx, cy = x0 + CELL / 2, y0 + CELL / 2
            B = cell_boundary(x0, y0)
            n = len(B)
            d = grid[r, c]
            for z, flip in ((HEIGHT, False), (0.0, True)):  # top faces +z, bottom -z
                bi = [m.v(px, py, z) for px, py in B]
                if d == 0:
                    ci = m.v(cx, cy, z)
                    for k in range(n):
                        m.tri(ci, bi[k], bi[(k + 1) % n], flip)
                else:
                    ang = np.arctan2(B[:, 1] - cy, B[:, 0] - cx)
                    ring = [
                        m.v(cx + d / 2 * np.cos(a), cy + d / 2 * np.sin(a), z)
                        for a in ang
                    ]
                    for k in range(n):
                        k1 = (k + 1) % n
                        m.quad(bi[k], bi[k1], ring[k1], ring[k], flip)
            if d > 0:  # hole wall, normals pointing into the hole
                ang = np.arctan2(B[:, 1] - cy, B[:, 0] - cx)
                pts = [(cx + d / 2 * np.cos(a), cy + d / 2 * np.sin(a)) for a in ang]
                for k in range(n):
                    (xa, ya), (xb, yb) = pts[k], pts[(k + 1) % n]
                    m.quad(
                        m.v(xa, ya, 0),
                        m.v(xa, ya, HEIGHT),
                        m.v(xb, yb, HEIGHT),
                        m.v(xb, yb, 0),
                    )

    # outer side walls, following the CCW outline of the whole block
    W, H = n_cols * CELL, n_rows * CELL
    t_x = np.arange(n_cols * N_SIDE) / N_SIDE * CELL
    t_y = np.arange(n_rows * N_SIDE) / N_SIDE * CELL
    outline = np.vstack(
        [
            np.c_[t_x, np.zeros_like(t_x)],
            np.c_[np.full_like(t_y, W), t_y],
            np.c_[W - t_x, np.full_like(t_x, H)],
            np.c_[np.zeros_like(t_y), H - t_y],
        ]
    )
    n = len(outline)
    for k in range(n):
        (xa, ya), (xb, yb) = outline[k], outline[(k + 1) % n]
        m.quad(m.v(xa, ya, 0), m.v(xb, yb, 0), m.v(xb, yb, HEIGHT), m.v(xa, ya, HEIGHT))
    return np.array(m.verts), np.array(m.tris)


def check_watertight(tris):
    edges = {}
    for a, b, c in tris:
        for e in ((a, b), (b, c), (c, a)):
            edges[e] = edges.get(e, 0) + 1
    bad = [e for e, k in edges.items() if k != 1 or edges.get((e[1], e[0]), 0) != 1]
    return len(bad) == 0


def signed_volume(V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6


def write_stl(path, V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    rec = np.zeros(
        len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]
    )
    rec["n"], rec["v"] = nrm, np.stack([a, b, c], axis=1)
    with open(path, "wb") as f:
        f.write(b"hole calibration coupon".ljust(80, b" "))
        f.write(np.uint32(len(T)).tobytes())
        f.write(rec.tobytes())


if __name__ == "__main__":
    grid = build_layout()
    V, T = build_mesh(grid)
    assert check_watertight(T), "mesh is not watertight"
    vol = signed_volume(V, T)
    size = V.max(0) - V.min(0)
    expected = size[0] * size[1] * HEIGHT - sum(
        np.pi * (d / 2) ** 2 * HEIGHT for d in grid.ravel() if d
    )
    print(f"block {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f} mm, {len(T)} triangles")
    print(f"volume {vol:.3f} mm^3 (analytic ~{expected:.3f})")
    write_stl(OUT, V, T)
    print(f"wrote {OUT}")
