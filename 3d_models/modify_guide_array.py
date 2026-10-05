"""Modify guide_array_64ch_Lshaped_mPFC_OFC_BG.stl for printing on the Sonic Mini 8K.

1. Re-lay the 16 guide holes on both plates with wider spacing, so the walls between
   holes are thick enough to print (the original 0.4 mm pitch leaves 0.05-0.10 mm walls):
     - mPFC: two staggered columns along AP (y), 'MPFC_ML_SEP' apart in ML (x),
       'MPFC_PITCH' along each column, staggered by half a pitch. Group centre kept.
     - OFC: 2 x 4 block, 'OFC_PITCH' in both directions. Group centre kept, except that
       the block is moved along +y just enough to keep 'MIN_PITCH' from the mPFC holes.
   Old holes are filled and new ones drilled (boolean ops), so the plates stay aligned.
2. Add vertical pillars between the bottom plate (top face z = -3.5) and the top
   plate (underside z = 0.2) so the top plate no longer bridges the 6 mm cavity.
   Pillars keep 'PILLAR_CLEARANCE' from every hole edge, i.e. from the tetrode paths.

Axes of the source STL: y = AP (columns of the mPFC group), x = ML.

Needs numpy + manifold3d (not in the lab conda envs; use a separate venv).

With '--pills', neighbouring holes are merged in pairs into pill-shaped slots (one slot
holds two guide tubes): mPFC (1,2) (3,4) (5,6) (7,8) along the zig-zag, OFC (1,2) (3,4) ...
along ML within each row. Hole numbering as in the .holes.csv (posterior -> anterior,
medial -> lateral).

Usage:
  python modify_guide_array.py           -> guide_array_64ch_Lshaped_mPFC_OFC_BG_mod2_h<hole um>.stl
  python modify_guide_array.py --pills   -> guide_array_64ch_Lshaped_mPFC_OFC_BG_mod2_h<hole um>_pills.stl
"""

import sys
from pathlib import Path

import manifold3d as m3d
import numpy as np

HERE = Path(__file__).parent
SRC = HERE / "guide_array_64ch_Lshaped_mPFC_OFC_BG.stl"
PILLS = "--pills" in sys.argv

# ---- source geometry ----
OLD_HOLE_D = 0.35
PLATES_Z = [(-3.8, -3.5), (0.2, 0.5)]  # hole plates (z ranges)
CAVITY_R = 3.0  # inner radius of the cavity between the plates
CAVITY_Z = (-3.5, 0.2)  # bottom-plate top face, top-plate underside
OFC_Y_MIN = 1.2  # source holes with y above this belong to the OFC block

# ---- new hole layout (mm) ----
NEW_HOLE_D = 0.35  # guide tubes are slightly under 0.30 mm OD; 0.32 printed closed at 30 um / 1.8 s, 0.36 opened at 20 um / 1.4 s
TUBE_OD = 0.29  # only used to report how far tubes can move inside a pill
OUT = HERE / f"guide_array_64ch_Lshaped_mPFC_OFC_BG_mod2_h{round(NEW_HOLE_D * 1000):03d}{'_pills' if PILLS else ''}.stl"
MPFC_PITCH = 0.5  # along AP within each column
MPFC_ML_SEP = 0.4  # between the two columns (kept from the source)
OFC_PITCH = 0.5  # both directions
MIN_PITCH = 0.47  # min centre distance between the mPFC and OFC groups

# ---- bregma mapping (STL +y = anterior, +x = lateral, no rotation) ----
BREGMA_REF = (2.0, 0.75)  # (AP, ML) of the most posterior hole (bottom of the medial mPFC column)
OFC_ML_START = 0.75  # bregma ML of the most medial OFC column (None = keep source centre)

# ---- pillars ----
PILLAR_D = 0.6
PILLAR_CLEARANCE = 0.4  # min gap between pillar and hole edge
PILLAR_WALL_GAP = 0.2  # min gap between pillar and cavity wall
PILLAR_GAP = 0.3  # min gap between neighbouring pillars
TARGET_SPAN = 1.6  # max unsupported span of the top plate (where pillars are allowed)
OVERLAP = 0.05  # pillar overlap into each plate, for a clean union


def read_stl(path):
    raw = path.read_bytes()
    n = np.frombuffer(raw[80:84], "<u4")[0]
    rec = np.frombuffer(raw[84 : 84 + 50 * n], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    tri = rec["v"].astype(float).reshape(-1, 3)
    V, idx = np.unique(np.round(tri, 5), axis=0, return_inverse=True)
    return V, idx.reshape(-1, 3)


def write_stl(path, V, T):
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(T), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"], rec["v"] = nrm, np.stack([a, b, c], axis=1)
    with open(path, "wb") as f:
        f.write(b"guide array, re-spaced holes + cavity pillars".ljust(80, b" "))
        f.write(np.uint32(len(T)).tobytes())
        f.write(rec.tobytes())


def face_normals(V, T):
    return np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])


def find_hole_centres(V, T):
    """Centres of the guide holes on the bottom plate: each hole wall is a connected set of vertical triangles."""
    z0, z1 = PLATES_Z[0]
    n = face_normals(V, T)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    zc = V[T][:, :, 2].mean(1)
    rc = np.hypot(*V[T][:, :, :2].mean(1).T)
    wall = (np.abs(n[:, 2]) < 1e-3) & (zc > z0) & (zc < z1) & (rc < CAVITY_R - 0.5)
    parent = {}

    def find(a):
        while parent.setdefault(a, a) != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, c in T[wall]:
        parent[find(b)] = find(a)
        parent[find(c)] = find(a)
    groups = {}
    for vi in parent:
        groups.setdefault(find(vi), []).append(vi)
    centres = []
    for g in groups.values():  # least-squares circle fit (vertex mean is biased by uneven spacing)
        x, y = V[g, 0], V[g, 1]
        A = np.c_[2 * x, 2 * y, np.ones_like(x)]
        cx, cy, _ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)[0]
        centres.append((cx, cy))
    return np.array(centres)


def min_pitch(pts):
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(D, np.inf)
    return D.min()


def new_layout(old):
    """New hole centres: (mpfc, ofc, ofc_shift)."""
    ofc_old, mpfc_old = old[old[:, 1] > OFC_Y_MIN], old[old[:, 1] <= OFC_Y_MIN]
    # mPFC: two columns; the source stagger has the +x column higher
    cols = np.unique(np.round(mpfc_old[:, 0], 3))
    assert len(cols) == 2, cols
    xc, yc = cols.mean(), (mpfc_old[:, 1].max() + mpfc_old[:, 1].min()) / 2
    span = 3.5 * MPFC_PITCH
    top = yc + span / 2
    k = np.arange(4)
    mpfc = np.r_[
        np.c_[np.full(4, xc + MPFC_ML_SEP / 2), top - k * MPFC_PITCH],
        np.c_[np.full(4, xc - MPFC_ML_SEP / 2), top - MPFC_PITCH / 2 - k * MPFC_PITCH],
    ]
    # OFC: 2 x 4 block around the old centre, then shift along +y until clear of mPFC
    oc = (ofc_old.max(0) + ofc_old.min(0)) / 2
    gx, gy = np.meshgrid((np.arange(4) - 1.5) * OFC_PITCH, (np.arange(2) - 0.5) * OFC_PITCH)
    ofc0 = oc + np.c_[gx.ravel(), gy.ravel()]
    if OFC_ML_START is not None:  # put the most medial OFC column at this bregma ML
        ref = mpfc[mpfc[:, 1].argmin()]
        ofc0[:, 0] += (OFC_ML_START - BREGMA_REF[1] + ref[0]) - ofc0[:, 0].min()
    shift = 0.0
    while min_pitch(np.r_[mpfc, ofc0 + [0, shift]]) < MIN_PITCH - 1e-9:
        shift += 0.005
    return mpfc, ofc0 + [0, shift], shift


def disc_grid(step=0.04):
    g = np.arange(-CAVITY_R, CAVITY_R + step, step)
    X, Y = np.meshgrid(g, g)
    P = np.c_[X.ravel(), Y.ravel()]
    return P[np.hypot(P[:, 0], P[:, 1]) < CAVITY_R]


def clearance(P, pillars):
    """Distance from each point to the nearest support (cavity wall or pillar edge)."""
    d = CAVITY_R - np.hypot(P[:, 0], P[:, 1])
    for p in pillars:
        d = np.minimum(d, np.linalg.norm(P - p, axis=1) - PILLAR_D / 2)
    return d


def pillar_positions(centres):
    """Greedy: put the next pillar where the top plate is least supported, until the span is small."""
    P = disc_grid()
    r_max = CAVITY_R - PILLAR_WALL_GAP - PILLAR_D / 2
    min_hole_dist = NEW_HOLE_D / 2 + PILLAR_CLEARANCE + PILLAR_D / 2
    ok = (np.hypot(P[:, 0], P[:, 1]) <= r_max) & (
        np.min([np.linalg.norm(P - c, axis=1) for c in centres], axis=0) >= min_hole_dist
    )
    cand = P[ok]
    # best achievable clearance at each point, given where pillars are allowed (the hole
    # cluster itself can only be supported from outside its keep-out zone)
    floor = CAVITY_R - np.hypot(P[:, 0], P[:, 1])
    for i in range(0, len(P), 2000):
        dc = np.sqrt(((P[i : i + 2000, None] - cand[None]) ** 2).sum(-1)).min(1) - PILLAR_D / 2
        floor[i : i + 2000] = np.minimum(floor[i : i + 2000], np.maximum(dc, 0))
    goal = np.maximum(TARGET_SPAN / 2, floor + 0.05)
    pillars = []
    while True:
        excess = clearance(P, pillars) - goal
        if excess.max() <= 0:
            break
        worst = P[excess.argmax()]
        free = cand
        if pillars:  # keep pillars separate (no merged chains)
            gap = np.min([np.linalg.norm(cand - p, axis=1) for p in pillars], axis=0)
            free = cand[gap >= PILLAR_D + PILLAR_GAP]
        if not len(free):
            break
        new = free[np.linalg.norm(free - worst, axis=1).argmin()]
        if clearance(worst[None], pillars + [new])[0] >= clearance(worst[None], pillars)[0] - 1e-9:
            goal[np.linalg.norm(P - worst, axis=1) < 0.05] = np.inf  # can't improve here; skip it
            continue
        pillars.append(new)
    return np.array(pillars)


def sort_holes(pts):
    """Posterior -> anterior, then medial -> lateral (the .holes.csv numbering)."""
    return pts[np.lexsort((pts[:, 0], pts[:, 1]))]


def seg_dist(a, b, c, d):
    """Min distance between 2D segments ab and cd (non-intersecting)."""

    def pt(p, q, r):
        t = np.clip(np.dot(p - q, r - q) / max(np.dot(r - q, r - q), 1e-12), 0, 1)
        return np.linalg.norm(p - (q + t * (r - q)))

    return min(pt(a, c, d), pt(b, c, d), pt(c, a, b), pt(d, a, b))


def pills(pairs, d, z0, z1, segments=96):
    out = None
    for a, b in pairs:
        ends = [m3d.Manifold.cylinder(z1 - z0, d / 2, d / 2, segments).translate([x, y, z0]) for x, y in (a, b)]
        c = m3d.Manifold.batch_hull(ends)
        out = c if out is None else out + c
    return out


def cylinders(centres, d, z0, z1, segments=96):
    out = None
    for x, y in centres:
        c = m3d.Manifold.cylinder(z1 - z0, d / 2, d / 2, segments).translate([x, y, z0])
        out = c if out is None else out + c
    return out


if __name__ == "__main__":
    V, T = read_stl(SRC)
    old = find_hole_centres(V, T)
    assert len(old) == 16, f"expected 16 holes, found {len(old)}"
    part = m3d.Manifold(m3d.Mesh(vert_properties=V.astype(np.float32), tri_verts=T.astype(np.uint32)))
    assert part.status() == m3d.Error.NoError, part.status()

    mpfc, ofc, shift = new_layout(old)
    mpfc, ofc = sort_holes(mpfc), sort_holes(ofc)
    new = np.r_[mpfc, ofc]
    pairs = [(pts[i], pts[i + 1]) for pts in (mpfc, ofc) for i in range(0, 8, 2)]
    for z0, z1 in PLATES_Z:
        part = part + cylinders(old, OLD_HOLE_D + 0.01, z0, z1)  # fill old holes
        if PILLS:
            part = part - pills(pairs, NEW_HOLE_D, z0 - 0.05, z1 + 0.05)
        else:
            part = part - cylinders(new, NEW_HOLE_D, z0 - 0.05, z1 + 0.05)  # drill new ones
    # keep-out for pillars: hole centres, or points along each pill's axis
    keep = np.vstack([np.linspace(a, b, 11) for a, b in pairs]) if PILLS else new

    pillars = pillar_positions(keep)
    h = CAVITY_Z[1] - CAVITY_Z[0] + 2 * OVERLAP
    for p in pillars:
        cyl = m3d.Manifold.cylinder(h, PILLAR_D / 2, PILLAR_D / 2, 64).translate([p[0], p[1], CAVITY_Z[0] - OVERLAP])
        part = part + cyl
    assert part.status() == m3d.Error.NoError, part.status()

    def ext(p):
        return f"x {p[:, 0].min():+.3f}..{p[:, 0].max():+.3f}, y {p[:, 1].min():+.3f}..{p[:, 1].max():+.3f}"

    print(f"mPFC holes: {ext(mpfc)} (AP span {np.ptp(mpfc[:, 1]):.3f}), nearest {min_pitch(mpfc):.3f}")
    print(f"OFC holes:  {ext(ofc)} (ML span {np.ptp(ofc[:, 0]):.3f}), nearest {min_pitch(ofc):.3f}, shifted +y {shift:.3f}")
    if PILLS:
        walls = [seg_dist(*pairs[i], *pairs[j]) - NEW_HOLE_D for i in range(8) for j in range(i + 1, 8)]
        for (a, b), name in zip(pairs, [f"mPFC{i}+{i + 1}" for i in (1, 3, 5, 7)] + [f"OFC{i}+{i + 1}" for i in (1, 3, 5, 7)]):
            L = np.linalg.norm(b - a)
            print(f"  pill {name:10s} length {L + NEW_HOLE_D:.3f} mm, tube centre spacing {TUBE_OD:.2f}..{L + NEW_HOLE_D - TUBE_OD:.3f} mm (design {L:.3f})")
        print(f"all pills: min wall between pills {min(walls):.3f}")
    else:
        print(f"all holes: nearest centre distance {min_pitch(new):.3f}, min wall {min_pitch(new) - NEW_HOLE_D:.3f}")
    print(f"pillars: {len(pillars)} x {PILLAR_D} mm, nearest hole edge "
          f"{min(np.linalg.norm(keep - p, axis=1).min() for p in pillars) - NEW_HOLE_D / 2 - PILLAR_D / 2:.2f} mm")
    print(f"largest unsupported top-plate span: {2 * clearance(disc_grid(), pillars).max():.2f} mm")
    print(f"volume {part.volume():.3f} mm^3, genus {part.genus()}")
    ref = new[new[:, 1].argmin()]
    rows = []
    for region, pts in (("mPFC", mpfc), ("OFC", ofc)):
        for i, (x, y) in enumerate(pts, 1):  # already sorted posterior -> anterior, medial -> lateral
            ap, ml = BREGMA_REF[0] + (y - ref[1]), BREGMA_REF[1] + (x - ref[0])
            pill = f"{region}{i - (i + 1) % 2}+{i + i % 2}" if PILLS else ""
            rows.append(f"{region},{region}{i},{ap:.3f},{ml:.3f},{x:.4f},{y:.4f},{pill}")
    csv = OUT.with_suffix(".holes.csv")
    csv.write_text("region,label,AP_mm,ML_mm,stl_x,stl_y,pill\n" + "\n".join(rows) + "\n")
    for region, pts in (("mPFC", mpfc), ("OFC", ofc)):
        ap = BREGMA_REF[0] + pts[:, 1] - ref[1]
        ml = BREGMA_REF[1] + pts[:, 0] - ref[0]
        print(f"{region} bregma: AP {ap.min():+.2f}..{ap.max():+.2f}, ML {ml.min():+.2f}..{ml.max():+.2f}")
    print(f"wrote {csv}")

    mesh = part.to_mesh()
    write_stl(OUT, np.asarray(mesh.vert_properties)[:, :3].astype(float), np.asarray(mesh.tri_verts))
    print(f"wrote {OUT}")
