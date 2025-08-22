import numpy as np
import pandas as pd

from protflow.utils.biopython_tools import load_structure_from_pdbfile


# ---------- Geometry helpers ----------

def fit_line_pca(points):
    """
    Fit a 3D line (axis) through a set of coordinates using PCA (SVD).
    Returns:
      c      = centroid (a point on the axis)
      v      = unit direction vector along the helix axis
      (s_min, s_max) = scalar range of projections of the points along v
      r_est  = RMS radial distance of points to the fitted axis (≈ helix radius)
    """
    P = np.asarray(points, dtype=float)        # ensure input is a NumPy float array of shape (N,3)
    c = P.mean(axis=0)                         # compute centroid of all points
    X = P - c                                  # center the data around the centroid
    U, S, Vt = np.linalg.svd(X, full_matrices=False)  
    # perform SVD; right-singular vectors in Vt are principal directions
    
    v = Vt[0]                                  # take the first principal component (direction of max variance)
    v = v / np.linalg.norm(v)                  # normalize to unit vector
    
    s_vals = X @ v                             # project all centered points onto axis → scalar coords
    s_min, s_max = s_vals.min(), s_vals.max()  # get min/max projected values (defines span along axis)
    
    perp = X - np.outer(s_vals, v)             # subtract projection → perpendicular offsets from axis
    r_est = np.sqrt((perp**2).sum(axis=1).mean())  
    # compute RMS of radial distances (average distance from axis)
    
    return c, v, (s_min, s_max), r_est         # return centroid, direction, span, and estimated radius

def fit_line_helix_centroids(points, window=4, trim_ends=0, outlier_z=2.5):
    """
    Helix-aware axis fit using sliding-window centroids (e.g., i..i+3 Cα).
    Returns:
      c : centroid on axis
      v : unit direction vector (N→C oriented)
      (s_min, s_max) : projection range of the ORIGINAL points along v
      r_est : RMS radial distance of ORIGINAL points to axis
    Args:
      points     : (N,3) array (ideally Cα atoms ordered from N to C)
      window     : sliding window length for centroids (4 is classic for α-helix)
      trim_ends  : optional number of residues to drop from each end before centroiding
      outlier_z  : remove centroid outliers farther than z * MAD from preliminary line
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < max(5, window+1):
        # fallback to your PCA if not enough points
        return fit_line_pca(P)

    # 0) optionally trim ragged ends before centroiding
    if trim_ends > 0:
        P_used = P[trim_ends: -trim_ends]
        if P_used.shape[0] < window+1:
            P_used = P  # not enough points to trim
    else:
        P_used = P

    # 1) sliding-window centroids (i..i+window-1)
    M = P_used.shape[0] - window + 1
    C = np.empty((M, 3), dtype=float)
    for i in range(M):
        C[i] = P_used[i:i+window].mean(axis=0)

    # 2) preliminary line from centroids via PCA
    Cc = C - C.mean(axis=0)
    _, _, Vt = np.linalg.svd(Cc, full_matrices=False)
    v0 = Vt[0]; v0 /= np.linalg.norm(v0)
    c0 = C.mean(axis=0)

    # 3) robust pass: remove centroid outliers (radial distance from preliminary line)
    #    distance to line through c0 with direction v0
    t = (C - c0) @ v0
    C_proj = c0 + np.outer(t, v0)
    resid = np.linalg.norm(C - C_proj, axis=1)
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) + 1e-9  # robust scale
    keep = resid <= (med + outlier_z * 1.4826 * mad)  # 1.4826 ≈ MAD→σ for Gauss

    if keep.sum() >= max(3, window):  # refit if we actually kept enough points
        Ck = C[keep]
        Ckc = Ck - Ck.mean(axis=0)
        _, _, Vt2 = np.linalg.svd(Ckc, full_matrices=False)
        v = Vt2[0]; v /= np.linalg.norm(v)
        c = Ck.mean(axis=0)
    else:
        v, c = v0, c0

    # 4) Orient axis N→C using original endpoints
    if ((P[-1] - P[0]) @ v) < 0:
        v = -v

    # 5) Project ORIGINAL points to get span and radius estimate
    X = P - c
    s_vals = X @ v
    s_min, s_max = float(s_vals.min()), float(s_vals.max())
    perp = X - np.outer(s_vals, v)
    r_est = float(np.sqrt((perp**2).sum(axis=1).mean()))

    return c, v, (s_min, s_max), r_est


def point_in_capped_cylinder(P, A, B, r):
    """
    Check whether points P (N,3) are inside the finite (capped) cylinder defined by:
      - axis endpoints A and B
      - radius r
    Returns a boolean mask of length N.
    """
    AB = B - A                                 # vector along cylinder axis
    L = np.linalg.norm(AB)                     # cylinder length
    
    if L == 0:                                 # degenerate case: A == B → cylinder is a sphere
        return np.linalg.norm(P - A, axis=1) <= r
    
    u = AB / L                                 # unit vector along axis
    AP = P - A                                 # vectors from base point A to all query points
    s = AP @ u                                 # projection lengths of AP onto axis direction
    s_clamped = np.clip(s, 0.0, L)             # clamp to [0,L] so projection is restricted to the finite segment
    
    radial2 = np.sum((AP - np.outer(s_clamped, u))**2, axis=1)  
    # squared perpendicular distance from each point to the axis segment
    
    return (radial2 <= r*r) & (s >= 0.0) & (s <= L)  
    # point is inside cylinder if: radial distance ≤ r AND projection lies between 0 and L


def segment_endpoints_from_axis(c, v, s0, s1):
    """
    Given axis c + s*v and a range [s0, s1], return endpoints of that segment in 3D.
    """
    return c + s0*v, c + s1*v

def bbox_for_segments(segments, pad):
    """
    Compute an axis-aligned bounding box (AABB) around a list of line segments.
    Padding 'pad' (e.g. radius + dx) ensures the cylinders are fully enclosed.
    Returns lo, hi (3D vectors).
    """
    pts = np.vstack([np.vstack([A, B]) for (A, B) in segments])
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    return lo, hi



# ---------- Main function ----------
# ----------------- Example -----------------
# Suppose pts1 and pts2 are Nx3 and Mx3 NumPy arrays of coordinates for each helix:
# res = extension_overlap_fraction(pts1, pts2, ext_A=5.0, radius1=2.3, radius2=2.3, dx=0.25)
# print(f"Overlap fraction: {res['overlap_fraction']:.4f}")
# print(f"Overlap volume: {res['overlap_volume_A3']:.2f} Å^3")
# print(f"Total extension volume: {res['total_extension_volume_A3']:.2f} Å^3")
# ---- Minimal helpers for surface XYZ export ----


# --- Simple one-call XYZ exporter: writes ONE XYZ block with all four cylinders ---
def create_cylinder_xyz(
    xyz_path,
    segs1, segs2,               # lists of (A,B) endpoints for helix1 and helix2 (2 each)
    radius1, radius2,           # radii (floats) for helix1 and helix2
    spacing=0.5,                # surface sampling density (Å)
    include_caps=True,          # include flat end caps
    element1="C", element2="N"  # element tags for helix1 and helix2
):
    # Small local helpers to keep things compact
    def _basis(u):
        a = np.array([1.0,0.0,0.0]) if abs(u[0]) < 0.9 else np.array([0.0,1.0,0.0])
        n1 = np.cross(u, a); n1 /= np.linalg.norm(n1)
        n2 = np.cross(u, n1); n2 /= np.linalg.norm(n2)
        return n1, n2

    def _surf(A, B, r, spacing, include_caps):
        AB = B - A
        L = np.linalg.norm(AB)
        if L == 0: return np.empty((0,3))
        u = AB / L
        n1, n2 = _basis(u)

        # side surface
        n_s = max(2, int(np.ceil(L / spacing)))
        n_theta = max(12, int(np.ceil(2*np.pi*r / spacing)))
        s_vals = np.linspace(0.0, L, n_s)
        thetas = np.linspace(0.0, 2*np.pi, n_theta, endpoint=False)
        side = [A + s*u + r*(np.cos(th)*n1 + np.sin(th)*n2) for s in s_vals for th in thetas]
        pts = np.array(side)

        if not include_caps: return pts

        # flat caps (concentric rings)
        n_r = max(2, int(np.ceil(r / spacing)))
        rs = np.linspace(0.0, r, n_r)
        for center in (A, B):
            for rr in rs:
                n_t = max(6, int(np.ceil((2*np.pi*max(rr, spacing*0.5)) / spacing)))
                ths = np.linspace(0.0, 2*np.pi, n_t, endpoint=False)
                ring = [center + rr*(np.cos(th)*n1 + np.sin(th)*n2) for th in ths]
                pts = np.vstack([pts, ring])
        return pts

    # Build all points + element tags (single XYZ block)
    all_pts = []
    elems = []

    for (A,B) in segs1:
        P = _surf(A, B, radius1, spacing, include_caps)
        all_pts.append(P); elems += [element1]*len(P)

    for (A,B) in segs2:
        P = _surf(A, B, radius2, spacing, include_caps)
        all_pts.append(P); elems += [element2]*len(P)

    if len(all_pts) == 0:
        # still write an empty valid XYZ
        with open(xyz_path, "w") as f:
            f.write("0\nempty\n")
        return 0

    P = np.vstack(all_pts)
    with open(xyz_path, "w") as f:
        f.write(f"{len(P)}\n")
        f.write("Helix extension cylinders (surface)\n")
        for (sym, (x,y,z)) in zip(elems, P):
            f.write(f"{sym} {x:.3f} {y:.3f} {z:.3f}\n")
    return len(P)


def extension_overlap_fraction(
    pts1, pts2,
    ext_1=5.0,
    ext_2=5.0,
    radius1=None, radius2=None,
    estimate_radius_from_points=False,
    dx=0.5,
    # --- simple XYZ surface export toggle ---
    xyz_surface_path=None,
    xyz_surface_spacing=0.5,
    xyz_surface_include_caps=True,
    xyz_elem1="C", xyz_elem2="N"
):
    """
    Returns a dict with overlap metrics + multiple terminal/endpoint distances.
    Assumes pts1/pts2 are ordered N -> C (as used by fit_line_helix_centroids).
    Requires: fit_line_helix_centroids, segment_endpoints_from_axis,
              point_in_capped_cylinder, bbox_for_segments, create_cylinder_xyz.
    """
    pts1 = np.asarray(pts1, dtype=float)
    pts2 = np.asarray(pts2, dtype=float)

    # 1) Fit axes (helix-aware; oriented N->C)
    c1, v1, (s1_min, s1_max), r1_est = fit_line_helix_centroids(pts1)
    c2, v2, (s2_min, s2_max), r2_est = fit_line_helix_centroids(pts2)

    # 2) Radii
    r1 = (r1_est if (estimate_radius_from_points or radius1 is None) else float(radius1))
    r2 = (r2_est if (estimate_radius_from_points or radius2 is None) else float(radius2))

    # 3) Single full-span segments with overhangs
    segs1 = [segment_endpoints_from_axis(c1, v1, s1_min - ext_1, s1_max + ext_1)]
    segs2 = [segment_endpoints_from_axis(c2, v2, s2_min - ext_2, s2_max + ext_2)]

    # -- Useful axis points (native and extended) for distances
    N1_axis = c1 + s1_min * v1
    C1_axis = c1 + s1_max * v1
    N2_axis = c2 + s2_min * v2
    C2_axis = c2 + s2_max * v2

    N1_ext_axis = c1 + (s1_min - ext_1) * v1
    C1_ext_axis = c1 + (s1_max + ext_1) * v1
    N2_ext_axis = c2 + (s2_min - ext_2) * v2
    C2_ext_axis = c2 + (s2_max + ext_2) * v2

    # 4) Analytic volumes of the two single cylinders
    h1 = (s1_max - s1_min) + (ext_1 * 2.0)
    h2 = (s2_max - s2_min) + (ext_2 * 2.0)
    V1 = np.pi * (r1**2) * h1
    V2 = np.pi * (r2**2) * h2
    V_total = V1 + V2

    # 5) Voxel overlap estimate
    lo, hi = bbox_for_segments(segs1 + segs2, pad=max(r1, r2) + dx)
    xs = np.arange(lo[0], hi[0] + dx/2, dx)
    ys = np.arange(lo[1], hi[1] + dx/2, dx)
    zs = np.arange(lo[2], hi[2] + dx/2, dx)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='xy')
    P = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    in_1 = point_in_capped_cylinder(P, segs1[0][0], segs1[0][1], r1)
    in_2 = point_in_capped_cylinder(P, segs2[0][0], segs2[0][1], r2)
    V_overlap = (in_1 & in_2).sum() * (dx**3)
    frac_sum = 0.0 if V_total == 0 else float(V_overlap / V_total)


    # 6) Surface XYZ export (single cylinders per helix)
    if xyz_surface_path is not None:
        create_cylinder_xyz(
            xyz_surface_path,
            segs1, segs2,
            r1, r2,
            spacing=xyz_surface_spacing,
            include_caps=xyz_surface_include_caps,
            element1=xyz_elem1, element2=xyz_elem2
        )

    # 7) Distances
    #   a) atomistic termini (first/last coordinates provided)
    d_N_C_terms_atoms = float(np.linalg.norm(pts1[0] - pts2[-1]))
    d_C_N_terms_atoms = float(np.linalg.norm(pts1[-1] - pts2[0]))

    #   b) axis points at native termini
    d_N_C_axis = float(np.linalg.norm(N1_axis - C2_axis))
    d_C_N_axis = float(np.linalg.norm(C1_axis - N2_axis))

    #   c) axis points at extended cylinder endpoints
    d_N_C_ext_axis = float(np.linalg.norm(N1_ext_axis - C2_ext_axis))
    d_C_N_ext_axis = float(np.linalg.norm(C1_ext_axis - N2_ext_axis))

    return {
        # overlap metrics
        "overlap": frac_sum,         # V_intersection / (V1 + V2)

        # distances: termini atoms (as given)
        "d_N_C_terms": d_N_C_terms_atoms,
        "d_C_N_terms": d_C_N_terms_atoms,

        # distances: axis points at native termini
        "d_N_C_axis": d_N_C_axis,
        "d_C_N_axis": d_C_N_axis,

        # distances: axis points at extended cylinder endpoints
        "d_N_C_ext_axis": d_N_C_ext_axis,
        "d_C_N_ext_axis": d_C_N_ext_axis
    }

def canonical_id(pose1, model1, pose2, model2):
    a = (str(pose1).strip(), int(model1))
    b = (str(pose2).strip(), int(model2))
    lo, hi = (a, b) if a <= b else (b, a)   # lexicographic order
    return (lo[0], lo[1], hi[0], hi[1])

def main(args):

    df1 = pd.read_pickle(args.frags1)
    df2 = pd.read_pickle(args.frags2)

    pose1_path = df1["poses"].iloc[0]
    pose2_path = df2["poses"].iloc[0]

    pose1 = load_structure_from_pdbfile(pose1_path, all_models=True)
    pose2 = load_structure_from_pdbfile(pose2_path, all_models=True)

    results = []
    for i, row1 in df1.iterrows():
        ent1 = pose1[row1['model_num']][row1['chain_id']]
        ent1_coords = [atom.coord for atom in ent1.get_atoms() if atom.id == "CA"]
        for j, row2 in df2.iterrows():
            ent2 = pose2[row2['model_num']][row2['chain_id']]
            ent2_coords = [atom.coord for atom in ent2.get_atoms() if atom.id == "CA"]
            data = extension_overlap_fraction(ent1_coords, ent2_coords, args.cylinder_height, args.cylinder_height, radius1=args.radius, radius2=args.radius, xyz_surface_path=None)
            data.update({
                'pose1_path': row1['poses'],
                'pose2_path': row2['poses'],
                'model1': row1['model_num'],
                'model2': row2['model_num'],
                'id': canonical_id(row1['poses'], row1['model_num'], row2['poses'], row2['model_num']),
            })
            results.append(data)

    df_out = pd.DataFrame(results)    
    df_out.to_pickle(args.out)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--frags1", type=str, required=True, help="First input JSON file")
    parser.add_argument("--frags2", type=str, required=True, help="Second input JSON file")
    parser.add_argument("--cylinder_height", type=float, required=True, help="Output directory")
    parser.add_argument("--radius", type=str, required=True, help="Prefix for output file")
    parser.add_argument("--out", type=str, default=1.5, help="VDW multiplier for backbone-backbone clashes")

    args = parser.parse_args()
    
    main(args)
