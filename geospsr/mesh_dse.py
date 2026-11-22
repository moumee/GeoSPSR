"""Mesh reconstruction with DSE meshing (Learning Delaunay Surface Elements).

Thin wrapper around ``third_party/dse-meshing`` (Rakotosaona et al., CVPR 2021) so it
can be called on in-memory arrays. DSE meshing needs TensorFlow 1.15 / Python 3.8, which
cannot live in the main venv, so ``dse_meshing_mesh`` runs this same file as a script in a
separate venv (``.venv-dse``) and reads the resulting mesh back. The upstream functions
are imported and reused as-is; only paths/globals are injected, and the patch alignment
loop is re-implemented because upstream hard-codes 10k points.

One-time setup (from the repo root)::

    git submodule update --init --recursive third_party/dse-meshing
    uv venv --python 3.8 .venv-dse
    uv pip install --no-config --python .venv-dse/bin/python \\
        --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match \\
        "nvidia-tensorflow==1.15.5+nv22.12" "numpy<1.24" scipy scikit-learn trimesh
    mkdir -p data/dse_meshing
    g++ -std=c++11 -O2 -I third_party/dse-meshing/triangle_selection/postprocess \\
        third_party/dse-meshing/triangle_selection/postprocess/postprocess.cpp \\
        -o data/dse_meshing/postprocess
    curl -L https://nuage.lix.polytechnique.fr/index.php/s/FTCyp5WHg7Z68EM/download \\
        | tar -xJ -C data/dse_meshing          # -> data/dse_meshing/pretrained_models
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DSE_DIR = ROOT / "third_party" / "dse-meshing"
DSE_PYTHON = ROOT / ".venv-dse" / "bin" / "python"
DATA = ROOT / "data" / "dse_meshing"
POSTPROCESS = DATA / "postprocess"
LOGMAP_MODEL = DATA / "pretrained_models" / "pretrained_logmap" / "model.ckpt"
CLASSIFIER_MODEL = DATA / "pretrained_models" / "pretrained_classifier" / "model.ckpt"

# upstream defaults (eval_network.py / align_patches.py / eval_align_meshes.py)
N_NEIGHBORS = 120
N_NEAREST_NEIGHBORS = 30
NAME = "shape"


def dse_meshing_mesh(points, verbose: bool = False):
    """Reconstruct a triangle mesh from an (N, 3) point cloud (no normals needed).

    Returns an ``open3d.geometry.TriangleMesh`` in the same coordinates as ``points``
    (vertices are exactly the input points). Upstream recommends a bbox diagonal > 1.
    """
    import numpy as np
    import open3d as o3d

    for p in [DSE_PYTHON, POSTPROCESS, LOGMAP_MODEL.with_suffix(".ckpt.index")]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; see setup in {__file__}")

    with tempfile.TemporaryDirectory(prefix="dse_") as work:
        np.savetxt(os.path.join(work, f"{NAME}.xyz"), np.asarray(points)[:, :3])
        # no __pycache__ inside the submodule (keeps it clean)
        env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2", "PYTHONDONTWRITEBYTECODE": "1"}
        # separate processes per stage, like upstream run.sh (TF state / fork safety)
        # single-threaded BLAS in the align pool (cpu_count workers); 20x faster
        single = {k: "1" for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]}
        for stage in ["logmap", "align", "select"]:
            r = subprocess.run([str(DSE_PYTHON), __file__, stage, work],
                               env={**env, **single} if stage == "align" else env,
                               capture_output=not verbose, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"DSE meshing stage {stage!r} failed:\n"
                                   f"{r.stdout[-2000:] if r.stdout else ''}\n"
                                   f"{r.stderr[-4000:] if r.stderr else ''}")
        mesh = o3d.io.read_triangle_mesh(os.path.join(work, f"final_mesh_{NAME}.ply"))
    return _orient(mesh)


def _orient(mesh, n_dirs=15, eps_frac=0.3, seed=0):
    """Upstream emits triangles with sorted (= random) winding, which breaks smooth shading.
    Orient each face outward by an orientation-free inside test on the mesh itself: ray
    parity votes (n_dirs random directions) at points offset to both sides of the face.
    Geometry is unchanged; only the vertex order per face (NC metric is unaffected)."""
    import numpy as np
    import open3d as o3d

    v, f = np.asarray(mesh.vertices), np.asarray(mesh.triangles).copy()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    tri = v[f]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    eps = eps_frac * np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1).mean()
    dirs = np.random.default_rng(seed).normal(size=(n_dirs, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    votes = np.zeros(len(f))  # > 0: +n side is outside
    for side in [1, -1]:
        origin = tri.mean(1) + side * eps * n
        for d in dirs:
            rays = np.hstack([origin, np.broadcast_to(d, origin.shape)]).astype(np.float32)
            inside = scene.count_intersections(o3d.core.Tensor(rays)).numpy() % 2 == 1
            votes += side * np.where(inside, -1, 1)
    flip = votes < 0
    f[flip] = f[flip][:, ::-1]
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Script part: runs inside .venv-dse (Python 3.8, TF 1.15)
# ---------------------------------------------------------------------------

def _stage_logmap(work):
    """1. logmap estimation (logmap_estimation/eval_network.py)"""
    sys.path.insert(0, str(DSE_DIR / "logmap_estimation"))
    import eval_network as en
    en.n_neighbors, en.n_nearest_neighbors = N_NEIGHBORS, N_NEAREST_NEIGHBORS
    en.reconstruct(f"{NAME}.xyz", str(CLASSIFIER_MODEL), str(LOGMAP_MODEL), work, work)


def _align_one(i):
    return _ap.align_patch(_maps, _nbrs, i)


def _stage_align(work):
    """2a. logmap patch alignment (logmap_alignment/align_patches.py, any #points)"""
    global _ap, _maps, _nbrs
    import multiprocessing
    import numpy as np
    sys.path.insert(0, str(DSE_DIR / "logmap_alignment"))
    import align_patches as _ap
    _ap.n_nearest_neighbors = N_NEAREST_NEIGHBORS
    _maps = np.load(os.path.join(work, f"predicted_map_{NAME}.npy")).astype(np.float64)
    _nbrs = np.load(os.path.join(work, f"predicted_neighborhood_indices_{NAME}.npy")).astype(np.int32)
    with multiprocessing.get_context("fork").Pool(os.cpu_count()) as pool:
        corrected = np.array(pool.map(_align_one, range(len(_maps)), chunksize=64))
    np.save(os.path.join(work, f"corrected_maps_{NAME}.npy"), corrected)


def _stage_select(work):
    """2b. triangle frequency (eval_align_meshes.py) + 3. triangle selection (select.py)"""
    import importlib.util
    import trimesh
    sys.path.insert(0, str(DSE_DIR / "logmap_alignment"))
    import eval_align_meshes as em
    # upstream select.py clashes with stdlib `select` (already in sys.modules) -> load by path
    spec = importlib.util.spec_from_file_location(
        "dse_select", DSE_DIR / "triangle_selection" / "select.py")
    sel = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sel)
    em.in_path = em.raw_prediction_path = em.res_path = work
    em.n_neighbors, em.n_nearest_neighbors = N_NEIGHBORS, N_NEAREST_NEIGHBORS
    em.reconstruct(f"{NAME}.xyz")

    sel.in_path = sel.align_path = sel.res_path = work
    bbox_diag = sel.write_candidates(NAME)
    out = os.path.join(work, f"final_mesh_{NAME}.ply")
    subprocess.run([str(POSTPROCESS), os.path.join(work, f"pred_{NAME}.txt"), out], check=True)
    mesh = trimesh.load(out, process=False)
    mesh.vertices *= bbox_diag / sel.BBOX
    mesh.export(out)


if __name__ == "__main__":
    # running as a script puts geospsr/ on sys.path; its utils.py would shadow upstream modules
    sys.path = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]
    stage, work_dir = sys.argv[1], sys.argv[2]
    {"logmap": _stage_logmap, "align": _stage_align, "select": _stage_select}[stage](work_dir)
