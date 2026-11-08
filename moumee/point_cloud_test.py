# %%
from functools import partial
import numpy as np
import jax.numpy as jnp
from jax import vmap

from geospsr.kernels import Matern32Kernel, ProductKernel
from geospsr.utils import trunc_Zd, periodic_stationary_interpolator
from geospsr.gp import poisson_cross_covariances, compute_mean

from skimage.measure import marching_cubes
from scipy.spatial import KDTree
import numpy as np
import open3d as o3d
from pyprojroot import here
from pathlib import Path
from huggingface_hub import snapshot_download

# %%
DATA_DIR = here("data/ppsurf")

if not (DATA_DIR / "famous_noisefree").exists():
    snapshot_download(
        "perler/ppsurf",
        repo_type="dataset",
        allow_patterns=["famous_*/*"],
        local_dir=DATA_DIR
    )

# %%
dragon_path = DATA_DIR / "famous_extra_noisy" / "04_pts_vis" / "dragon.xyz.ply"
pcd = o3d.io.read_point_cloud(str(dragon_path))
p = np.asarray(pcd.points)
# Use KD-tree to get nearest neighbor points
k = 50
tree = KDTree(p)
dist, idx = tree.query(p, k=k, workers=-1)

neighbors = p[idx]
centered = neighbors - neighbors.mean(axis=1, keepdims=True)

# Get the normals using eigen value decomposition (for educational purposes)
cov = np.einsum("nki,nkj->nij", centered, centered) / k
w, V = np.linalg.eigh(cov)
normals = V[:, :, 0]
pcd.normals = o3d.utility.Vector3dVector(normals)

# Open 3D version of estimating normals
pcd_o3d = o3d.io.read_point_cloud(str(dragon_path))
pcd_o3d.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=50))
normals_o3d = np.array(pcd_o3d.normals)
pcd_o3d.orient_normals_consistent_tangent_plane(k=15)

# %%
o3d.visualization.draw_geometries([pcd],
                                  window_name="Dragon - PCA normals",
                                  point_show_normal=True)

o3d.visualization.draw_geometries([pcd_o3d],
                                  window_name="Dragon - Open3D normals (oriented)",
                                  point_show_normal=True)

# %%
