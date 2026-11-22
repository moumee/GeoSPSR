"""Normal estimation with WNNC (Winding Number Normal Consistency).

Thin wrapper around ``third_party/WNNC/main_wnnc.py`` (Lin et al., SIGGRAPH Asia 2024)
so it can be called on in-memory arrays instead of files.

Requires the ``wnnc`` dependency group (``uv sync --group wnnc``). torch and
wn_treecode are imported lazily so ``import geospsr`` keeps working without them.
"""

from __future__ import annotations

import numpy as np

# (wsmin, wsmax) presets from main_wnnc.py
PRESET_WIDTHS = {
    "l0": (0.002, 0.016),  # uniform, noise-free points
    "l1": (0.01, 0.04),    # real scans
    "l2": (0.02, 0.08),    # noise sigma = 0.25%
    "l3": (0.03, 0.12),    # noise sigma = 0.5%
    "l4": (0.04, 0.16),    # noise sigma = 1%
    "l5": (0.05, 0.2),     # sparse points / 3D sketches
}


def estimate_normals_wnnc(
    points: np.ndarray,
    width_config: str | tuple[float, float] = "l0",
    iters: int = 40,
    dtype: str = "float",
    device: str | None = None,
    progress: bool = False,
) -> np.ndarray:
    """Estimate globally consistently oriented normals for a point cloud.

    Args:
        points: (N, 3) array of point positions (any scale; normalized internally).
        width_config: preset name ``"l0"``..``"l5"`` or a custom ``(wsmin, wsmax)`` tuple.
            Use larger levels for noisier / sparser inputs.
        iters: number of WNNC iterations (>= 2).
        dtype: ``"float"`` or ``"double"``.
        device: ``"cuda"`` or ``"cpu"``; ``None`` picks cuda when available.
        progress: show a tqdm progress bar.

    Returns:
        (N, 3) float64 array of unit normals, in the same order as ``points``.
    """
    import torch
    import torch.nn.functional as F
    import wn_treecode

    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    points = points[:, :3]
    if iters < 2:
        raise ValueError("iters must be >= 2")

    if isinstance(width_config, str):
        if width_config not in PRESET_WIDTHS:
            raise ValueError(f"unknown width_config {width_config!r}; choose from {list(PRESET_WIDTHS)} or pass (wsmin, wsmax)")
        wsmin, wsmax = PRESET_WIDTHS[width_config]
    else:
        wsmin, wsmax = width_config
    if wsmin > wsmax:
        raise ValueError(f"wsmin ({wsmin}) must be <= wsmax ({wsmax})")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = {"float": torch.float32, "double": torch.float64}[dtype]

    # Normalize into the box used by the presets (bbox -> [-1, 1] with 1.1 margin)
    bbox_min, bbox_max = points.min(0), points.max(0)
    center = (bbox_min + bbox_max) / 2.0
    scale = 2.0 / ((bbox_max - bbox_min).max() * 1.1)
    pts = torch.from_numpy((points - center) * scale).to(device=device, dtype=torch_dtype).contiguous()

    n = pts.shape[0]
    normals = torch.zeros_like(pts)
    b = torch.full((n, 1), 0.5, device=device, dtype=torch_dtype)
    widths = torch.ones(n, device=device, dtype=torch_dtype)

    wn_func = wn_treecode.WindingNumberTreecode(pts)

    steps = range(iters)
    if progress:
        from tqdm.auto import tqdm
        steps = tqdm(steps, desc="WNNC")

    with torch.no_grad():
        for i in steps:
            w = widths * (wsmin + ((iters - 1 - i) / (iters - 1)) * (wsmax - wsmin))

            # gradient step on || A mu - b ||^2
            A_mu = wn_func.forward_A(normals, w)
            r = wn_func.forward_AT(b, w) - wn_func.forward_AT(A_mu, w)
            A_r = wn_func.forward_A(r, w)
            alpha = (r * r).sum() / (A_r * A_r).sum()
            normals = normals + alpha * r

            # WNNC step: replace directions by the winding-number gradient, keep lengths
            out_normals = F.normalize(wn_func.forward_G(normals, w), dim=-1).contiguous()
            normals = out_normals * torch.linalg.norm(normals, dim=-1, keepdim=True)

    return out_normals.cpu().numpy().astype(np.float64)
