import jax
import jax.numpy as jnp
from jax import jit, vmap

from .sdd import sdd_solve
from .utils import bmap


def poisson_cross_covariances(x_diff, eigenvectors, eigenvalues):
    """
    Compute the cross-covariance between f and v of Poisson Surface Reconstruction
    given the Karhunen-Loeve expansion of the kernel on v

    x_diff: (d,)
    evecs: (n, d)
    evals: (n,)

    returns: (d,)
    """
    evec_norm_sq = (eigenvectors**2).sum(-1)
    evec_norm_sq = jnp.where(evec_norm_sq > 0, evec_norm_sq, 1)
    dot = jnp.dot(eigenvectors, x_diff)
    sin = jnp.sin(dot)
    return (eigenvectors.T * eigenvalues**2 * sin / evec_norm_sq).sum(-1)


def f_eigenvalues_from_v_eigenvalues(eigenvectors, eigenvalues):
    """
    Compute the Karhunen-Loeve expansion's eigenvalues of the kernel on f given the
    Mercer expansion of the kernel on v

    eigenvectors: (n, d)
    eigenvalues: (n,)

    returns: (n,)
    """
    factor = ((eigenvectors * eigenvalues[:, None]) ** 2).sum(-1) ** 0.5
    evec_norms = (eigenvectors**2).sum(-1)
    return jnp.where(evec_norms > 0, factor / evec_norms, 0)


def f_gamma_from_v_xi(xi, eigenvectors, eigenvalues):
    """
    Compute the random variables in the Karhunen-Loeve expansion of samples of f
    given the Karhunen-Loeve expansion of samples of v

    xi: (m, d, n, 2)
    eigenvectors: (n, d)
    eigenvalues: (n,)

    returns: (m, n, 2)
    """
    factor = (eigenvectors**2 * eigenvalues[:, None] ** 2).sum(-1) ** 0.5
    gamma = (
        eigenvectors.T[..., None]
        * eigenvalues[..., None]
        * jnp.flip(xi, -1)
        * jnp.array([-1, 1])
    ).sum(1) / jnp.where(factor > 0, factor, 1)[..., None]

    return gamma


def evaluate_karhunen_loeve_sample(x, eigenvectors, eigenvalues, xi):
    """
    Evaluate the Karhunen-Loeve expansion of samples

    x: (d,)
    eigenvectors: (n, d)
    eigenvalues: (n,)
    xi: (n, 2)

    returns: (,)
    """
    dot = jnp.dot(eigenvectors, x)

    cos_term = xi[:, 0] * jnp.cos(dot)
    sin_term = xi[:, 1] * jnp.sin(dot)

    return (eigenvalues * (cos_term + sin_term)).sum()


def evaluate_karhunen_loeve_kernel(x_diff, eigenvectors, eigenvalues):
    """
    Evaluate the Karhunen-Loeve expansion of a kernel

    x1: (d,)
    x2: (d,)
    eigenvectors: (n, d)
    eigenvalues: (n,)

    returns: (,)
    """
    dot = jnp.dot(eigenvectors, x_diff)
    return ((eigenvalues**2) * jnp.cos(dot)).sum()


def _noisy_kernel_cholesky(x_data, kernel_v, sigma):
    """
    Lower Cholesky factor L of K_vv(x_data, x_data) + sigma^2 I, where sigma is
    the observation noise std of v. Solve with cho_solve((L, True), rhs).

    x_data: (nd, d)

    returns: (nd, nd)

    [MODIFIED from upstream GeoSPSR] Upstream inlined this in compute_mean,
    compute_variance, compute_covariance and sample_pathwise_conditioning as
        kv_dd = (
            vmap(vmap(kernel_v, (None, 0)), (0, None))(x_data, x_data)
            + jnp.eye(x_data.shape[0]) * sigma
        )
        jax.scipy.linalg.solve(kv_dd, rhs, assume_a="pos")
    Three changes:
    1. sigma -> sigma**2: the SDD solver (sdd.py) adds sigma**2 * I, so the
       Cholesky and SDD paths were solving different systems (50x apart for
       sigma = 0.02). Unified to the GP convention sigma**2 (sigma = noise std).
    2. Wrapped in jit: built eagerly, every elementwise op of the kernel
       allocates its own (nd, nd) temporary (peak ~7x the matrix; OOM at
       nd = 24k on a 12 GB GPU). jit lets XLA fuse them into one kernel.
    3. Factorize once, inside the same jit, and keep only L: upstream kept
       kv_dd alive while solve() made its own copies (peak ~3x the matrix, OOM
       for nd = 24k when run after other jobs), and re-factorized kv_dd for
       every query batch in compute_variance / compute_covariance.
    """

    @jit
    def build(x):
        kv_dd = (
            vmap(vmap(kernel_v, (None, 0)), (0, None))(x, x)
            + jnp.eye(x.shape[0]) * sigma**2
        )
        # kv_dd is exactly symmetric (kernels depend on |x1 - x2|), so skip the
        # (kv_dd + kv_dd^T) / 2 that jnp.linalg.cholesky does: one fewer (nd, nd) buffer
        return jax.lax.linalg.cholesky(kv_dd, symmetrize_input=False)

    return build(x_data)


def compute_mean(
    x_query,
    x_data,
    v_data,
    kernel_v,
    kernel_fv,
    sigma,
    bs=2**15,
    sdd_params=None,
    use_jit=True,
    verbose=False,
):
    """
    Compute the mean of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)
    v_data: (nd, d)

    returns: (nq,)
    """
    if not sdd_params:
        # [MODIFIED] upstream: eager kernel matrix + sigma * I, then solve()
        # (see _noisy_kernel_cholesky)
        chol = _noisy_kernel_cholesky(x_data, kernel_v, sigma)
        alpha = jax.scipy.linalg.cho_solve((chol, True), v_data)
    else:
        alpha = sdd_solve(x_data, v_data, kernel_v, sigma, **sdd_params)

    def kfv_row(x):
        return vmap(kernel_fv, (None, 0))(x, x_data)

    def update(x):
        return (kfv_row(x) * alpha).sum()

    update = jit(update) if use_jit else update

    update_query = bmap(update, x_query, bs, verbose, "Computing mean")

    return update_query


def compute_variance(
    x_query,
    x_data,
    kernel_v,
    kernel_fv,
    var_f,
    sigma,
    bs=2**15,
    sdd_params=None,
    use_jit=False,
    verbose=False,
):
    """
    Compute the variance of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)

    returns: (nq, nq)
    """
    if not sdd_params:
        # [MODIFIED] upstream: eager kernel matrix + sigma * I, then solve()
        # (see _noisy_kernel_cholesky)
        chol = _noisy_kernel_cholesky(x_data, kernel_v, sigma)

        def solver(row):
            return jax.scipy.linalg.cho_solve((chol, True), row)
    else:

        def solver(row):
            return sdd_solve(x_data, row, kernel_v, sigma, **sdd_params)

    def kfv_row(x):
        return vmap(kernel_fv, (None, 0))(x, x_data)

    def var(x):
        row = kfv_row(x)
        return var_f - (row * solver(row)).sum()

    var = jit(var) if use_jit else var

    return bmap(var, x_query, bs, verbose, "Computing variance")


def compute_covariance(
    x_query,
    x_data,
    kernel_v,
    kernel_fv,
    kernel_f,
    sigma,
    bs_kfv=2**13,
    bs_solve=2**7,
    bs_kf=2**7,
    sdd_params=None,
    verbose=False,
):
    """
    Compute the covariance of f at x_query conditioned on the v observations of v_data
    at x_data

    x_query: (nq, d)
    x_data: (nd, d)

    returns: (nq, nq)
    """
    if not sdd_params:
        # [MODIFIED] upstream: eager kernel matrix + sigma * I, then solve()
        # (see _noisy_kernel_cholesky)
        chol = _noisy_kernel_cholesky(x_data, kernel_v, sigma)

        def solver(row):
            return jax.scipy.linalg.cho_solve((chol, True), row)
    else:

        def solver(row):
            return sdd_solve(x_data, row, kernel_v, sigma, **sdd_params)

    def kfv_row(x):
        return vmap(kernel_fv, (None, 0))(x, x_data)

    kfv = bmap(kfv_row, x_query, bs_kfv, verbose, "Computing cross covariance")
    kvv_inv_kvf = bmap(
        solver, kfv, bs_solve, verbose, "Solving cross covariance transpose"
    )
    update = jnp.einsum("ijk, ljk->il", kfv, kvv_inv_kvf)

    def prior_row(x):
        return vmap(kernel_f, (None, 0))(x, x_query)

    prior = bmap(prior_row, x_query, bs_kf, verbose, "Computing prior covariance")

    return prior + update


def sample_pathwise_conditioning(
    x_query,
    x_data,
    v_data,
    kernel_v,
    kernel_fv,
    sigma,
    xi,
    gamma,
    v_eigenvectors,
    v_eigenvalues,
    f_eigenvectors,
    f_eigenvalues,
    bs_v,
    bs_kfv,
    bs_f,
    sdd_params=None,
    verbose=False,
):
    """
    Compute samples of f at x_query conditioned on the v observations of v_data
    at x_data using pathwise conditioning

    x_query: (nq, d)
    x_data: (nd, d)
    xi: (ns, 3, nv, 2)
    gamma: (ns, nv, 2)
    v_eigenvectors: (nv, d)
    v_eigenvalues: (nv,)
    f_eigenvectors: (nf,)
    f_eigenvalues: (nf,)

    returns: (ns, nq)
    """

    def vs(x):
        return vmap(
            vmap(evaluate_karhunen_loeve_sample, (None, None, None, 0)),
            (None, None, None, 0),
        )(x, v_eigenvectors, v_eigenvalues, xi)

    v_prior = bmap(vs, x_data, bs_v, verbose, "Computing prior V samples").transpose(
        1, 0, 2
    )
    v_residual = v_data - v_prior

    if not sdd_params:
        # [MODIFIED] upstream: eager kernel matrix + sigma * I, then solve()
        # (see _noisy_kernel_cholesky)
        chol = _noisy_kernel_cholesky(x_data, kernel_v, sigma)

        def solver(row):
            return jax.scipy.linalg.cho_solve((chol, True), row)
    else:

        def solver(row):
            return sdd_solve(x_data, row, kernel_v, sigma, **sdd_params)

    alpha = vmap(solver)(v_residual)

    def kfv_row(x):
        return vmap(kernel_fv, (None, 0))(x, x_data)

    def update(x):
        return (kfv_row(x) * alpha).sum((-2, -1))

    f_update = bmap(update, x_query, bs_kfv, verbose, "Computing f update").transpose(
        1, 0
    )

    def fs(x):
        return vmap(evaluate_karhunen_loeve_sample, (None, None, None, 0))(
            x, f_eigenvectors, f_eigenvalues, gamma
        )

    f_prior = bmap(fs, x_query, bs_f, verbose, "Computing prior f samples").transpose(
        1, 0
    )

    return f_prior + f_update
