from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.starry.surface import Surface


# -------- Grid (equal-area-ish Fibonacci; tiny + deterministic) --------
def fibonacci_grid(oversample: int, lmax: int):
    n = oversample * (lmax**2)
    i = jnp.arange(n)
    phi = (1.0 + jnp.sqrt(5.0)) / 2.0
    theta = jnp.arccos(1.0 - 2.0 * (i + 0.5) / n)  # colatitude in [0, pi]
    lon = (2.0 * jnp.pi) * ((i / phi) % 1.0)  # [0, 2pi)
    lat = (jnp.pi / 2.0) - theta  # latitude in [-pi/2, pi/2]
    return jnp.stack([lat, lon], axis=-1)  # (n, 2)


# -------- Helpers to keep angles in-range (differentiable) -------------
def _wrap_lon(lon):
    # Wrap to [0, 2pi)
    two_pi = 2.0 * jnp.pi
    return lon - two_pi * jnp.floor(lon / two_pi)


def _clip_lat(lat, eps=1e-6):
    # Clip to (-pi/2, pi/2) by tiny epsilon to avoid singularities
    return jnp.clip(lat, -jnp.pi / 2 + eps, jnp.pi / 2 - eps)


def _project_coord(x):
    lat = _clip_lat(x[0])
    lon = _wrap_lon(x[1])
    return jnp.array([lat, lon])


# -------- Fixed-iteration damped Newton in 2D (lat, lon) --------------
@partial(
    jax.jit, static_argnames=("oversample", "lmax", "newton_iters", "damping", "step")
)
def surface_min_intensity(
    surface: Surface,
    oversample: int,
    lmax: int,
    newton_iters: int = 12,
    damping: float = 1e-3,
    step: float = 1.0,
):
    """
    Fully JAX, end-to-end differentiable approximate global min:
      1) seed from tiny equal-area grid
      2) run fixed M Newton steps from each seed in parallel
      3) take global min across seeds

    Args:
      surface: jaxoplanet Surface object with .intensity(lat, lon)
      oversample, lmax: define N = oversample * lmax^2 seeds
      newton_iters: fixed Newton iterations per seed (no line search)
      damping: Levenberg-Marquardt diagonal added to Hessian
      step: Newton step scaling (e.g., 1.0 or 0.5)

    Returns:
      (lat_min, lon_min), min_val
    """

    # Scalar objective taking a 2-vector x = [lat, lon]
    def f(x):
        lat, lon = x
        return surface.intensity(lat, lon)

    # Grad & Hessian w.r.t. x
    grad_f = jax.grad(f)
    hess_f = jax.hessian(f)

    def newton_one_seed(x0):
        def body(_, x):
            g = grad_f(x)  # (2,)
            H = hess_f(x)  # (2,2)
            H_damped = H + damping * jnp.eye(2, dtype=x.dtype)

            # Solve H p = g  (Newton step uses p = H^{-1} g)
            p = jnp.linalg.solve(H_damped, g)
            x_new = x - step * p

            # keep angles valid
            return _project_coord(x_new)

        xT = jax.lax.fori_loop(0, newton_iters, body, _project_coord(x0))
        return xT, f(xT)

    # Seeds
    seeds = fibonacci_grid(oversample, lmax)  # (N,2)

    # Run Newton from all seeds in parallel, no Python loops, fixed iteration count
    xs, vals = jax.vmap(newton_one_seed)(seeds)  # xs: (N,2), vals: (N,)

    # Take global min
    i_min = jnp.argmin(vals)
    x_min = xs[i_min]
    v_min = vals[i_min]

    return x_min, v_min
