"""This module provides the functions needed to compute a limb darkened light curve as
described by `Agol et al. (2020) <https://arxiv.org/abs/1908.03222>`_.
"""

__all__ = ["light_curve"]

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import binom, roots_legendre

from jaxoplanet.types import Array
from jaxoplanet.utils import get_dtype_eps, zero_safe_sqrt


@partial(jax.jit, static_argnames=("order",))
def light_curve(u: Array, b: Array, r: Array, *, order: int = 60):
    """Compute the light curve for arbitrary polynomial limb darkening

    See `Agol et al. (2020) <https://arxiv.org/abs/1908.03222>`_ for more technical
    details. The returned quantity is the fractional flux deficit (the flux minus
    one), computed directly from deficit-sized quantities so that it is accurate
    relative to the transit depth rather than the out-of-transit flux; this matters
    most in single precision. The zeroth and quadratic terms are evaluated in
    closed form, and the others by Gauss-Legendre quadrature after a change of
    variables that keeps the integrands smooth in the grazing regime.

    Args:
        u (Array): The coefficients of the polynomial limb darkening model
        b (Array): The center-to-center distance between the occultor and the occulted
            body
        r (Array): The radius ratio between the occultor and the occulted body
        order (int): The quadrature order used for the linear term and the terms of
            degree 3 and higher
    """

    u = jnp.asarray(u)
    assert u.ndim == 1
    if u.shape[0] == 0:
        g = jnp.full((1,), 1.0 / jnp.pi)
    else:
        g = greens_basis_transform(u)
        g /= jnp.pi * (g[0] + g[1] / 1.5)
    ds = deficit_vector(len(g) - 1, order=order)(b, r)
    return ds @ g


def deficit_vector(l_max: int, order: int = 60) -> Callable[[Array, Array], Array]:
    """The solution vector minus its no-occultation limit, ``(pi, 2 pi / 3, 0, ...)``,
    with every term built from deficit-sized quantities (no large-value
    cancellations); see ``light_curve``
    """
    n_max = l_max + 1

    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        eps = get_dtype_eps(b)

        full_occ = 1 + b <= r
        no_occ = jnp.logical_or(b >= 1 + r, r <= eps)
        occ = jnp.logical_not(jnp.logical_or(full_occ, no_occ))
        b_ = jnp.where(occ, b, 0.5)
        r_ = jnp.where(occ, r, 0.25)

        b2 = jnp.square(b_)
        r2 = jnp.square(r_)
        area, kap0, kap1 = kappas(b_, r_)
        intersect = jnp.logical_and(b_ > jnp.abs(1 - r_), b_ < 1 + r_)
        big = jnp.logical_or(jnp.logical_not(intersect), b_ <= eps)

        # Closed forms for s0 - pi and s2, rearranged so that no term is larger
        # than the deficit itself
        ds0 = jnp.where(big, -jnp.pi * r2, -(kap1 + r2 * kap0 - 0.5 * area))
        ds2 = jnp.where(
            big,
            2 * jnp.pi * r2 * (r2 + 2 * b2 - 1),
            2 * kap0 * r2 * (r2 + 2 * b2 - 1) + 0.5 * area * (1 - 5 * r2 - b2),
        )

        ds = [ds0[None]]
        if l_max >= 1:
            P = p_integral(order, l_max, b_, r_, include_s1=True)

            # s1 - 2 pi / 3 = -P1 - 2 kappa_1 / 3, with kappa_1 computed from
            # the same primitives as the quadrature (2 b sin(kappa_1) =
            # sqrt(-onembpr2 * onembmr2)) so that the noise floor of the sum
            # stays below the size of its terms
            onembmr2 = (b_ + (1 - r_)) * ((1 + r_) - b_)
            onembpr2 = ((1 - r_) - b_) * ((1 + r_) + b_)
            kap1_c = jnp.arctan2(
                zero_safe_sqrt(jnp.maximum(-onembpr2 * onembmr2, 0.0)),
                b2 + (1 - r_) * (1 + r_),
            )
            kap1_c = jnp.where(onembpr2 < 0, kap1_c, 0.0)
            ds1 = -P[0] - 2 * kap1_c / 3
            ds.append(ds1[None])
        if l_max >= 2:
            ds.append(ds2[None])
        if l_max >= 3:
            ds.append(-P[1:])
        out = jnp.concatenate(ds, axis=0)

        free = np.zeros(n_max)
        free[0] = -np.pi
        if l_max >= 1:
            free[1] = -2 * np.pi / 3
        return jnp.where(no_occ, 0.0, jnp.where(full_occ, jnp.asarray(free), out))

    return impl


def solution_vector(l_max: int, order: int = 60) -> Callable[[Array, Array], Array]:
    """The limb darkening solution vector, computed as the deficit vector plus
    its no-occultation limit"""
    free = np.zeros(l_max + 1)
    free[0] = np.pi
    if l_max >= 1:
        free[1] = 2 * np.pi / 3
    deficit = deficit_vector(l_max, order=order)

    def impl(b: Array, r: Array) -> Array:
        return deficit(b, r) + free

    return impl


def greens_basis_transform(u: Array) -> Array:
    dtype = jnp.dtype(u)
    u = jnp.concatenate((-jnp.ones(1, dtype=dtype), u))
    size = len(u)
    i = np.arange(size)
    arg = binom(i[None, :], i[:, None]) @ u
    p = (-1) ** (i + 1) * arg
    g = [jnp.zeros((), dtype=dtype) for _ in range(size + 2)]
    for n in range(size - 1, 1, -1):
        g[n] = p[n] / (n + 2) + g[n + 2]
    g[1] = p[1] + 3 * g[3]
    g[0] = p[0] + 2 * g[2]
    return jnp.stack(g[:-2])


def kappas(b: Array, r: Array) -> tuple[Array, Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(cond, b, jnp.ones_like(b))
    area: Array = jnp.where(cond, kite_area(r, b_, jnp.ones_like(r)), jnp.zeros_like(r))
    return area, jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def p_integral(
    order: int, l_max: int, b: Array, r: Array, *, include_s1: bool = False
) -> Array:
    """The quadrature terms of the (deficit) solution vector, from Equation (47)
    of Agol et al. (2020)

    In the grazing regime (k^2 < 1) the integrands have (k^2 - sin^2 x)^(n/2)
    endpoint singularities, so we substitute sin(x) = k sin(theta), which maps
    the integrals onto a fixed range with smooth integrands; when the occultor
    is fully inside the disk (k^2 >= 1) the original angle variable is already
    smooth. When ``include_s1`` is true, the first row is the linear (n = 1)
    term, used by the flux deficit computation; the remaining rows are the terms
    of degree 3 and higher.
    """
    return _p_integral(order, l_max, include_s1, b, r)


@partial(jax.custom_jvp, nondiff_argnums=(0, 1, 2))
def _p_integral(order: int, l_max: int, include_s1: bool, b: Array, r: Array) -> Array:
    P, _, _ = _p_integral_impl(order, l_max, include_s1, b, r)
    return P


@_p_integral.defjvp
def _p_integral_jvp(order, l_max, include_s1, primals, tangents):
    b, r = primals
    bt, rt = tangents
    P, dPdb, dPdr = _p_integral_impl(order, l_max, include_s1, b, r)
    return P, dPdb * bt + dPdr * rt


def _p_integral_impl(
    order: int, l_max: int, include_s1: bool, b: Array, r: Array
) -> tuple[Array, Array, Array]:
    """The quadrature rows and their partials with respect to b and r, computed
    by differentiating the integrands analytically on the same quadrature nodes
    (much cheaper than reverse-mode autodiff through the node sums); under JIT
    the partials are eliminated as dead code when only the values are used
    """
    assert include_s1 or l_max >= 3
    tiny = jnp.finfo(jnp.result_type(b)).tiny

    theta, weights = quad_nodes(order)
    st2 = np.square(np.sin(theta))
    ct = np.cos(theta)
    ct2 = np.square(ct)

    bmr = b - r
    fourbr = 4 * b * r
    # Groupings of (1 - (b -/+ r)^2) that avoid catastrophic cancellation near
    # the contact points, where 1 - r is exact (Sterbenz)
    onembmr2 = (b + (1 - r)) * ((1 + r) - b)
    onembpr2 = ((1 - r) - b) * ((1 + r) + b)
    domr2_db = -2 * bmr
    domr2_dr = 2 * bmr

    # k^2 >= 1 if and only if b + r <= 1; when b * r underflows we route those
    # points through the "inside" branch, which never needs k^2
    degenerate = fourbr < jnp.sqrt(tiny)
    inside = jnp.logical_or(onembpr2 >= 0, degenerate)
    fourbr_safe = jnp.where(degenerate, 1.0, fourbr)
    k2 = jnp.where(degenerate, 2.0, onembmr2 / fourbr_safe)
    k2c = jnp.clip(k2, 0.0, 1.0)
    k = zero_safe_sqrt(k2c)

    # The k^2 tangents only matter on the outside branch; the k^2 = 0 gate keeps
    # the 1 / k factor in the Jacobian derivative finite
    k2_ok = jnp.logical_and(jnp.logical_not(inside), k2c > 0.0)
    dk2_db = jnp.where(k2_ok, (domr2_db - 4 * r * k2c) / fourbr_safe, 0.0)
    dk2_dr = jnp.where(k2_ok, (domr2_dr - 4 * b * k2c) / fourbr_safe, 0.0)

    raw = jnp.where(inside, onembmr2 - fourbr * st2, onembmr2 * ct2)
    base = jnp.maximum(raw, 0.0)
    pos = raw > 0
    dbase_db = jnp.where(
        pos, jnp.where(inside, domr2_db - 4 * r * st2, domr2_db * ct2), 0.0
    )
    dbase_dr = jnp.where(
        pos, jnp.where(inside, domr2_dr - 4 * b * st2, domr2_dr * ct2), 0.0
    )

    s2x = jnp.where(inside, st2, k2c * st2)
    ds2x_db = dk2_db * st2
    ds2x_dr = dk2_dr * st2

    omk2st2 = 1 - k2c * st2
    jac = jnp.where(inside, 1.0, k * ct / jnp.sqrt(omk2st2))
    k_safe = jnp.where(k2_ok, k, 1.0)
    djac_fac = jnp.where(k2_ok, ct / (2 * k_safe * omk2st2 * jnp.sqrt(omk2st2)), 0.0)
    djac_db = dk2_db * djac_fac
    djac_dr = dk2_dr * djac_fac

    A = 2 * r * (r - b + 2 * b * s2x)
    dA_db = 2 * r * (2 * s2x - 1 + 2 * b * ds2x_db)
    dA_dr = 2 * (2 * r - b + 2 * b * s2x) + 4 * b * r * ds2x_dr

    AJ = A * jac
    dAJ_db = dA_db * jac + A * djac_db
    dAJ_dr = dA_dr * jac + A * djac_dr

    rows, drows_db, drows_dr = [], [], []
    if include_s1:
        z = zero_safe_sqrt(base)
        F = (1 + base / (1 + z)) / 3
        dF_dbase = (1 + 0.5 * z) / (3 * jnp.square(1 + z))
        rows.append((AJ * F)[None, :])
        drows_db.append((dAJ_db * F + AJ * dF_dbase * dbase_db)[None, :])
        drows_dr.append((dAJ_dr * F + AJ * dF_dbase * dbase_dr)[None, :])
    if l_max >= 3:
        n = jnp.arange(3, l_max + 1)
        pw = base[None, :] ** (0.5 * n[:, None])
        dpw = 0.5 * n[:, None] * base[None, :] ** (0.5 * n[:, None] - 1)
        rows.append(pw * AJ[None, :])
        drows_db.append(dAJ_db[None, :] * pw + AJ[None, :] * dpw * dbase_db[None, :])
        drows_dr.append(dAJ_dr[None, :] * pw + AJ[None, :] * dpw * dbase_dr[None, :])

    P = jnp.concatenate(rows, axis=0) @ weights
    dPdb = jnp.concatenate(drows_db, axis=0) @ weights
    dPdr = jnp.concatenate(drows_dr, axis=0) @ weights
    return P, dPdb, dPdr


def quad_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Quadrature nodes over theta in (0, pi / 2) and their weights, including
    the factor of 2 for the symmetric other half of the range

    The Gauss-Legendre nodes are remapped twice through theta -> (pi / 2)
    sin(theta), which clusters them near the upper limit. The integrands are
    analytic in a neighborhood of the real axis except for a branch point at
    distance ~ sqrt(|k^2 - 1|) from theta = pi / 2, which throttles convergence
    near the contact points; each remap takes a distance d to ~ d^(1/4),
    restoring fast convergence uniformly in k^2. Since the nodes are constants,
    this costs nothing at runtime.
    """
    roots, weights = roots_legendre(order)
    theta = 0.25 * np.pi * (roots + 1)
    weights = 0.5 * np.pi * weights
    for _ in range(2):
        weights = weights * 0.5 * np.pi * np.cos(theta)
        theta = 0.5 * np.pi * np.sin(theta)
    return theta, weights


def kite_area(a: Array, b: Array, c: Array) -> Array:
    def sort2(a: Array, b: Array) -> tuple[Array, Array]:
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return zero_safe_sqrt(jnp.maximum(0, square_area))
