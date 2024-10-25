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
from jaxoplanet.utils import zero_safe_sqrt


@partial(jax.jit, static_argnames=("order",))
def light_curve(u: Array, b: Array, r: Array, *, order: int = 10):
    """Compute the light curve for arbitrary polynomial limb darkening

    See `Agol et al. (2020) <https://arxiv.org/abs/1908.03222>`_ for more technical
    details. Unlike in that paper, here we don't evaluate all the solution vector
    integrals in closed form. Instead, for all but the lowest order terms, we numerically
    integrate the relevant 1D line integral using Gauss-Legendre quadrature at fixed
    order ``order``.


    Args:
        u (Array): The coefficients of the polynomial limb darkening model
        b (Array): The center-to-center distance between the occultor and the occulted
            body
        r (Array): The radius ratio between the occultor and the occulted body
        order (int): The quadrature order to use when numerically computing the the 1D
            line integral
    """

    u = jnp.asarray(u)
    assert u.ndim == 1
    if u.shape[0] == 0:
        g = jnp.full((1,), 1.0 / jnp.pi)
    else:
        g = greens_basis_transform(u)
        g /= jnp.pi * (g[0] + g[1] / 1.5)
    s = solution_vector(len(g) - 1, order=order)(b, r)
    return s @ g - 1


def solution_vector(l_max: int, order: int = 10) -> Callable[[Array, Array], Array]:
    n_max = l_max + 1

    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        # TODO: Are all of these conditions necessary?
        b = jnp.abs(b)
        r = jnp.abs(r)
        area, kappa0, kappa1 = kappas(b, r)

        no_occ = jnp.greater_equal(b, 1 + r)
        full_occ = jnp.less_equal(1 + b, r)
        cond = jnp.logical_or(no_occ, full_occ)
        b_: Array = jnp.where(cond, jnp.ones_like(b), b)

        b2 = jnp.square(b_)
        r2 = jnp.square(r)

        s0, s2 = s0s2(b_, r, b2, r2, area, kappa0, kappa1)
        s0: Array = jnp.where(no_occ, jnp.pi, s0)
        s0 = jnp.where(full_occ, jnp.zeros_like(s0), s0)
        s2 = jnp.where(cond, jnp.zeros_like(s2), s2)

        s = [s0[None]]
        if l_max >= 1:
            P: Array = p_integral(order, l_max, b_, r, b2, r2, kappa0)
            P = jnp.where(cond, jnp.zeros_like(P), P)
            s.append(-P[:1] - 2 * (kappa1 - jnp.pi) / 3)
            if l_max >= 2:
                s.append(s2[None])  # type: ignore
            if l_max >= 3:
                s.append(-P[1:])
        return jnp.concatenate(s, axis=0)

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


def s0s2(
    b: Array,
    r: Array,
    b2: Array,
    r2: Array,
    area: Array,
    kappa0: Array,
    kappa1: Array,
) -> tuple[Array, Array]:
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    # Large k
    s0_lrg = jnp.pi * (1 - r2)
    s2_lrg = 2 * s0_lrg + 4 * jnp.pi * (eta2 - 0.5)

    # Small k
    Alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = jnp.pi - Alens
    s2_sml = 2 * s0_sml + 2 * (
        -(jnp.pi - kappa1) + 2 * eta2 * kappa0 - 0.25 * area * (1 + 5 * r2 + b2)
    )

    delta = 4 * b * r
    cond = jnp.greater(onembpr2 + delta, delta)
    return jnp.where(cond, s0_lrg, s0_sml), jnp.where(cond, s2_lrg, s2_sml)


def p_integral(
    order: int,
    l_max: int,
    b: Array,
    r: Array,
    b2: Array,
    r2: Array,
    kappa0: Array,
) -> Array:
    # This is a hack for when r -> 0 or b -> 0, so k2 -> inf
    factor = 4 * b * r
    k2_cond = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
    factor: Array = jnp.where(k2_cond, jnp.ones_like(factor), factor)
    k2 = jnp.maximum(jnp.zeros_like(factor), (1 - r2 - b2 + 2 * b * r) / factor)

    roots, weights = roots_legendre(order)
    rng = 0.5 * kappa0
    phi = rng * roots
    c = jnp.cos(phi + 0.5 * kappa0)

    # integrand is symmetrical so we change integration limits
    s = jnp.sin((phi + rng) / 2)
    s2 = jnp.square(s)

    arg = []
    if l_max >= 1:
        omz2 = jnp.maximum(0, r2 + b2 - 2 * b * r * c)
        z2 = 1 - omz2
        m = jnp.less(z2, 10 * jnp.finfo(omz2.dtype).eps)
        z2 = jnp.where(m, jnp.ones_like(z2), z2)
        z3 = z2 * zero_safe_sqrt(z2)
        cond = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
        omz2 = jnp.where(cond, jnp.ones_like(z2), omz2)
        result = 2 * r * (r - b * c) * (1 - z3) / (3 * omz2)
        arg.append(jnp.where(cond, jnp.zeros_like(result[None, :]), result[None, :]))
    if l_max >= 3:
        f0 = jnp.maximum(
            jnp.zeros_like(r2), jnp.where(k2_cond, 1 - r2, factor * (k2 - s2))
        )
        n = jnp.arange(3, l_max + 1)
        f = f0[None, :] ** (0.5 * n[:, None])
        f *= 2 * r * (r - b + 2 * b * s2[None, :])
        arg.append(f)

    return rng * jnp.sum(jnp.concatenate(arg, axis=0) * weights[None, :], axis=1)


def kite_area(a: Array, b: Array, c: Array) -> Array:
    def sort2(a: Array, b: Array) -> tuple[Array, Array]:
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return zero_safe_sqrt(jnp.maximum(0, square_area))
