# -*- coding: utf-8 -*-

__all__ = ["quad_coeff", "quad_soln"]

from functools import partial

import jax
import jax.numpy as jnp
from scipy.special import roots_legendre


@jax.jit
def quad_coeff(u1, u2):
    c = jnp.array([1 - u1 - 1.5 * u2, u1 + 2 * u2, -0.25 * u2])
    c /= jnp.pi * (c[0] + c[1] / 1.5)
    return c


@partial(jax.jit, static_argnames=("order",))
def quad_soln(b, r, *, order=10):
    return quad_soln_impl(b, r, order)


def quad_soln_impl(b, r, order):
    b = jnp.abs(b)
    r = jnp.abs(r)
    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(
        jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r)
    )
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, kite_area(r, b_, 1), 0)
    kappa0 = jnp.arctan2(area, b2 + factor)
    kappa1 = jnp.arctan2(area, b2 - factor)

    # s0 and s2:

    # Large k
    s0_lrg = jnp.pi * (1 - r2)
    s2_lrg = 2 * s0_lrg + 4 * jnp.pi * (eta2 - 0.5)

    # Small k
    Alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = jnp.pi - Alens
    s2_sml = 2 * s0_sml + 2 * (
        -(jnp.pi - kappa1)
        + 2 * eta2 * kappa0
        - 0.25 * area * (1 + 5 * r2 + b2)
    )

    cond = jnp.greater(onembpr2 / (4 * b * r) + 1, 1)
    s0 = jnp.where(cond, s0_lrg, s0_sml)
    s2 = jnp.where(cond, s2_lrg, s2_sml)

    s1_cond = jnp.logical_and(jnp.less(b, 1 + r), jnp.greater(b, r - 1))
    b_ = jnp.where(s1_cond, b, 1)
    s1 = jnp.where(
        s1_cond,
        s1_impl(b_, r, kappa0, kappa1, order=order),
        2 * (jnp.pi - kappa1) / 3,
    )

    return jnp.stack((s0, s1, s2), axis=-1)


def s1_impl(b, r, kappa0, kappa1, *, order=5):
    # Pre-compute this before broadcasting
    P0 = kappa0 * r

    # Expand the shapes of the arguments to handle the grid dimension
    b = b[..., None]
    r = r[..., None]
    kappa0 = kappa0[..., None]

    # Extract the Legendre polynomial parameters
    roots, weights = roots_legendre(order)

    # Set up the numerical integration grid
    middle = 0.5 * kappa0
    theta = kappa0 * (0.5 * roots + 0.5)
    cos = jnp.cos(theta)

    # Compute the integrand on the grid
    omz2 = jnp.square(r) + jnp.square(b) - 2 * b * r * cos
    z2 = jnp.maximum(0, 1 - omz2)
    integrand = (r - b * cos) * (1 - z2 * jnp.sqrt(z2)) / omz2

    P = P0 * jnp.sum(weights * integrand, axis=-1)
    Q = 2 * (jnp.pi - kappa1)
    return (Q - P) / 3


def kite_area(a, b, c):
    def sort2(a, b):
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(jnp.maximum(square_area, 0.0))
