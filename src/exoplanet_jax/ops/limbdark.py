# -*- coding: utf-8 -*-

__all__ = ["quad_soln"]

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
from scipy.special import roots_legendre

LimbdarkArgs = namedtuple(
    "Args", ["b", "r", "b2", "r2", "area", "kappa0", "kappa1"]
)


@partial(jax.jit, static_argnames=["order"])
def quad_soln(b, r, *, order=10):
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    area = kite_area(r, b, 1)
    kappa0 = jnp.arctan2(area, b2 + factor)
    kappa1 = jnp.arctan2(area, b2 - factor)
    args = LimbdarkArgs(b, r, b2, jnp.square(r), area, kappa0, kappa1)
    s0, s2 = s0_s2_impl(args)
    s1 = s1_impl(args, order=order)
    return s0, s1, s2


def s0_s2_impl(args):
    b = args.b
    r = args.r
    b2 = args.b2
    r2 = args.r2
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    # Large k
    s0_lrg = jnp.pi * (1 - r2)
    s2_lrg = 2 * s0_lrg + 4 * jnp.pi * (eta2 - 0.5)

    # Small k
    Alens = args.kappa1 + r2 * args.kappa0 - args.area * 0.5
    s0_sml = jnp.pi - Alens
    s2_sml = 2 * s0_sml + 2 * (
        -(jnp.pi - args.kappa1)
        + 2 * eta2 * args.kappa0
        - 0.25 * args.area * (1 + 5 * r2 + b2)
    )

    cond = jnp.greater(onembpr2 / (4 * b * r) + 1, 1)
    return jnp.where(cond, s0_lrg, s0_sml), jnp.where(cond, s2_lrg, s2_sml)


def s1_impl(args, *, order=5):
    # Expand the shapes of the arguments to handle the grid dimension
    b = args.b[..., None]
    r = args.r[..., None]
    kappa0 = args.kappa0[..., None]

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

    P = args.kappa0 * args.r * jnp.sum(weights * integrand, axis=-1)
    Q = 2 * (jnp.pi - args.kappa1)
    return (Q - P) / 3


def kite_area(a, b, c):
    def sort2(a, b):
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(jnp.maximum(square_area, 0.0))
