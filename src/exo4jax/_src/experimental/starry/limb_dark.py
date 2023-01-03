from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import binom, roots_legendre

from exo4jax._src.quad import kite_area
from exo4jax._src.types import Array


@partial(jax.jit, static_argnames=("order",))
def light_curve(u: Array, b: Array, r: Array, *, order: int = 10):
    g = greens_basis_transform(u)
    g /= jnp.pi * (g[0] + g[1] / 1.5)
    s = solution_vector(len(g) - 1, order=order)(b, r)
    return s @ g - 1


def solution_vector(
    l_max: int, order: int = 20
) -> Callable[[Array, Array], Array]:
    n_max = l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = kappas(b, r)
        cond = jnp.logical_and(
            jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r)
        )
        b_ = jnp.where(cond, b, 1)
        P = p_integral(order, l_max, b_, r, kappa0)
        P = jnp.where(cond, P, 0)

        # Hardcoding the Q integral here because Q=0 for n>=2
        lam = 0.5 * jnp.pi - kappa1
        update = [kappa1 - jnp.pi - jnp.cos(lam) * jnp.sin(lam)]
        if l_max > 0:
            update.append(2 * (kappa1 - jnp.pi) / 3)
        return -P.at[jnp.arange(len(update))].add(jnp.stack(update))

    return impl


def greens_basis_transform(u: Array) -> Array:
    u = jnp.append(-1, u)
    size = len(u)
    i = np.arange(size)
    arg = binom(i[None, :], i[:, None]) @ u
    p = (-1) ** (i + 1) * arg
    g = [0 for _ in range(size + 2)]
    for n in range(size - 1, 1, -1):
        g[n] = p[n] / (n + 2) + g[n + 2]
    g[1] = p[1] + 3 * g[3]
    g[0] = p[0] + 2 * g[2]
    return jnp.stack(g[:-2])


def kappas(b: Array, r: Array) -> Tuple[Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(
        jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r)
    )
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, kite_area(r, b_, 1), 0)
    return jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def p_integral(
    order: int, l_max: int, b: Array, r: Array, kappa0: Array
) -> Array:
    b2 = jnp.square(b)
    r2 = jnp.square(r)

    # This is a hack for when r -> 0 or b -> 0, so k2 -> inf
    factor = 4 * b * r
    k2_cond = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
    factor = jnp.where(k2_cond, 1, factor)
    k2 = jnp.maximum(0, (1 - r2 - b2 + 2 * b * r) / factor)

    roots, weights = roots_legendre(order)
    rng = 0.5 * kappa0
    phi = rng * roots
    c = jnp.cos(phi + 0.5 * kappa0)
    s = jnp.sin(phi)
    s2 = jnp.square(s)

    arg = [8 * r2 * (s2 - jnp.square(s2))[None, :]]
    if l_max >= 1:
        omz2 = r2 + b2 - 2 * b * r * c
        cond = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
        omz2 = jnp.where(cond, 1, omz2)
        z2 = jnp.maximum(0, 1 - omz2)
        result = 2 * r * (r - b * c) * (1 - z2 * jnp.sqrt(z2)) / (3 * omz2)
        arg.append(jnp.where(cond, 0, result[None, :]))
    if l_max >= 2:
        f0 = jnp.maximum(0, jnp.where(k2_cond, 1 - r2, factor * (k2 - s2)))
        n = jnp.arange(2, l_max + 1)
        f = f0[None, :] ** (0.5 * n[:, None])
        f *= 2 * r * (r - b + 2 * b * s2[None, :])
        arg.append(f)

    return rng * jnp.sum(
        jnp.concatenate(arg, axis=0) * weights[None, :], axis=1
    )
