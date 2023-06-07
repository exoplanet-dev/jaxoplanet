from functools import partial
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.core.limb_dark import kite_area
from jaxoplanet.types import Array


def solution_vector(l_max: int, order: int = 20) -> Callable[[Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = kappas(b, r)
        P = p_integral(order, l_max, b, r, kappa0)
        Q = q_integral(l_max, 0.5 * jnp.pi - kappa1)
        return Q - P

    return impl


def kappas(b: Array, r: Array) -> Tuple[Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, kite_area(r, b_, 1), 0)
    return jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def q_integral(l_max: int, lam: Array) -> Array:
    zero = jnp.zeros_like(lam)
    c = jnp.cos(lam)
    s = jnp.sin(lam)
    h = {
        (0, 0): 2 * lam + jnp.pi,
        (0, 1): -2 * c,
    }

    def get(u: int, v: int) -> Array:
        if (u, v) in h:
            return h[(u, v)]
        if u >= 2:
            comp = 2 * c ** (u - 1) * s ** (v + 1) + (u - 1) * get(u - 2, v)
        else:
            assert v >= 2
            comp = -2 * c ** (u + 1) * s ** (v - 1) + (v - 1) * get(u, v - 2)
        comp /= u + v
        h[(u, v)] = comp
        return comp

    U = []
    for l in range(l_max + 1):  # noqa
        for m in range(-l, l + 1):
            if l == 1 and m == 0:
                U.append((np.pi + 2 * lam) / 3)
                continue
            mu = l - m
            nu = l + m
            if (mu % 2) == 0 and (mu // 2) % 2 == 0:
                u = mu // 2 + 2
                v = nu // 2
                assert u % 2 == 0
                U.append(get(u, v))
            else:
                U.append(zero)

    return jnp.stack(U)


def p_integral(order: int, l_max: int, b: Array, r: Array, kappa0: Array) -> Array:
    b2 = jnp.square(b)
    r2 = jnp.square(r)

    # This is a hack for when r -> 0 or b -> 0, so k2 -> inf
    factor = 4 * b * r
    k2_cond = jnp.less(factor, 10 * jnp.finfo(factor.dtype).eps)
    factor = jnp.where(k2_cond, 1, factor)
    k2 = jnp.maximum(0, (1 - r2 - b2 + 2 * b * r) / factor)

    # And for when r -> 0
    r_cond = jnp.less(r, 10 * jnp.finfo(r.dtype).eps)
    delta = (b - r) / (2 * jnp.where(r_cond, 1, r))

    roots, weights = roots_legendre(order)
    rng = 0.5 * kappa0
    phi = rng * roots
    c = jnp.cos(phi + 0.5 * kappa0)
    s = jnp.sin(phi)
    s2 = jnp.square(s)

    f0 = jnp.maximum(0, jnp.where(k2_cond, 1 - r2, factor * (k2 - s2))) ** 1.5
    a1 = s2 - jnp.square(s2)
    a2 = jnp.where(r_cond, 0, delta + s2)
    a4 = 1 - 2 * s2

    ind = []
    arg = []
    n = 0
    for l in range(l_max + 1):  # noqa
        fa3 = (2 * r) ** (l - 1) * f0
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m

            if mu == 1 and l == 1:
                omz2 = r2 + b2 - 2 * b * r * c
                cond = jnp.less(omz2, 10 * jnp.finfo(omz2.dtype).eps)
                omz2 = jnp.where(cond, 1, omz2)
                z2 = jnp.maximum(0, 1 - omz2)
                result = 2 * r * (r - b * c) * (1 - z2 * jnp.sqrt(z2)) / (3 * omz2)
                arg.append(jnp.where(cond, 0, result))

            elif mu % 2 == 0 and (mu // 2) % 2 == 0:
                arg.append(
                    2 * (2 * r) ** (l + 2) * a1 ** (0.25 * (mu + 4)) * a2 ** (0.5 * nu)
                )

            elif mu == 1 and l % 2 == 0:
                arg.append(fa3 * a1 ** (l // 2 - 1) * a4)

            elif mu == 1:
                arg.append(fa3 * a1 ** ((l - 3) // 2) * a2 * a4)

            elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
                arg.append(2 * fa3 * a1 ** ((mu - 1) // 4) * a2 ** (0.5 * (nu - 1)))

            else:
                n += 1
                continue

            ind.append(n)
            n += 1

    P0 = rng * jnp.sum(jnp.stack(arg) * weights[None, :], axis=1)
    P = jnp.zeros(l_max**2 + 2 * l_max + 1)

    # Yes, using np not jnp here: 'ind' is always static.
    inds = np.stack(ind)

    return P.at[inds].set(P0)
