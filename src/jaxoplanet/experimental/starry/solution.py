from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.core.limb_dark import kite_area
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.types import Array


def solution_vector(
    l_max: int, order: int = 20, diagonal: bool = False
) -> Callable[[Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = kappas(b, r)
        P = p_integral(order, l_max, b, r, kappa0, diagonal=diagonal)
        Q = q_integral(l_max, 0.5 * jnp.pi - kappa1, diagonal=diagonal)
        return Q - P

    return impl


def kappas(b: Array, r: Array) -> tuple[Array, Array]:
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    b_cond = jnp.logical_and(jnp.greater(b, jnp.abs(1 - r)), jnp.less(b, 1 + r))
    b_ = jnp.where(b_cond, b, 1)
    area = jnp.where(b_cond, kite_area(r, b_, 1), 0)
    return jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)


def q_integral(l_max: int, lam: Array, diagonal: bool = False) -> Array:
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

    diagonal_lm_poly = [Pijk.ijk2lm(*ijk) for ijk in Pijk.diagonal_ijk(l_max)]

    for l in range(l_max + 1):  # noqa
        for m in range(-l, l + 1):
            if diagonal and (l, m) not in diagonal_lm_poly:
                U.append(zero)
                continue

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


def p_integral(
    order: int, l_max: int, b: Array, r: Array, kappa0: Array, diagonal: bool = False
) -> Array:
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

    diagonal_lm_poly = [Pijk.ijk2lm(*ijk) for ijk in Pijk.diagonal_ijk(l_max)]

    for l in range(l_max + 1):  # noqa
        fa3 = (2 * r) ** (l - 1) * f0
        for m in range(-l, l + 1):
            if diagonal and (l, m) not in diagonal_lm_poly:
                n += 1
                continue

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


def rT(lmax: int) -> Array:
    rt = [0.0 for _ in range((lmax + 1) * (lmax + 1))]
    amp0 = jnp.pi
    lfac1 = 1.0
    lfac2 = 2.0 / 3.0
    for ell in range(0, lmax + 1, 4):
        amp = amp0
        for m in range(0, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * (ell + 2) * (ell + 2)

    amp0 = 0.5 * jnp.pi
    lfac1 = 0.5
    lfac2 = 4.0 / 15.0
    for ell in range(2, lmax + 1, 4):
        amp = amp0
        for m in range(2, ell + 1, 4):
            mu = ell - m
            nu = ell + m
            rt[ell * ell + ell + m] = amp * lfac1
            rt[ell * ell + ell - m] = amp * lfac1
            if ell < lmax:
                rt[(ell + 1) * (ell + 1) + ell + m + 1] = amp * lfac2
                rt[(ell + 1) * (ell + 1) + ell - m + 1] = amp * lfac2
            amp *= (nu + 2.0) / (mu - 2.0)
        lfac1 /= (ell / 2 + 2) * (ell / 2 + 3)
        lfac2 /= (ell / 2 + 2.5) * (ell / 2 + 3.5)
        amp0 *= 0.0625 * ell * (ell + 4)
    return np.array(rt)
