from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import roots_legendre

from jaxoplanet.core.limb_dark import kite_area, quad_nodes, s0s1s2
from jaxoplanet.types import Array
from jaxoplanet.utils import get_dtype_eps, zero_safe_sqrt


def solution_vector(l_max: int, order: int = 20) -> Callable[[Array, Array], Array]:
    n_max = l_max**2 + 2 * l_max + 1

    @jax.jit
    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        kappa0, kappa1 = kappas(b, r)
        s0, s1, _ = s0s1s2(b, r)
        P = p_integral(order, l_max, b, r, kappa0)
        Q = q_integral(l_max, 0.5 * jnp.pi - kappa1)
        s = Q - P
        # The first term and the linear limb darkening term ((l, m) = (1, 0))
        # have closed form solutions
        s = s.at[0].set(s0)
        if l_max >= 1:
            s = s.at[2].set(s1)
        return s

    return impl


def kappas(b: Array, r: Array) -> tuple[Array, Array]:
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
    """Numerical integration of the P integral using the Gauss-Legendre quadrature.

    As described in Equation D32 of Luger et al. (2019), there are 6 cases to consider.
    The first case (mu/2 even) is precise at very low order in the original angle
    variable, so ``low_order = min(order, 20)`` is used there. The other cases
    (except the linear limb darkening term, which is computed in closed form
    elsewhere) carry a factor of ``(k^2 - sin^2(x))^(3/2)`` with an endpoint
    singularity in the grazing regime (k^2 < 1), so for those terms we substitute
    ``sin(x) = k sin(theta)``, which maps the integrals onto a fixed range with
    smooth integrands. The integrands are symmetrical in phi, so we evaluate the
    integrals over half the range and multiply by 2.


    Parameters
    ----------
    order : int
        The order of the Gauss-Legendre quadrature.
    l_max : int
        The maximum degree of the spherical harmonic expansion.
    b : Array
        Impact parameter.
    r : Array
        Occultor radius.
    kappa0 : Array
        k0 angle.

    Returns
    -------
    Array
        The integral of the P function over the occultor surface.
    """
    eps = 10 * get_dtype_eps(b)
    tiny = jnp.finfo(jnp.result_type(b)).tiny

    fourbr = 4 * b * r
    # Stable groupings of (1 - (b -/+ r)^2); see jaxoplanet.core.limb_dark
    onembmr2 = (b + (1 - r)) * ((1 + r) - b)
    onembpr2 = ((1 - r) - b) * ((1 + r) + b)

    # k^2 >= 1 if and only if b + r <= 1; when b * r underflows we route those
    # points through the "inside" branch, which never needs k^2
    degenerate = jnp.less(fourbr, jnp.sqrt(tiny))
    inside = jnp.logical_or(onembpr2 >= 0, degenerate)
    k2 = jnp.where(degenerate, 2.0, onembmr2 / jnp.where(degenerate, 1.0, fourbr))
    k2c = jnp.clip(k2, 0.0, 1.0)
    k = zero_safe_sqrt(k2c)

    # This is a hack for when r -> 0
    r_cond = jnp.less(r, eps)
    delta = (b - r) / (2 * jnp.where(r_cond, 1, r))
    rng = 0.25 * kappa0

    # low order variables, in the original angle variable over (0, kappa0 / 2)
    low_order = np.min([order, 20])
    roots, low_weights = roots_legendre(low_order)
    phi = rng * (roots + 1)
    low_s2 = jnp.square(jnp.sin(phi))
    low_a1 = low_s2 - jnp.square(low_s2)
    low_a2 = jnp.where(r_cond, 0, delta + low_s2)

    # high order variables, at theta nodes over the fixed range (0, pi / 2); the
    # substitution contributes the extra Jacobian factor high_jac
    theta, high_weights = quad_nodes(order)
    st2 = np.square(np.sin(theta))
    ct = np.cos(theta)
    ct2 = np.square(ct)

    high_s2 = jnp.where(inside, st2, k2c * st2)
    f0 = jnp.maximum(jnp.where(inside, onembmr2 - fourbr * st2, onembmr2 * ct2), 0.0)
    high_jac = jnp.where(inside, 1.0, k * ct / jnp.sqrt(1 - k2c * st2))
    high_f0 = f0**1.5 * high_jac
    high_a1 = high_s2 - jnp.square(high_s2)
    high_a2 = jnp.where(r_cond, 0, delta + high_s2)
    high_a4 = 1 - 2 * high_s2

    low_indices = []
    low_integrand = []

    high_indices = []
    high_integrand = []

    n = 0

    for l in range(l_max + 1):  # noqa
        high_fa3 = (2 * r) ** (l - 1) * high_f0
        for m in range(-l, l + 1):
            mu = l - m
            nu = l + m

            if mu == 1 and l == 1:
                # Computed in closed form and set directly in solution_vector
                n += 1
                continue

            elif mu % 2 == 0 and (mu // 2) % 2 == 0:
                f = (
                    2
                    * (2 * r) ** (l + 2)
                    * low_a1 ** (0.25 * (mu + 4))
                    * low_a2 ** (0.5 * nu)
                )
                low_integrand.append(2 * f)
                low_indices.append(n)

            elif mu == 1 and l % 2 == 0:
                f = high_fa3 * high_a1 ** (l // 2 - 1) * high_a4
                high_integrand.append(2 * f)
                high_indices.append(n)

            elif mu == 1:
                f = high_fa3 * high_a1 ** ((l - 3) // 2) * high_a2 * high_a4
                high_integrand.append(2 * f)
                high_indices.append(n)

            elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
                f = (
                    2
                    * high_fa3
                    * high_a1 ** ((mu - 1) // 4)
                    * high_a2 ** (0.5 * (nu - 1))
                )
                high_integrand.append(2 * f)
                high_indices.append(n)

            else:
                n += 1
                continue

            n += 1

    P = jnp.zeros(l_max**2 + 2 * l_max + 1)

    if low_integrand:
        low_P0 = rng * jnp.sum(jnp.stack(low_integrand) * low_weights, axis=1)
        P = P.at[np.stack(low_indices)].set(low_P0)

    if high_integrand:
        # The factor of 1/2 compensates the symmetry factor of 2 included in the
        # integrands, which quad_nodes' weights already account for
        high_P0 = 0.5 * jnp.sum(jnp.stack(high_integrand) * high_weights, axis=1)
        P = P.at[np.stack(high_indices)].set(high_P0)

    return P


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
