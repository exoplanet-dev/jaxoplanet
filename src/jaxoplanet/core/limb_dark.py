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

from jaxoplanet.core.cel import cel
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


def solution_vector(l_max: int, order: int = 20) -> Callable[[Array, Array], Array]:
    n_max = l_max + 1

    @partial(jnp.vectorize, signature=f"(),()->({n_max})")
    def impl(b: Array, r: Array) -> Array:
        b = jnp.abs(b)
        r = jnp.abs(r)
        s0, s1, s2 = s0s1s2(b, r)
        s = [s0[None]]
        if l_max >= 1:
            s.append(s1[None])
        if l_max >= 2:
            s.append(s2[None])
        if l_max >= 3:
            no_occ = jnp.logical_or(jnp.greater_equal(b, 1 + r), jnp.less_equal(r, 0))
            full_occ = jnp.less_equal(1 + b, r)
            cond = jnp.logical_or(no_occ, full_occ)
            b_ = jnp.where(cond, jnp.ones_like(b), b)
            P = p_integral(order, l_max, b_, r)
            s.append(jnp.where(cond, jnp.zeros_like(P), -P))
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


@jax.custom_jvp
def s0s1s2(b: Array, r: Array) -> tuple[Array, Array, Array]:
    """The first three terms of the solution vector, in closed form"""
    s, _, _ = _s0s1s2_impl(b, r)
    return s


@s0s1s2.defjvp
def _s0s1s2_jvp(primals, tangents):
    b, r = primals
    bt, rt = tangents
    s, dsdb, dsdr = _s0s1s2_impl(b, r)
    return s, tuple(db * bt + dr * rt for db, dr in zip(dsdb, dsdr, strict=True))


def _s0s1s2_impl(
    b: Array, r: Array
) -> tuple[tuple[Array, ...], tuple[Array, ...], tuple[Array, ...]]:
    """Closed form (s0, s1, s2) and their partials with respect to b and r, as a
    branch-free port of ``quad_solution_vector`` from ``exoplanet-core``; under
    JIT the partials are eliminated as dead code when only the values are used
    """
    eps = get_dtype_eps(b)
    tiny = jnp.finfo(jnp.result_type(b)).tiny

    # Fix an instability that exists really close to b = r = 0.5
    hack = jnp.logical_and(jnp.abs(b - r) <= 5 * eps, jnp.abs(r - 0.5) <= 5 * eps)
    b = jnp.where(hack, 0.5 + 5 * eps, b)

    full_occ = b < r - 1
    no_occ = jnp.logical_or(r <= eps, b > r + 1)
    occ = jnp.logical_not(jnp.logical_or(full_occ, no_occ))

    # Placeholder coordinates for the masked-out points, so that every branch
    # below stays finite
    b_ = jnp.where(occ, b, 0.5)
    r_ = jnp.where(occ, r, 0.25)

    b2 = jnp.square(b_)
    r2 = jnp.square(r_)
    invr = 1.0 / r_
    invb = 1.0 / jnp.maximum(b_, tiny)
    bmr = b_ - r_
    bpr = b_ + r_
    fourbr = 4 * b_ * r_
    invfourbr = 0.25 * invr * invb
    # Stable groupings of (1 - (b -/+ r)^2); see p_integral
    onembmr2 = (b_ + (1 - r_)) * ((1 + r_) - b_)
    onembmr2inv = 1.0 / jnp.maximum(onembmr2, tiny)
    onembpr2 = ((1 - r_) - b_) * ((1 + r_) + b_)
    sqonembmr2 = jnp.sqrt(jnp.maximum(onembmr2, 0.0))
    sqbr = jnp.sqrt(b_ * r_)

    area, kap0, kap1 = kappas(b_, r_)

    ksq = onembpr2 * invfourbr + 1.0
    # Accurate complementary modulus on each side of ksq = 1
    kcsq_lt = -onembpr2 * invfourbr
    kcsq_gt = onembpr2 * onembmr2inv
    invksq = fourbr * onembmr2inv

    cb0 = b_ <= eps
    cbr = jnp.abs(bmr) <= eps
    cr_half = jnp.abs(r_ - 0.5) <= eps
    case6 = jnp.logical_and(cbr, cr_half)
    case5 = jnp.logical_and(cbr, jnp.logical_and(~cr_half, r_ < 0.5))
    case7 = jnp.logical_and(cbr, jnp.logical_and(~cr_half, r_ >= 0.5))
    generic = jnp.logical_not(jnp.logical_or(cb0, cbr))
    case2 = jnp.logical_and(generic, ksq < 1)
    case3 = jnp.logical_and(generic, ksq > 1)

    # A single fused cel evaluation covers every case: the modulus is
    # case-dependent, but the second and third integrals are always E(k) and
    # (E(k) - (1 - k^2) K(k)) / k^2 for the selected modulus
    m = 4 * r2
    minv = 1.0 / jnp.maximum(m, tiny)

    def sel(v2, v3, v5, v7, default):
        out = jnp.where(case2, v2, default)
        out = jnp.where(case3, v3, out)
        out = jnp.where(case5, v5, out)
        return jnp.where(case7, v7, out)

    k2_sel = sel(ksq, invksq, m, minv, jnp.full_like(b_, 0.5))
    kcsq_sel = sel(kcsq_lt, kcsq_gt, 1 - m, 1 - minv, jnp.full_like(b_, 0.5))
    kcsq_sel = jnp.maximum(kcsq_sel, 0.0)
    kc_sel = jnp.sqrt(kcsq_sel)

    # The first (Pi-like) integral is only used in cases 2 and 3
    bmrdbpr = bmr / bpr
    mu = 3 * bmrdbpr * onembmr2inv
    p3 = jnp.square(bmrdbpr) * onembpr2 * onembmr2inv
    p_sel = jnp.maximum(
        jnp.where(case2, jnp.square(bmr) * kcsq_lt, jnp.where(case3, p3, 1.0)), tiny
    )
    a1_sel = jnp.where(case3, 1 + mu, 0.0)
    b1_sel = jnp.where(case2, 3 * kcsq_lt * bmr * bpr, jnp.where(case3, p3 + mu, 0.0))

    ones = jnp.ones_like(b_)
    Piofk, Eofk, Em1mKdm = cel(
        k2_sel, kc_sel, p_sel, a1_sel, b1_sel, ones, kcsq_sel, ones, jnp.zeros_like(b_)
    )

    # S0 and S2 (the "eta" form), from Agol et al. (2020) Section 3
    big = jnp.logical_or(ksq > 1, cb0)
    s0 = jnp.where(big, jnp.pi * (1 - r2), jnp.pi - (kap1 + r2 * kap0 - 0.5 * area))
    ds0db = jnp.where(big, 0.0, area * invb)
    ds0dr = jnp.where(big, -2 * jnp.pi * r_, -2 * r_ * kap0)

    r2pb2 = r2 + b2
    eta2 = 0.5 * r2 * (r2pb2 + b2)
    four_pi_eta = jnp.where(
        big,
        4 * jnp.pi * (eta2 - 0.5),
        2 * (-(jnp.pi - kap1) + 2 * eta2 * kap0 - 0.25 * area * (1 + 5 * r2 + b2)),
    )
    detadr = jnp.where(big, 8 * jnp.pi * r_ * r2pb2, 8 * r_ * (r2pb2 * kap0 - area))
    detadb = jnp.where(
        big,
        8 * jnp.pi * b_ * r2,
        2 * invb * (4 * b2 * r2 * kap0 - (1 + r2pb2) * area),
    )
    s2 = 2 * s0 + four_pi_eta
    ds2db = 2 * ds0db + detadb
    ds2dr = 2 * ds0dr + detadr

    # S1, via Lambda_1 from Agol et al. (2020) Section 4; the case numbering
    # follows Table 1 of that paper
    sqbrinv = 1.0 / jnp.maximum(sqbr, tiny)

    # Case 2 / Case 8: generic ksq < 1
    lam2 = (
        onembmr2
        * (Piofk + (-3 + 6 * r2 + 2 * b_ * r_) * Em1mKdm - fourbr * Eofk)
        * sqbrinv
        / 3.0
    )
    ds1db_2 = 2 * r_ * onembmr2 * (-Em1mKdm + 2 * Eofk) * sqbrinv / 3.0
    ds1dr_2 = -2 * r_ * onembmr2 * Em1mKdm * sqbrinv

    # Case 3 / Case 9: generic ksq > 1
    lam3 = 2 * sqonembmr2 * (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) / 3.0
    ds1db_3 = -4 * r_ * sqonembmr2 * (Eofk - 2 * Em1mKdm) / 3.0
    ds1dr_3 = -4 * r_ * sqonembmr2 * Eofk

    # Case 4: ksq == 1 (b + r == 1)
    rootr1mr = jnp.sqrt(jnp.maximum(r_ * (1 - r_), 0.0))
    lam4 = (
        2 * jnp.arccos(jnp.clip(1 - 2 * r_, -1.0, 1.0))
        - 4.0 / 3.0 * (3 + 2 * r_ - 8 * r2) * rootr1mr
        - 2 * jnp.pi * (r_ > 0.5)
    )
    ds1dr_4 = -8 * r_ * rootr1mr
    ds1db_4 = -ds1dr_4 / 3.0

    # Case 5: b == r < 1/2
    lam5 = jnp.pi + 2.0 / 3.0 * ((2 * m - 3) * Eofk - m * Em1mKdm)
    ds1db_5 = -4.0 / 3.0 * r_ * (Eofk - 2 * Em1mKdm)
    ds1dr_5 = -4 * r_ * Eofk

    # Case 6: b == r == 1/2
    lam6 = jnp.full_like(b_, jnp.pi - 4.0 / 3.0)
    ds1db_6 = jnp.full_like(b_, 2.0 / 3.0)
    ds1dr_6 = jnp.full_like(b_, -2.0)

    # Case 7: b == r > 1/2
    lam7 = jnp.pi + invr / 3.0 * (-m * Eofk + (2 * m - 3) * Em1mKdm)
    ds1db_7 = 2.0 / 3.0 * (2 * Eofk - Em1mKdm)
    ds1dr_7 = -2 * Em1mKdm

    # Case 10: b == 0
    sqrt1mr2 = jnp.sqrt(jnp.maximum(1 - r2, 0.0))
    lam10 = -2 * jnp.pi * sqrt1mr2 * sqrt1mr2 * sqrt1mr2
    ds1db_10 = jnp.zeros_like(b_)
    ds1dr_10 = -2 * jnp.pi * r_ * sqrt1mr2

    def select(v10, v6, v5, v7, v2, v3, v4):
        out = jnp.where(case2, v2, jnp.where(case3, v3, v4))
        out = jnp.where(case5, v5, out)
        out = jnp.where(case6, v6, out)
        out = jnp.where(case7, v7, out)
        return jnp.where(cb0, v10, out)

    lam = select(lam10, lam6, lam5, lam7, lam2, lam3, lam4)
    ds1db = select(ds1db_10, ds1db_6, ds1db_5, ds1db_7, ds1db_2, ds1db_3, ds1db_4)
    ds1dr = select(ds1dr_10, ds1dr_6, ds1dr_5, ds1dr_7, ds1dr_2, ds1dr_3, ds1dr_4)

    s1 = (jnp.where(r_ > b_, 0.0, 2 * jnp.pi) - lam) / 3.0

    # Handle the complete occultation and no occultation limits
    def mask(value, no_occ_value, ds):
        v = jnp.where(no_occ, no_occ_value, jnp.where(full_occ, 0.0, value))
        return v, jnp.where(occ, ds[0], 0.0), jnp.where(occ, ds[1], 0.0)

    s0, ds0db, ds0dr = mask(s0, jnp.pi, (ds0db, ds0dr))
    s1, ds1db, ds1dr = mask(s1, 2 * jnp.pi / 3.0, (ds1db, ds1dr))
    s2, ds2db, ds2dr = mask(s2, 0.0, (ds2db, ds2dr))

    return (s0, s1, s2), (ds0db, ds1db, ds2db), (ds0dr, ds1dr, ds2dr)


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
