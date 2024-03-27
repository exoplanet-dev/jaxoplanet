import jax
import jax.numpy as jnp

from jaxoplanet.experimental.starry.cel import cel
from jaxoplanet.utils import get_dtype_eps, where


def solution_vector(b, r):
    b = jnp.abs(b)
    r = jnp.abs(r)
    return _solution_vector(b, r)


@jax.custom_jvp
def _solution_vector(b, r):
    return _solution_vector_jvp((b, r), (jnp.zeros_like(b), jnp.zeros_like(r)))[0]


@_solution_vector.defjvp
def _solution_vector_jvp(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents

    tolerance = 10 * get_dtype_eps(b, r)
    (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr)) = _s0_s2_and_grad(b, r)
    s1, (ds1db, ds1dr) = _s1_and_grad(b, r)

    comp_occ = jnp.less_equal(b, r - 1)
    no_occ = jnp.logical_or(jnp.less(r, tolerance), jnp.greater_equal(b, r + 1))
    cond = jnp.logical_or(comp_occ, no_occ)

    s0 = where(no_occ, jnp.pi, where(comp_occ, 0, s0))
    s1 = where(no_occ, 2 * jnp.pi / 3, where(comp_occ, 0, s1))
    s2 = where(cond, 0, s2)

    ds0 = b_dot * where(cond, 0, ds0db) + r_dot * where(cond, 0, ds0dr)
    ds1 = b_dot * where(cond, 0, ds1db) + r_dot * where(cond, 0, ds1dr)
    ds2 = b_dot * where(cond, 0, ds2db) + r_dot * where(cond, 0, ds2dr)

    return (s0, s1, s2), (ds0, ds1, ds2)


@jax.custom_jvp
def _s0s2(b, r):
    return _s0_s2_and_grad(b, r)[0]


@_s0s2.defjvp
def _(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents
    (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr)) = _s0_s2_and_grad(b, r)
    return (s0, s2), ((b_dot * ds0db + r_dot * ds0dr), (b_dot * ds2db + r_dot * ds2dr))


def _s0_s2_and_grad(b, r):
    area, kappa0, kappa1 = kappas(b, r)

    b2 = jnp.square(b)
    r2 = jnp.square(r)
    r2pb2 = r2 + b2
    bpr = b + r
    onembpr2 = (1 + bpr) * (1 - bpr)
    eta2 = 0.5 * r2 * (r2 + 2 * b2)

    # ksq > 1
    s0_lrg = jnp.pi * (1 - r2)
    offset_lrg = 4 * jnp.pi * (eta2 - 0.5)

    # ksq <= 1
    Alens = kappa1 + r2 * kappa0 - area * 0.5
    s0_sml = jnp.pi - Alens
    offset_sml = 2 * (
        -(jnp.pi - kappa1) + 2 * eta2 * kappa0 - 0.25 * area * (1 + 5 * r2 + b2)
    )

    delta = 4 * b * r
    cond = jnp.greater(onembpr2 + delta, delta)
    s0 = where(cond, s0_lrg, s0_sml)
    s2 = 2 * s0 + where(cond, offset_lrg, offset_sml)

    ds0db = where(cond, 0, area / b)
    ds0dr = -2 * r * where(cond, jnp.pi, kappa0)

    ds2db = 2 * ds0db + where(
        cond,
        8 * jnp.pi * b * r2,
        2 * (4 * b2 * r2 * kappa0 - (1 + r2pb2) * area) / b,
    )
    ds2dr = 2 * ds0dr + where(
        cond, 8 * jnp.pi * r * r2pb2, 8 * r * (r2pb2 * kappa0 - area)
    )

    return (s0, s2), ((ds0db, ds0dr), (ds2db, ds2dr))


@jax.custom_jvp
def _s1(b, r):
    return _s1_and_grad(b, r)[0]


@_s1.defjvp
def _(primals, tangents):
    (b, r) = primals
    (b_dot, r_dot) = tangents
    s, (dsdb, dsdr) = _s1_and_grad(b, r)
    return s, b_dot * dsdb + r_dot * dsdr


def _s1_and_grad(b, r):
    # TODO(dfm): Handle the following edge cases:
    #            - (b >= 1 + r) || (r == 0): no occultation
    #            - b <= r - 1: full occultation
    #            - b == 0
    #            - b == r
    #
    # Perhaps these should be handled at the top level calling function?

    # TODO(dfm): Check this tolerance and add test
    tolerance = 10 * jnp.finfo(jnp.result_type(b, r)).eps

    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    bmr = b - r
    onembpr2 = (1 + bpr) * (1 - bpr)
    onembmr2 = (1 + bmr) * (1 - bmr)
    sqonembmr2 = jnp.sqrt(onembmr2)
    sqbr = jnp.sqrt(b * r)
    ratio = onembpr2 / onembmr2
    fourbr = 4 * b * r
    invfourbr = 1 / fourbr
    bmrdbpr = bmr / bpr
    mu = 3 * bmrdbpr / onembmr2

    ksq = onembpr2 * invfourbr + 1
    ksq_cond = jnp.less(ksq, 1.0)
    kcsq = where(ksq_cond, -onembpr2 * invfourbr, ratio)

    k_arg = where(ksq_cond, ksq, 1 / ksq)
    p = where(ksq_cond, jnp.square(bmr), jnp.square(bmrdbpr)) * kcsq

    a1 = where(ksq_cond, 0.0, 1 + mu)
    b1 = where(ksq_cond, 3 * kcsq * bmr * bpr, p + mu)
    Piofk = cel(k_arg, p, a1, b1)
    Eofk = cel(k_arg, 1.0, 1.0, kcsq)
    Em1mKdm = cel(k_arg, 1.0, 1.0, 0.0)

    lambda1 = where(
        ksq_cond,
        onembmr2
        * (Piofk + (-3 + 6 * r2 + 2 * b * r) * Em1mKdm - fourbr * Eofk)
        / (3 * sqbr),
        2 * sqonembmr2 * (onembpr2 * Piofk - (4 - 7 * r2 - b2) * Eofk) / 3,
    )

    dsdb = where(
        ksq_cond,
        2 * r * onembmr2 * (-Em1mKdm + 2 * Eofk) / (3 * sqbr),
        -4 * r * sqonembmr2 * (Eofk - 2 * Em1mKdm) / 3,
    )
    dsdr = where(
        ksq_cond, -2 * r * onembmr2 * Em1mKdm / sqbr, -4 * r * sqonembmr2 * Eofk
    )

    # Below we handle the edge cases:

    # 1. ksq ~ 1
    ksq_one = jnp.less(jnp.abs(kcsq), tolerance)
    rootr1mr = jnp.sqrt(r * (1 - r))
    lambda1 = where(
        ksq_one,
        2 * jnp.arccos(1.0 - 2.0 * r)
        - 4 * (3 + 2 * r - 8 * r2) * rootr1mr / 3
        - 2 * jnp.pi * (r > 0.5),
        lambda1,
    )
    tmp = 8 * r * rootr1mr
    dsdb = where(ksq_one, -tmp, dsdb)
    dsdr = where(ksq_one, tmp / 3, dsdr)

    return (2 * jnp.pi * (1 - (r > b)) - lambda1) / 3, (dsdb, dsdr)


def kite_area(a, b, c):
    def sort2(a, b):
        return jnp.minimum(a, b), jnp.maximum(a, b)

    a, b = sort2(a, b)
    b, c = sort2(b, c)
    a, b = sort2(a, b)

    square_area = (a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (a + (b - c))
    return jnp.sqrt(jnp.maximum(0, square_area))  # We might not need this maximum


def kappas(b, r):
    # In this version, unlike the one in core/limb_dark.py, we don't need to
    # worry about zero/NaN-safety for the gradients since we also provide closed
    # form JVPs.
    b2 = jnp.square(b)
    factor = (r - 1) * (r + 1)
    area = kite_area(r, b, jnp.ones_like(r))
    return area, jnp.arctan2(area, b2 + factor), jnp.arctan2(area, b2 - factor)
