"""Bulirsch's general complete elliptic integral ``cel``, following Bulirsch (1965)
as used by `Agol et al. (2020) <https://arxiv.org/abs/1908.03222>`_.
"""

__all__ = ["cel"]

import jax
import jax.numpy as jnp

from jaxoplanet.types import Array
from jaxoplanet.utils import get_dtype_eps

# The recursion converges quadratically in at most ~9 iterations for any kc down
# to machine epsilon, and it is self-similar after convergence, so running a
# fixed (branch-free) number of iterations doesn't corrupt the result
CEL_MAX_ITER = 10


def cel(
    k2: Array,
    kc: Array,
    p: Array,
    a1: Array,
    b1: Array,
    a2: Array,
    b2: Array,
    a3: Array,
    b3: Array,
    *,
    unroll: int = 5,
) -> tuple[Array, Array, Array]:
    """Evaluate ``cel(kc, p, a1, b1)``, ``cel(kc, 1, a2, b2)``, and
    ``cel(kc, 1, a3, b3)`` elementwise, sharing a single modulus recursion; the
    second and third integrals also share their ``p = 1`` characteristic
    recursion. ``kc`` is passed separately from ``k2 = 1 - kc^2`` since it can
    often be computed more accurately directly. The recursion runs as a
    ``lax.scan`` with the given ``unroll`` factor, which trades compile time
    against runtime.
    """
    eps = get_dtype_eps(k2)
    tiny = jnp.finfo(jnp.result_type(k2)).tiny

    k2 = jnp.clip(k2, 0.0, 1.0)

    # If k^2 is very small we get better precision computing kc directly, and
    # kc must be nonzero, so push exact zeros to a small number
    kc = jnp.where(k2 < 1e-5, jnp.sqrt(1.0 - k2), kc)
    kc = jnp.where(kc <= 0.0, eps * jnp.maximum(k2, eps), kc)

    # Apply the p <= 0 transformation to the first integral; both branches are
    # always evaluated so the guards keep the unselected branch finite
    pos = jnp.greater(p, 0.0)
    g_ = jnp.where(pos, 1.0, 1.0 - p)
    f_ = g_ - k2
    q_ = k2 * (b1 - a1 * p)
    p_neg = jnp.sqrt(jnp.maximum(f_, tiny) / g_)
    a1_neg = (a1 - b1) / g_
    b1_neg = -q_ / (g_ * g_ * p_neg) + a1_neg * p_neg
    p_pos = jnp.sqrt(jnp.where(pos, p, 1.0))
    b1_pos = b1 / p_pos
    pa = jnp.where(pos, p_pos, p_neg)
    a1 = jnp.where(pos, a1, a1_neg)
    b1 = jnp.where(pos, b1_pos, b1_neg)

    pb = jnp.ones_like(pa)

    def update(ee, pa, a1, b1, pb, a2, b2, a3, b3):
        rpa = 1.0 / pa
        ga = ee * rpa
        a1, b1 = a1 + b1 * rpa, 2.0 * (b1 + a1 * ga)
        pa = pa + ga
        rpb = 1.0 / pb
        gb = ee * rpb
        a2, b2 = a2 + b2 * rpb, 2.0 * (b2 + a2 * gb)
        a3, b3 = a3 + b3 * rpb, 2.0 * (b3 + a3 * gb)
        pb = pb + gb
        return pa, a1, b1, pb, a2, b2, a3, b3

    ee = kc
    m = jnp.ones_like(kc)
    pa, a1, b1, pb, a2, b2, a3, b3 = update(ee, pa, a1, b1, pb, a2, b2, a3, b3)
    m = m + kc

    def body(carry, _):
        kc, ee, m, pa, a1, b1, pb, a2, b2, a3, b3 = carry
        kc = 2.0 * jnp.sqrt(ee)
        ee = kc * m
        pa, a1, b1, pb, a2, b2, a3, b3 = update(ee, pa, a1, b1, pb, a2, b2, a3, b3)
        m = m + kc
        return (kc, ee, m, pa, a1, b1, pb, a2, b2, a3, b3), None

    (kc, ee, m, pa, a1, b1, pb, a2, b2, a3, b3), _ = jax.lax.scan(
        body,
        (kc, ee, m, pa, a1, b1, pb, a2, b2, a3, b3),
        None,
        length=CEL_MAX_ITER,
        unroll=unroll,
    )

    f1 = 0.5 * jnp.pi * (a1 * m + b1) / (m * (m + pa))
    norm = 0.5 * jnp.pi / (m * (m + pb))
    f2 = norm * (a2 * m + b2)
    f3 = norm * (a3 * m + b3)
    return f1, f2, f3
