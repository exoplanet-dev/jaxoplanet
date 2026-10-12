"""Limb-darkened transit light curves with an elementary interior series

This module implements the interior series of ``pizzetti`` (Hippke et al., in
preparation) for the polynomial limb darkening laws supported by jaxoplanet.

Write the intensity as a sum of powers of ``mu = sqrt(1 - r^2)``,
``I(mu) = sum_a c_a mu^a``. For an occultor of radius ``p`` at separation ``z`` that
lies entirely inside the stellar disk (``z + p <= 1``) the flux deficit is

    delta = -(p^2 / 2) sum_a c_a D_{a/2}(z, p) / sum_a c_a / (a + 2),

where ``D_g`` is the mean of ``(1 - r^2)^g`` over the occultor. With
``w = 1 - z^2`` and ``v = 1 / w`` the mean has the binomial expansion

    D_g = w^g sum_{m>=0} beta_m <eps^m>,   beta_m = (-1)^m binom(g, m),

whose disk averages ``<eps^m>`` are polynomials in ``v``, ``z^2`` and ``p^2``. For
integer ``g`` the series terminates (``D_g`` is a polynomial); for half-integer ``g``
it is truncated after ``m = M`` and, because all neglected terms have the same sign,
the truncation error has the rigorous bound

    |S_M - D_g| <= w^g |beta_{M+1}| <eps^(M+1)> / (1 - e),   e = (2 z p + p^2) / w < 1,

which increases with ``z``. A single cut ``z_c`` per light curve, found by bisection,
therefore guarantees a given accuracy for every point with ``z <= z_c``. The points
beyond the cut (near the stellar limb) are computed with the solution of Agol et al.
(2020) from :mod:`jaxoplanet.core.limb_dark` on a compacted array.
"""

__all__ = ["light_curve"]

import math
from collections.abc import Callable
from functools import cache, partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

from jaxoplanet.core.limb_dark import light_curve as _agol_light_curve
from jaxoplanet.types import Array

#: Truncation order of the series (odd): a polynomial of degree M in 1 / (1 - z^2)
M = 11

#: The truncation bound needs M + 1 > g for every exponent g = a / 2, so the
#: polynomial limb darkening law can have at most this many coefficients
MAX_COEFFS = 2 * M


# ------------------------------------------------------------------ static tables
def _beta(g: float, m: int) -> float:
    """(-1)^m binom(g, m)"""
    b = 1.0
    for i in range(1, m + 1):
        b = b * (i - 1 - g) / i
    return b


def _G(m: int, l: int) -> float:
    return math.comb(m, l) * math.comb(l, l // 2) * 2.0 / (2 * m - l + 2)


@cache
def _table(g: float) -> np.ndarray:
    """``A[j, q]`` such that the series truncated after ``m = M`` is
    ``w^g sum_j v^j sum_q A[j, q] p^(2 q)``"""
    A = np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        bm = _beta(g, m)
        if bm == 0.0:
            continue
        for l in range(0, m + 1, 2):
            c = bm * _G(m, l)
            q = m - l // 2
            h = l // 2
            for i in range(h + 1):
                A[m - i, q] += c * math.comb(h, i) * (-1.0) ** i
    return A


@cache
def _law_tables(n: int):
    """Static data for a polynomial law with ``n`` coefficients

    The exponents of mu are ``a = 0, ..., n``. Returns the matrix mapping the
    coefficients of ``(1 - mu)^k`` to those of ``mu^a``, the series tables, and, for
    the even (integer ``g``) and odd (half-integer ``g``) families, the Laurent
    structure of ``sum_a c_a w^g_a P_a(v)`` as a polynomial in ``w`` and ``v``
    """
    na = n + 1
    to_mu = np.array(
        [[(-1.0) ** a * math.comb(k, a) for k in range(na)] for a in range(na)]
    )
    tables = np.stack([_table(a / 2) for a in range(na)])  # (na, M + 1, M + 1)

    families = []
    for parity in (0, 1):
        entries = []  # (a, j, e) with e = floor(g) - j, the power of w
        for a in range(parity, na, 2):
            for j in range(M + 1):
                if np.any(tables[a, j] != 0.0):
                    entries.append((a, j, a // 2 - j))
        if not entries:
            families.append(None)
            continue
        e = np.array([t[2] for t in entries])
        emin, emax = int(e.min()), int(e.max())
        families.append(
            (
                np.array([t[0] for t in entries]),
                np.array([t[1] for t in entries]),
                e - emin,
                emin,
                emax,
            )
        )

    # the half-integer exponents, for the truncation bound
    odd = np.arange(1, na, 2)
    log_beta = np.array([math.log(abs(_beta(a / 2, M + 1))) for a in odd])
    Gt = np.array([_G(M + 1, l) for l in range(0, M + 2, 2)])
    return to_mu, tables, families, odd, log_beta, Gt


def _horner(coeffs: Array, x: Array) -> Array:
    """``coeffs[0] + coeffs[1] x + ...`` for a static number of coefficients"""
    n = coeffs.shape[0]
    out = coeffs[n - 1] * jnp.ones_like(x)
    for i in range(n - 2, -1, -1):
        out = out * x + coeffs[i]
    return out


def _laurent(kappa: Array, emin: int, emax: int, w: Array, v: Array) -> Array:
    """``sum_e kappa[e - emin] w^e`` for static ``emin <= e <= emax``"""
    if emin >= 0:
        return jnp.power(w, emin) * _horner(kappa, w) if emin > 0 else _horner(kappa, w)
    out = v * _horner(kappa[:-emin][::-1], v)
    if emax >= 0:
        out = out + _horner(kappa[-emin:], w)
    return out


# ------------------------------------------------------------------- the series
def mu_coefficients(u: Array) -> Array:
    """Coefficients ``c_a`` of ``I(mu) = sum_a c_a mu^a`` for the limb darkening law
    ``I = 1 - sum_k u_k (1 - mu)^k``"""
    u = jnp.asarray(u)
    to_mu = jnp.asarray(_law_tables(u.shape[0])[0], dtype=u.dtype)
    d = jnp.concatenate((jnp.ones(1, dtype=u.dtype), -u))
    return to_mu @ d


def _scale(c: Array, r: Array) -> Array:
    a = jnp.arange(c.shape[0], dtype=c.dtype)
    return r**2 / (2 * jnp.sum(c / (a + 2)))


def interior_deficit(c: Array, z: Array, r: Array) -> Array:
    """The truncated interior series for the flux deficit, valid (and bounded) for
    ``z <= z_c``; ``z`` must be smaller than one"""
    n = c.shape[0] - 1
    _, tables, families, _, _, _ = _law_tables(n)
    tables = jnp.asarray(tables, dtype=c.dtype)
    r2 = jnp.square(r)
    pw = r2 ** jnp.arange(M + 1, dtype=c.dtype)
    cj = c[:, None] * jnp.einsum("ajq,q->aj", tables, pw)  # (na, M + 1)

    z2 = jnp.square(z)
    w = 1 - z2
    v = 1 / w
    total = jnp.zeros_like(z)
    for parity, fam in enumerate(families):
        if fam is None:
            continue
        a_idx, j_idx, e_off, emin, emax = fam
        kappa = (
            jnp.zeros(emax - emin + 1, dtype=c.dtype).at[e_off].add(cj[a_idx, j_idx])
        )
        term = _laurent(kappa, emin, emax, w, v)
        total = total + (jnp.sqrt(w) * term if parity == 1 else term)
    return -_scale(c, r) * total


# ---------------------------------------------------------------- bound and cut
def _log_flux_bound(c: Array, z: Array, r: Array) -> Array:
    """Log of ``sum_a |c_a| B_M(z, r; a / 2)`` over the half-integer exponents, where
    ``B_M`` bounds the truncation error of the disk mean; the flux error relative
    to the central depth is below this"""
    n = c.shape[0] - 1
    _, _, _, odd, log_beta, Gt = _law_tables(n)
    dtype = c.dtype
    tiny = jnp.finfo(dtype).tiny
    z = jnp.maximum(z, jnp.sqrt(tiny))
    w = 1 - z**2
    e = (2 * z * r + r**2) / w
    lam = 2 * z / (2 * z + r)
    log_h = jnp.log(0.5 * lam)
    log_q = jnp.log(r / (2 * z + r))
    ls = jnp.arange(0, M + 2, 2, dtype=dtype)
    log_s = logsumexp(
        jnp.log(jnp.asarray(Gt, dtype=dtype)) + ls * log_h + (M + 1 - ls) * log_q
    )
    g = jnp.asarray(odd, dtype=dtype) / 2
    log_b = (
        g * jnp.log(w)
        + jnp.asarray(log_beta, dtype=dtype)
        + (M + 1) * jnp.log(e)
        + log_s
        - jnp.log1p(-e)
    )
    log_c = jnp.log(jnp.abs(c[odd]))
    out = logsumexp(log_c + log_b)
    return jnp.where(e < 1, out, jnp.inf)


def series_cut(c: Array, r: Array, tol: Array) -> Array:
    """The largest separation ``z_c`` for which the truncation error of the interior
    series is rigorously below ``tol`` times the central transit depth; -1 if the
    series cannot be used"""
    dtype = c.dtype
    eps = jnp.finfo(dtype).eps
    hi0 = 1 - r
    usable = jnp.logical_and(r > 0, r < 1)
    n = c.shape[0] - 1
    if n < 1:  # uniform: the series is exact
        return jnp.where(usable, hi0 * (1 - 4 * eps), -1.0)

    ltol = jnp.log(jnp.asarray(tol, dtype=dtype))
    r_ = jnp.where(usable, r, 0.5)

    def ok(z):
        return _log_flux_bound(c, z, r_) <= ltol

    def body(_, state):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        good = ok(mid)
        return jnp.where(good, mid, lo), jnp.where(good, hi, mid)

    lo, _ = jax.lax.fori_loop(0, 60, body, (jnp.zeros_like(r_), 1 - r_))
    return jnp.where(jnp.logical_and(usable, ok(jnp.zeros_like(r_))), lo, -1.0)


# ------------------------------------------------------------------ the dispatch
#: The limb points are gathered in blocks of this many consecutive samples
BLOCK = 16
#: The smallest number of blocks in a compacted array
MIN_BLOCKS = 4
#: The factor between the numbers of blocks of successive compacted array sizes
BLOCK_GROWTH = 4


def _capacities(nb: int) -> tuple[int, ...]:
    """The static numbers of blocks of the compacted arrays, from which the smallest
    that holds all the flagged blocks is selected at run time"""
    caps = {min(nb, MIN_BLOCKS * BLOCK_GROWTH**k) for k in range(32)}
    return tuple(sorted(caps | {nb}))


def _compact_apply(fn: Callable[[Array], Array], z: Array, mask: Array) -> Array:
    """``fn(z)`` evaluated only near the points selected by ``mask``, and zero elsewhere

    JAX needs static shapes, so the blocks of ``BLOCK`` consecutive samples that
    contain a selected point are gathered into an array of one of a few fixed sizes,
    chosen with ``lax.switch`` from the number of such blocks. The selected points
    of a light curve are the ingress and egress, which are contiguous in time, so
    this is efficient if the samples are roughly ordered in time. Gathering whole
    blocks, rather than single points, makes the search for the selected points
    ~100 times cheaper than ``jnp.nonzero`` on the individual samples. Without any
    clustering all the blocks are selected and the cost is that of evaluating
    ``fn`` on every point. Under ``vmap`` the predicate is batched and
    ``lax.switch`` evaluates all the branches, so there is no saving there either.

    The result is only meaningful where ``mask`` is true.
    """
    n = z.shape[0]
    nb = -(-n // BLOCK)
    if nb <= MIN_BLOCKS:
        return jnp.where(mask, fn(jnp.where(mask, z, 0.5)), 0.0)

    pad = nb * BLOCK - n
    zb = jnp.pad(z, (0, pad), constant_values=0.5).reshape(nb, BLOCK)
    flag = jnp.pad(mask, (0, pad)).reshape(nb, BLOCK).any(axis=1)
    caps = _capacities(nb)

    def make_branch(cap: int) -> Callable[[Array], Array]:
        def branch(zb: Array) -> Array:
            idx = jnp.nonzero(flag, size=cap, fill_value=nb)[0]
            zs = jnp.take(zb, idx, axis=0, mode="fill", fill_value=0.5)
            vals = fn(zs.reshape(-1)).reshape(cap, BLOCK)
            return jnp.zeros_like(zb).at[idx].set(vals, mode="drop")

        return branch

    which = jnp.sum(jnp.sum(flag) > jnp.asarray(caps[:-1]))
    out = jax.lax.switch(which, [make_branch(cap) for cap in caps], zb)
    return out.reshape(-1)[:n]


def _flat_deficit(
    u: Array, z: Array, r: Array, order: int, tol: Array, *, compact: bool = True
) -> Array:
    """Flux deficit for a one-dimensional array ``z`` and a scalar radius ratio"""
    c = mu_coefficients(u)
    z_c = jax.lax.stop_gradient(series_cut(c, r, tol))

    in_series = z <= z_c
    series = interior_deficit(c, jnp.where(in_series, z, 0.0), r)

    # Points beyond the cut and inside the outer contact need the limb solution
    on_limb = jnp.logical_and(jnp.logical_not(in_series), z < 1 + r)
    limb_fn = partial(_agol_light_curve, u, r=r, order=order)
    if compact:
        limb = _compact_apply(lambda zs: limb_fn(b=zs), z, on_limb)
    else:
        limb = jnp.where(on_limb, limb_fn(b=jnp.where(on_limb, z, 0.5)), 0.0)
    return jnp.where(in_series, series, limb)


@partial(jax.jit, static_argnames=("order",))
def light_curve(
    u: Array, b: Array, r: Array, *, order: int = 40, series_tol: float = 1e-10
) -> Array:
    """The flux deficit (flux minus one) for polynomial limb darkening

    The part of the transit where the occultor is entirely inside the stellar disk
    is computed from an elementary series with a rigorous error bound; the rest uses
    the solution of Agol et al. (2020), see
    :func:`jaxoplanet.core.limb_dark.light_curve`.

    Args:
        u (Array): The coefficients of the polynomial limb darkening model
        b (Array): The center-to-center distance between the occultor and the
            occulted body
        r (Array): The radius ratio between the occultor and the occulted body
        order (int): The quadrature order of the limb solution
        series_tol (float): The bound on the truncation error of the series
            relative to the central depth of the transit. It is raised to a few
            times the machine precision of the working dtype if lower.
    """
    dtype = jnp.result_type(float, u, b, r)
    u = jnp.asarray(u, dtype=dtype)
    b = jnp.abs(jnp.asarray(b, dtype=dtype))
    r = jnp.abs(jnp.asarray(r, dtype=dtype))
    assert u.ndim == 1
    # The uniform law is closed form already, and the bound is only proven for
    # exponents smaller than M + 1
    if u.shape[0] == 0 or u.shape[0] > MAX_COEFFS:
        return _agol_light_curve(u, b, r, order=order)
    tol = jnp.maximum(jnp.asarray(series_tol, dtype=dtype), 8 * jnp.finfo(dtype).eps)

    if r.ndim == 0:
        return _flat_deficit(u, b.reshape(-1), r, order, tol).reshape(b.shape)

    shape = jnp.broadcast_shapes(b.shape, r.shape)
    bb = jnp.broadcast_to(b, shape).reshape(-1)
    rr = jnp.broadcast_to(r, shape).reshape(-1)
    out = jax.vmap(lambda bi, ri: _flat_deficit(u, bi[None], ri, order, tol)[0])(bb, rr)
    return out.reshape(shape)
