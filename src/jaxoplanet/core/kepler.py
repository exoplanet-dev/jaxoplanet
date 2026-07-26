"""This module provides the core functionality to solve Kepler's equation in JAX. For
more details, see the :ref:`core-from-scratch` tutorial.
"""

__all__ = ["kepler"]

import jax
import jax.numpy as jnp
from jax.interpreters import ad

from jaxoplanet.types import Array

# Cody-Waite style split of 2*pi used for extended-precision range reduction:
# TWOPI_HI and TWOPI_MID each carry 24 significant bits, so n * TWOPI_HI and
# n * TWOPI_MID are exact for |n| < 2**29 wraps, and TWOPI_LO carries the rest
# of the true 2*pi, including the 2.449e-16 that the float64 representation of
# 2*pi drops. The residual beyond TWOPI_LO is ~3.4e-31.
TWOPI_HI = 6.2831854820251465
TWOPI_MID = -1.7484555314695172e-07
TWOPI_LO = -6.8604979977715316e-15
INV_TWOPI = 0.15915494309189535


@jax.jit
def kepler(M: Array, ecc: Array) -> tuple[Array, Array]:
    """Solve Kepler's equation to compute the true anomaly

    Args:
        M: Mean anomaly
        ecc: Eccentricity

    Returns:
        The sine and cosine of the true anomaly
    """
    return _kepler(M, ecc)


@jax.custom_jvp
def _kepler(M: Array, ecc: Array) -> tuple[Array, Array]:
    # Wrap into the range [0, pi); 'high' records M in (pi, 2*pi), which maps
    # to a sign flip of sin(E)
    high, M = range_reduce(M)

    # Solve
    ome = 1 - ecc
    E = starter(M, ecc, ome)
    E = refine(M, ecc, ome, E)

    # Re-wrap back into the full range
    E = jnp.where(high, 2 * jnp.pi - E, E)

    # Convert to true anomaly; tan(0.5 * f)
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * jnp.tan(0.5 * E)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = 2 * tan_half_f * denom
    cosf = (1 - tan2_half_f) * denom

    return sinf, cosf


@_kepler.defjvp
def _(primals, tangents):
    M, e = primals
    M_dot, e_dot = tangents
    sinf, cosf = _kepler(M, e)

    # Pre-compute some things
    ecosf = e * cosf
    ome2 = 1 - e**2

    def make_zero(tan):
        if type(tan) is ad.Zero:
            return ad.zeros_like_aval(tan.aval)
        else:
            return tan

    # Propagate the derivatives
    f_dot = make_zero(M_dot) * (1 + ecosf) ** 2 / ome2**1.5
    f_dot += make_zero(e_dot) * (2 + ecosf) * sinf / ome2

    return (sinf, cosf), (cosf * f_dot, -sinf * f_dot)


def range_reduce(M: Array) -> tuple[Array, Array]:
    """Reduce M modulo 2*pi to the range [0, pi] in extended precision

    A naive ``M % (2 * jnp.pi)`` pays the rounding error of the float64
    representation of 2*pi once per wrap, so the phase error grows linearly
    with the number of elapsed orbits (~2.4e-16 * M / (2 * pi)). Reducing
    against a 3-term Cody-Waite split of 2*pi instead keeps the reduction
    exact for any physically meaningful M.

    Args:
        M: Mean anomaly, unrestricted

    Returns:
        A tuple ``(high, M_reduced)`` where ``M_reduced`` is in ``[0, pi]``
        and ``high`` flags points that came from ``(pi, 2*pi)`` mod 2*pi,
        i.e. where ``sin(E)`` must be negated.
    """
    n = jnp.round(M * INV_TWOPI)
    r = ((M - n * TWOPI_HI) - n * TWOPI_MID) - n * TWOPI_LO

    # n can be off by one when M * INV_TWOPI rounds across a half-integer; a
    # single correction step brings r back into [-pi, pi]. Since it undoes at
    # most one wrap, a 2-term compensation (float64's 2*pi plus its rounding
    # defect) is enough here
    corr = jnp.where(jnp.abs(r) > jnp.pi, jnp.sign(r), 0.0)
    r = (r - corr * (2 * jnp.pi)) - corr * 2.4492935982947064e-16

    return r < 0, jnp.minimum(jnp.abs(r), jnp.pi)


def starter(M: Array, ecc: Array, ome: Array) -> Array:
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = jnp.square(jnp.cbrt(jnp.abs(r) + jnp.sqrt(q2 * q + r * r)))
    return (2 * r * w / (jnp.square(w) + w * q + q2) + M) / d


def refine(M: Array, ecc: Array, ome: Array, E: Array) -> Array:
    sE = E - jnp.sin(E)
    cE = 1 - jnp.cos(E)

    f_0 = ecc * sE + E * ome - M
    f_1 = ecc * cE + ome
    f_2 = ecc * (E - sE)
    f_3 = 1 - f_1
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    return E + dE
