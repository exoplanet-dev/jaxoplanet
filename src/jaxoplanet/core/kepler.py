"""This module provides the core functionality to solve Kepler's equation in JAX. For
more details, see the :ref:`core-from-scratch` tutorial.

The implementation uses Markley's (1995) starter followed by a single 4th-order
Householder refinement, like the solver in ``exoplanet-core``, but it never
calls a libm trig function:

- the mean anomaly is reduced to ``[0, pi]`` with an extended-precision
  Cody-Waite reduction, so the accuracy doesn't degrade with the number of
  elapsed orbits,
- ``sin(E)`` and ``cos(E)`` are evaluated with a short polynomial plus a square
  root (following Brandt's calcEA; valid because ``E`` is range-reduced), and
  updated through the refinement step with a series in the correction,
- the conversion to true anomaly uses the half-angle identities
  ``tan(E/2) = sin(E) / (1 + cos(E)) = (1 - cos(E)) / sin(E)``, picking
  whichever is well conditioned, instead of calling ``tan``.
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

# Polynomial coefficients for sin(x) with |x| <= pi/4 (the Cephes/xsimd fit)
SIN_COEFFS = (
    -1.66666666666666307295e-1,
    8.33333333332211858878e-3,
    -1.98412698295895385996e-4,
    2.75573136213857245213e-6,
    -2.50507477628578072866e-8,
    1.58962301576546568060e-10,
)


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
    # Wrap into the range [0, pi]; 'high' records M in (pi, 2*pi), which maps
    # to a sign flip of sin(E) and sin(f)
    high, M = range_reduce(M)

    # Solve: Markley starter, then one 4th-order Householder step. E is in
    # [0, pi] so sin(E) and cos(E) come from a short polynomial plus a square
    # root, and are updated through the refinement with a series in dE
    ome = 1 - ecc
    E = starter(M, ecc, ome)
    sinE, cosE = sincos_reduced(E)

    f_0 = E - ecc * sinE - M
    f_1 = 1 - ecc * cosE
    f_2 = ecc * sinE
    f_3 = ecc * cosE
    d_3 = -f_0 / (f_1 - 0.5 * f_0 * f_2 / f_1)
    d_4 = -f_0 / (f_1 + 0.5 * d_3 * f_2 + (d_3 * d_3) * f_3 / 6)
    d_42 = d_4 * d_4
    dE = -f_0 / (f_1 + 0.5 * d_4 * f_2 + d_42 * f_3 / 6 - d_42 * d_4 * f_2 / 24)

    # sin(E + dE) and cos(E + dE) by series: cos(dE) to O(dE^4) and sin(dE) to
    # O(dE^5), matching the error order of the Householder step itself
    dE2 = dE * dE
    cos_dE = 1 - 0.5 * dE2 * (1 - dE2 / 12)
    sin_dE = dE * (1 - dE2 / 6 * (1 - dE2 / 20))
    sinE, cosE = sinE * cos_dE + sin_dE * cosE, cosE * cos_dE - sin_dE * sinE

    # Undo the range reduction
    sinE = jnp.where(high, -sinE, sinE)

    # tan(0.5 * f) = sqrt((1 + ecc) / (1 - ecc)) * tan(0.5 * E), evaluating
    # tan(0.5 * E) as sin(E) / (1 + cos(E)) in the first half plane and
    # (1 - cos(E)) / sin(E) in the second to avoid the cancellation in
    # 1 + cos(E) as E -> pi. The remaining singular point is E == pi exactly,
    # where f = pi
    use_first = cosE > 0
    num = jnp.where(use_first, sinE, 1 - cosE)
    den = jnp.where(use_first, 1 + cosE, sinE)
    safe = den != 0
    tan_half_f = jnp.sqrt((1 + ecc) / (1 - ecc)) * num / jnp.where(safe, den, 1.0)
    tan2_half_f = jnp.square(tan_half_f)

    # Then we compute sin(f) and cos(f) using:
    #  sin(f) = 2*tan(0.5*f)/(1 + tan(0.5*f)^2), and
    #  cos(f) = (1 - tan(0.5*f)^2)/(1 + tan(0.5*f)^2)
    denom = 1 / (1 + tan2_half_f)
    sinf = jnp.where(safe, 2 * tan_half_f * denom, 0.0)
    cosf = jnp.where(safe, (1 - tan2_half_f) * denom, -1.0)

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


def shortsin(x: Array) -> Array:
    """sin(x) for |x| <= pi/4 by polynomial"""
    z = x * x
    y = SIN_COEFFS[-1]
    for c in SIN_COEFFS[-2::-1]:
        y = y * z + c
    return x + x * z * y


def sincos_reduced(x: Array) -> tuple[Array, Array]:
    """sin(x) and cos(x) for x in [0, pi], without calling libm trig

    One polynomial evaluation on an argument folded into [-pi/4, pi/4] gives
    one of sin(x) or cos(x); the other follows from a square root, which is
    cheaper than a second trig call.
    """
    lo = x < 0.25 * jnp.pi
    hi = x > 0.75 * jnp.pi
    arg = jnp.where(lo, x, jnp.where(hi, jnp.pi - x, 0.5 * jnp.pi - x))
    s = shortsin(arg)
    root = jnp.sqrt(1 - s * s)
    sinx = jnp.where(lo | hi, s, root)
    cosx = jnp.where(lo, root, jnp.where(hi, -root, s))
    return sinx, cosx
