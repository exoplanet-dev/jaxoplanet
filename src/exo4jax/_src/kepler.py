from typing import Tuple

import jax.numpy as jnp

from exo4jax._src.types import Array


def kepler(M: Array, ecc: Array) -> Tuple[Array, Array]:
    """Solve Kepler's equation to compute the true anomaly

    Args:
        M: Mean anomaly
        ecc: Eccentricity

    Returns:
        The sine and cosine of the true anomaly
    """
    # Wrap into the right range
    M = M % (2 * jnp.pi)

    # We can restrict to the range [0, pi)
    high = M > jnp.pi
    M = jnp.where(high, 2 * jnp.pi - M, M)

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


def starter(M: Array, ecc: Array, ome: Array) -> Array:
    M2 = jnp.square(M)
    alpha = 3 * jnp.pi / (jnp.pi - 6 / jnp.pi)
    alpha += 1.6 / (jnp.pi - 6 / jnp.pi) * (jnp.pi - M) / (1 + ecc)
    d = 3 * ome + alpha * ecc
    alphad = alpha * d
    r = (3 * alphad * (d - ome) + M2) * M
    q = 2 * alphad * ome - M2
    q2 = jnp.square(q)
    w = (jnp.abs(r) + jnp.sqrt(q2 * q + r * r)) ** (2.0 / 3)
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
    dE = -f_0 / (
        f_1 + 0.5 * d_4 * f_2 + d_4 * d_4 * f_3 / 6 - d_42 * d_4 * f_2 / 24
    )

    return E + dE
