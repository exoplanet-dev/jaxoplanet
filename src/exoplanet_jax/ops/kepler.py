# -*- coding: utf-8 -*-

__all__ = ["kepler"]

import jax.numpy as jnp


def kepler(M, ecc):
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


def starter(M, ecc, ome):
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


def refine(M, ecc, ome, E):
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


# def series_starter(mean_anom, ecc):
#     ome = 1.0 - ecc
#     sqrt_ome = jnp.sqrt(ome)

#     chi = mean_anom / (sqrt_ome * ome)
#     Lam = jnp.sqrt(8.0 + 9.0 * jnp.square(chi))
#     S = jnp.cbrt(Lam + 3 * chi)
#     sigma = 6 * chi / (2 + S * S + 4.0 / jnp.square(S))
#     s2 = jnp.square(sigma)
#     s4 = jnp.square(s2)

#     denom = 1.0 / (s2 + 2.0)
#     return (
#         sqrt_ome
#         * sigma
#         * (
#             1
#             + s2
#             * ome
#             * denom
#             * (
#                 (s2 + 20) / 60.0
#                 + s2
#                 * ome
#                 * denom
#                 * denom
#                 * (s2 * s4 + 25 * s4 + 340 * s2 + 840)
#                 / 1400
#             )
#         )
#     )


# def poly_starter(mean_anom, ecc):
#     pi_d_12 = jnp.pi / 12.0

#     g2s_e = 0.2588190451025207623489 * ecc
#     g3s_e = 0.5 * ecc
#     g4s_e = 0.7071067811865475244008 * ecc
#     g5s_e = 0.8660254037844386467637 * ecc
#     g6s_e = 0.9659258262890682867497 * ecc

#     bounds = jnp.stack(
#         [
#             jnp.zeros_like(ecc),
#             pi_d_12 - g2s_e,
#             jnp.pi / 6.0 - g3s_e,
#             0.25 * jnp.pi - g4s_e,
#             jnp.pi / 3.0 - g5s_e,
#             5.0 * pi_d_12 - g6s_e,
#             0.5 * jnp.pi - ecc,
#             7.0 * pi_d_12 - g6s_e,
#             2.0 * jnp.pi / 3.0 - g5s_e,
#             0.75 * jnp.pi - g4s_e,
#             5.0 * jnp.pi / 6.0 - g3s_e,
#             11.0 * pi_d_12 - g2s_e,
#             jnp.full_like(ecc, jnp.pi),
#         ],
#         axis=-1,
#     )
#     k = jnp.clip(
#         jnp.vectorize(
#             partial(jnp.searchsorted, side="right"), signature="(k),()->()"
#         )(bounds, mean_anom),
#         0,
#         11,
#     )

#     bounds_0 = jnp.take(bounds, k)
#     bounds_1 = jnp.take(bounds, k + 1)

#     EA0 = k * pi_d_12
#     EA6 = (k + 1) * pi_d_12

#     sign = jnp.where(jnp.greater_equal(k, 6), 1.0, -1.0)
#     x = 1 / (1 - ((6 - k) * pi_d_12 + sign * jnp.take(bounds, jnp.abs(6 - k))))
#     y = -0.5 * (k * pi_d_12 - bounds_0)
#     EA1 = x
#     EA2 = y * x * x * x

#     x = 1 / (1 - ((5 - k) * pi_d_12 + sign * jnp.take(bounds, jnp.abs(5 - k))))
#     y = -0.5 * ((k + 1) * pi_d_12 - bounds_1)
#     EA7 = x
#     EA8 = y * x * x * x

#     # Solve a matrix equation to get the rest of the coefficients.
#     idx = 1 / (bounds_1 - bounds_0)

#     B0 = idx * (-EA2 - idx * (EA1 - idx * pi_d_12))
#     B1 = idx * (-2 * EA2 - idx * (EA1 - EA7))
#     B2 = idx * (EA8 - EA2)

#     EA3 = B2 - 4 * B1 + 10 * B0
#     EA4 = (-2 * B2 + 7 * B1 - 15 * B0) * idx
#     EA5 = (B2 - 3 * B1 + 6 * B0) * idx * idx

#     dx = mean_anom - bounds_0
#     return EA0 + dx * (EA1 + dx * (EA2 + dx * (EA3 + dx * (EA4 + dx * EA5))))


# def starter(mean_anom, ecc):
#     singular = jnp.logical_and(
#         jnp.greater(ecc, 0.78), jnp.less(2 * mean_anom + (1 - ecc), 0.2)
#     )
#     return jnp.where(
#         singular, series_starter(mean_anom, ecc), poly_starter(mean_anom, ecc)
#     )


# def refine(mean_anom, ecc, ecc_anom):
#     one_over_ecc = 1.0 / ecc

#     sin = jnp.sin(ecc_anom)
#     cos = jnp.cos(ecc_anom)

#     num = (mean_anom - ecc_anom) * one_over_ecc + sin
#     denom = one_over_ecc - cos

#     de = num * (denom * denom + 0.5 * num * sin)
#     de /= denom * denom * denom + num * (denom * sin + num * cos / 6.0)
#     de2o6 = jnp.square(de) / 6.0
#     f1 = 1.0 - 3.0 * de2o6
#     f2 = 1.0 - de2o6
#     sin = sin * f1 + de * f2 * cos
#     cos = cos * f1 - de * f2 * sin
#     return ecc_anom + de, sin, cos


def starter(M, ecc, ome):
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


def refine(M, ecc, ome, E):
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
