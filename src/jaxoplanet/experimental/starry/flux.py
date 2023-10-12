import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.basis import A1, basis
from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.experimental.starry.wigner import dot_rotation_matrix


def flux(deg, theta, xo, yo, zo, ro, inc, obl, y, u, f):
    b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
    b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + ro), jnp.less_equal(zo, 0.0))
    b_occ = jnp.logical_not(b_rot)

    # Occultation
    theta_z = jnp.arctan2(xo, yo)
    sT = solution_vector(deg)(b, ro)
    sTA = sT @ basis(deg)
    sTAR = dot_rotation_matrix(deg, None, None, 1.0, theta_z)(sTA)

    x = jnp.where(b_occ, sTAR, rTA1(deg))

    x = dot_rotation_matrix(
        deg, -jnp.cos(obl), -jnp.sin(obl), 0.0, -(0.5 * jnp.pi - inc)
    )(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, obl)(x)
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(x)
    x = dot_rotation_matrix(deg, None, None, 1.0, theta)(x)
    x = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(x)

    return x @ y


def rT(lmax):
    rt = [0 for _ in range((lmax + 1) * (lmax + 1))]
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


def rTA1(lmax):
    return rT(lmax) @ A1(lmax)
