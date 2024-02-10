import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.experimental.starry.basis import A1, U0, A2_inv
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.solution import solution_vector


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
def light_curve(ydeg, y, u, inc, obl, theta, xo, yo, zo, ro):
    udeg = len(u)
    U = jnp.array([1, *u])
    full_deg = ydeg + udeg
    b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
    b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + ro), jnp.less_equal(zo, 0.0))
    b_occ = jnp.logical_not(b_rot)

    # Occultation
    theta_z = jnp.arctan2(xo, yo)
    sT = solution_vector(full_deg)(b, ro)
    A2 = scipy.sparse.linalg.inv(A2_inv(full_deg))
    A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
    sTA2 = sT @ A2

    # full rotation
    rotated_y = left_project(ydeg, inc, obl, theta, theta_z, y)

    # limb darkening product
    A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=ydeg)
    p_u = Pijk.from_dense(U @ U0(udeg), degree=udeg)
    p_y = p_y * p_u

    x = jnp.where(b_occ, sTA2, rT(full_deg))
    norm = np.pi / (p_u.tosparse() @ rT(udeg))

    return (p_y.tosparse() @ x) * norm


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