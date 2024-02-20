from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.experimental.starry.basis import A1, U0, A2_inv
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.solution import solution_vector


def light_curve(system, time):
    xos, yos, zos = system.relative_position(time)
    theta = time * 2 * jnp.pi / system.central.map.period

    central_body_lc = jax.vmap(map_light_curve, in_axes=(None, None, 0, 0, 0, 0))
    central_bodies_lc = jax.vmap(central_body_lc, in_axes=(None, 0, 0, 0, 0, None))
    central_light_curve = central_bodies_lc(
        system.central.map,
        system.radius.magnitude,
        xos.magnitude,
        yos.magnitude,
        zos.magnitude,
        theta,
    )

    @partial(system.body_vmap, in_axes=(0, 0, 0, None))
    def bodies_lc(body, x, y, z, theta):
        return jax.vmap(map_light_curve, in_axes=(None, None, 0, 0, 0, 0))(
            body.map, 1.0, x, y, z, theta
        )

    bodies_light_curves = bodies_lc(
        -xos.magnitude, -yos.magnitude, -zos.magnitude, theta
    )
    return np.vstack([central_light_curve, bodies_light_curves])


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
# TODO this is failing for ydeg=0
def map_light_curve(map, r, xo, yo, zo, theta):
    U = jnp.array([1, *map.u])
    b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
    b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(zo, 0.0))
    b_occ = jnp.logical_not(b_rot)

    # Occultation
    theta_z = jnp.arctan2(xo, yo)
    sT = solution_vector(map.deg)(b, r)
    A2 = scipy.sparse.linalg.inv(A2_inv(map.deg))
    A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
    sTA2 = sT @ A2

    # full rotation
    rotated_y = left_project(
        map.ydeg, map.inc, map.obl, theta, theta_z, map.y.todense()
    )

    # limb darkening product
    A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(map.ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=map.ydeg)
    p_u = Pijk.from_dense(U @ U0(map.udeg), degree=map.udeg)
    p_y = p_y * p_u

    x = jnp.where(b_occ, sTA2, rT(map.deg))
    norm = np.pi / (p_u.tosparse() @ rT(map.udeg))

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
