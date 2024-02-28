from collections.abc import Callable
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from jax.experimental import sparse

from jaxoplanet.experimental.starry.basis import A1, U0, A2_inv
from jaxoplanet.experimental.starry.orbit import SurfaceMapSystem
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import quantity_input, unit_registry as ureg


def light_curve(
    system: SurfaceMapSystem,
) -> Callable[[Quantity], tuple[Optional[Array], Optional[Array]]]:
    central_bodies_lc = jax.vmap(map_light_curve, in_axes=(None, 0, 0, 0, 0, None))

    @partial(system.surface_map_vmap, in_axes=(0, 0, 0, 0, None))
    def compute_body_light_curve(surface_map, radius, x, y, z, time):
        theta = time.magnitude * 2 * jnp.pi / surface_map.period
        return (
            map_light_curve(
                surface_map,
                (system.central.radius / radius).magnitude,
                (x / radius).magnitude,
                (y / radius).magnitude,
                (z / radius).magnitude,
                theta,
            )
            * surface_map.amplitude
        )

    @quantity_input(time=ureg.day)
    @vectorize
    def light_curve_impl(time: Quantity) -> tuple[Optional[Array], Optional[Array]]:
        xos, yos, zos = system.relative_position(time)

        if system.surface_map is None:
            central_light_curves = None
        else:
            theta = time.magnitude * 2 * jnp.pi / system.surface_map.period
            central_radius = system.central.radius
            central_light_curves = (
                central_bodies_lc(
                    system.surface_map,
                    (system.radius / central_radius).magnitude,
                    (xos / central_radius).magnitude,
                    (yos / central_radius).magnitude,
                    (zos / central_radius).magnitude,
                    theta,
                )
                * system.surface_map.amplitude
            )

        if all(surface_map is None for surface_map in system.surface_maps):
            body_light_curves = None
        else:
            body_light_curves = compute_body_light_curve(  # type: ignore
                system.radius, -xos, -yos, -zos, time
            )

        return central_light_curves, body_light_curves

    return light_curve_impl


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
def map_light_curve(map, r: float, xo: float, yo: float, zo: float, theta: float):
    """Light curve of a map

    Args:
        map (Map): map object
        r (float): radius of the occulting body, relative to the current map body
        xo (float): x position of the occulting body, relative to the current map body
        yo (float): y position of the occulting body, relative to the current map body
        zo (float): z position of the occulting body, relative to the current map body
        theta (float): rotation angle of the map

    Returns:
        ArrayLike: light curve
    """
    if map is None:
        return jnp.zeros_like(theta)

    U = jnp.array([1, *map.u])
    b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
    b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(zo, 0.0))
    b_occ = jnp.logical_not(b_rot)

    # Occultation
    theta_z = jnp.arctan2(xo, yo)
    sT = solution_vector(map.deg)(b, r)
    # the reason for this if is that scipy.sparse.linalg.inv of a sparse matrix[[1]]
    # is a non-sparse [[1]], hence from_scipy_sparse raises an error (case deg=0) ...
    if map.deg > 0:
        A2 = scipy.sparse.linalg.inv(A2_inv(map.deg))
        A2 = sparse.BCOO.from_scipy_sparse(A2)
    else:
        A2 = jnp.array([1])

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
    rt = [0.0 for _ in range((lmax + 1) * (lmax + 1))]
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
