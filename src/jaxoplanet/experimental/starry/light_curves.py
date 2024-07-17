from collections.abc import Callable
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.experimental.starry.basis import U0
from jaxoplanet.experimental.starry.mpcore.solution import rT
from jaxoplanet.experimental.starry.mpcore.basis import A1, A2_inv
from jaxoplanet.experimental.starry.mpcore.utils import to_numpy
from jaxoplanet.experimental.starry.orbit import SurfaceSystem
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import quantity_input, unit_registry as ureg


def light_curve(
    system: SurfaceSystem, order: int = 20
) -> Callable[[Quantity], tuple[Optional[Array], Optional[Array]]]:
    central_bodies_lc = jax.vmap(
        surface_light_curve, in_axes=(None, 0, 0, 0, 0, None, None)
    )

    @partial(system.surface_vmap, in_axes=(0, 0, 0, 0, None))
    def compute_body_light_curve(surface, radius, x, y, z, time):
        if surface is None:
            return 0.0
        else:
            theta = surface.rotational_phase(time.magnitude)
            return surface_light_curve(
                surface,
                (system.central.radius / radius).magnitude,
                (x / radius).magnitude,
                (y / radius).magnitude,
                (z / radius).magnitude,
                theta,
                order=order,
            )

    @quantity_input(time=ureg.day)
    @vectorize
    def light_curve_impl(time: Quantity) -> Array:
        xos, yos, zos = system.relative_position(time)
        n = len(xos.magnitude)

        if system.central_surface is None:
            central_light_curves = jnp.array([0.0])
        else:
            theta = system.central_surface.rotational_phase(time.magnitude)
            central_radius = system.central.radius
            central_phase_curve = surface_light_curve(
                system.central_surface, theta=theta, order=order
            )
            central_light_curves = (
                central_bodies_lc(
                    system.central_surface,
                    (system.radius / central_radius).magnitude,
                    (xos / central_radius).magnitude,
                    (yos / central_radius).magnitude,
                    (zos / central_radius).magnitude,
                    theta,
                )
                * system.central_surface.amplitude
            )

            if n > 1 and central_light_curves is not None:
                central_light_curves = central_light_curves.sum(
                    0
                ) - central_phase_curve * (n - 1)
                central_light_curves = jnp.expand_dims(central_light_curves, 0)

        body_light_curves = compute_body_light_curve(
            system.radius, -xos, -yos, -zos, time
        )

        return jnp.hstack([central_light_curves, body_light_curves])

    return light_curve_impl


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
def surface_light_curve(
    surface,
    r: float = None,
    xo: float = None,
    yo: float = None,
    zo: float = None,
    theta: float = 0.0,
    order: int = 20,
):
    """Light curve of an occulted map.

    Args:
        map (Map): map object
        r (float or None): radius of the occulting body, relative to the current map
           body
        xo (float or None): x position of the occulting body, relative to the current
           map body
        yo (float or None): y position of the occulting body, relative to the current
           map body
        zo (float or None): z position of the occulting body, relative to the current
           map body
        theta (float): rotation angle of the map

    Returns:
        ArrayLike: flux
    """
    rT_deg = to_numpy(rT(surface.deg))

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        x = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(zo, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(xo, yo)

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 0.0, r)
        b = jnp.where(b_rot, 0.0, b)
        sT = solution_vector(surface.deg, order=order)(b, r)

        # scipy.sparse.linalg.inv of a sparse matrix[[1]] is a non-sparse [[1]], hence
        # `from_scipy_sparse`` raises an error (case deg=0)
        if surface.deg > 0:
            A2 = to_numpy(A2_inv(surface.deg) ** -1)
        else:
            A2 = jnp.array([1])

        x = jnp.where(b_occ, sT @ A2, rT_deg)

    # TODO(lgrcia): Is this the right behavior when map.y is None?
    if surface.y is None:
        rotated_y = jnp.zeros(surface.ydeg)
    else:
        rotated_y = left_project(
            surface.ydeg,
            surface.inc,
            surface.obl,
            theta + surface.phase,
            theta_z,
            surface.y.todense(),
        )

    # limb darkening
    U = jnp.array([1, *surface.u])
    A1_val = to_numpy(A1(surface.ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)
    p_u = Pijk.from_dense(U @ U0(surface.udeg), degree=surface.udeg)
    p_y = p_y * p_u

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    return surface.amplitude * (p_y.tosparse() @ x) * norm


def rT(lmax: int) -> Array:
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


def rTA1(lmax: int) -> Array:
    return rT(lmax) @ A1(lmax)
