from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy

from jaxoplanet.experimental.starry.basis import A1, A2_inv, U
from jaxoplanet.experimental.starry.orbit import SurfaceSystem
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.solution import rT, solution_vector
from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import quantity_input, unit_registry as ureg


def light_curve(
    system: SurfaceSystem,
) -> Callable[[Quantity], tuple[Array | None, Array | None]]:
    central_bodies_lc = jax.vmap(surface_light_curve, in_axes=(None, 0, 0, 0, 0, None))

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
                system.central_surface, theta=theta
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
    surface: Surface,
    r: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    theta: float = 0.0,
):
    """Light curve of an occulted surface.

    Args:
        map (Map): Surface object
        r (float or None): radius of the occulting body, relative to the current map
           body
        x (float or None): x position of the occulting body, relative to the current
           map body. By default (None) 0.0
        y (float or None): y position of the occulting body, relative to the current
           map body. By default (None) 0.0
        z (float or None): z position of the occulting body, relative to the current
           map body. By default (None) 0.0
        theta (float):
            rotation angle of the map, in radians. By default 0.0

    Returns:
        ArrayLike: flux
    """
    rT_deg = rT(surface.deg)

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        design_matrix_p = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(x) + jnp.square(y))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(z, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(x, y)
        sT = solution_vector(surface.deg)(b, r)

        # scipy.sparse.linalg.inv of a sparse matrix[[1]] is a non-sparse [[1]], hence
        # `from_scipy_sparse`` raises an error (case deg=0)
        if surface.deg > 0:
            A2 = scipy.sparse.linalg.inv(A2_inv(surface.deg))
            A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([1])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    # TODO(lgrcia): Is this the right behavior when map.y is None?
    if surface.y is None:
        rotated_y = jnp.zeros(surface.ydeg)
    else:
        rotated_y = left_project(
            surface.ydeg, surface.inc, surface.obl, theta, theta_z, surface.y.todense()
        )

    # limb darkening
    u = jnp.array([1, *surface.u])
    A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)
    p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)
    p_y = p_y * p_u

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    return (p_y.tosparse() @ design_matrix_p) * norm
