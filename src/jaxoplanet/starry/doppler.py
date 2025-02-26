import jax
import numpy as np
import jax.numpy as jnp
import scipy

from jaxoplanet.starry.core import basis
from jaxoplanet.starry.core.basis import A1, A2_inv, U
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.core.rotation import dot_rz, left_project
from jaxoplanet.starry.core.solution import rT, solution_vector
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.system_observable import system_observable


def rv_map_expansion(veq=None, inc=None, obl=None, alpha=None):
    cosi = jnp.cos(inc) if inc is not None else 0.0
    sini = jnp.sin(inc) if inc is not None else 1.0
    cosl = jnp.cos(obl) if obl is not None else 1.0
    sinl = jnp.sin(obl) if obl is not None else 0.0
    A = sini * cosl
    B = sini * sinl
    C = cosi
    A2 = A**2
    B2 = B**2
    C2 = C**2
    sq3 = jnp.sqrt(3)

    if veq is None:
        return jnp.array([0.0, 0.0, 0.0, 0.0])

    if alpha is None:
        return jnp.array([0.0, veq * sq3 * B / 3, 0, veq * sq3 * A / 3]) * jnp.pi

    else:
        return (
            jnp.array(
                [
                    0,
                    veq * sq3 * B * (-(A2) * alpha - B2 * alpha - C2 * alpha + 5) / 15,
                    0,
                    veq * sq3 * A * (-(A2) * alpha - B2 * alpha - C2 * alpha + 5) / 15,
                    0,
                    0,
                    0,
                    0,
                    0,
                    veq * alpha * jnp.sqrt(70) * B * (3 * A2 - B2) / 70,
                    veq * alpha * 2 * jnp.sqrt(105) * C * (-(A2) + B2) / 105,
                    veq * alpha * jnp.sqrt(42) * B * (A2 + B2 - 4 * C2) / 210,
                    0,
                    veq * alpha * jnp.sqrt(42) * A * (A2 + B2 - 4 * C2) / 210,
                    veq * alpha * 4 * jnp.sqrt(105) * A * B * C / 105,
                    veq * alpha * jnp.sqrt(70) * A * (A2 - 3 * B2) / 70,
                ]
            )
            * jnp.pi
        )


def surface_radial_velocity(
    surface: Surface,
    r: float | None = None,
    x: float | None = None,
    y: float | None = None,
    z: float | None = None,
    theta: float | None = None,
    order: int = 20,
    higher_precision: bool = False,
):
    if higher_precision:
        try:
            from jaxoplanet.starry.multiprecision import (
                basis as basis_mp,
                utils as utils_mp,
            )
        except ImportError as e:
            raise ImportError(
                "The `mpmath` Python package is required for higher_precision=True."
            ) from e

    total_deg = surface.deg + surface.vdeg

    rT_deg = rT(total_deg)

    x = 0.0 if x is None else x
    y = 0.0 if y is None else y
    z = 0.0 if z is None else z

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

        # trick to avoid nan `x=jnp.where...` grad caused by nan sT
        r = jnp.where(b_rot, 1.0, r)
        b = jnp.where(b_rot, 1.0, b)

        sT = solution_vector(total_deg, order=order)(b, r)

        if total_deg > 0:
            if higher_precision:
                A2 = np.atleast_2d(utils_mp.to_numpy(basis_mp.A2(total_deg)))
            else:
                A2 = scipy.sparse.linalg.inv(A2_inv(total_deg))
                A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([[1]])

        design_matrix_p = jnp.where(b_occ, sT @ A2, rT_deg)

    if surface.ydeg == 0:
        rotated_y = surface.y.todense()
    else:
        rotated_y = left_project(
            surface.ydeg,
            surface._inc,
            surface._obl,
            theta,
            theta_z,
            surface.y.todense(),
        )

    # limb darkening
    if surface.udeg == 0:
        p_u = Pijk.from_dense(jnp.array([1]))
    else:
        u = jnp.array([1, *surface.u])
        p_u = Pijk.from_dense(u @ U(surface.udeg), degree=surface.udeg)

    # surface map * limb darkening map
    if higher_precision:
        A1_val = np.atleast_2d(utils_mp.to_numpy(basis_mp.A1(surface.ydeg)))
    else:
        A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(surface.ydeg))

    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=surface.ydeg)

    norm = np.pi / (p_u.tosparse() @ rT(surface.udeg))

    # the radial velocity map
    y_rv = rv_map_expansion(
        inc=surface._inc, obl=surface._obl, veq=surface.veq, alpha=None
    )

    # rotated to account for the occultor
    rotated_rv_y = dot_rz(surface.vdeg, -theta_z)(y_rv)

    p_rv = Pijk.from_dense(
        jax.experimental.sparse.BCOO.from_scipy_sparse(basis.A1(surface.vdeg)).todense()
        @ rotated_rv_y
    )

    rv = (p_y * p_rv * p_u).todense() @ design_matrix_p
    p_yu = p_y * p_u

    n_deg = (surface.deg + 1) ** 2

    intensity = (p_yu.tosparse() @ design_matrix_p[0:n_deg]) * norm

    return surface.amplitude * rv * norm / intensity


def radial_velocity(system, order=20):
    bodies_rv = system_observable(surface_radial_velocity, order=order)(system)

    def impl(time):
        rvs = bodies_rv(time)
        rvs = rvs.at[:, 0].set(rvs[:, 0] + system.radial_velocity(time)[0].magnitude)
        return rvs

    return impl
