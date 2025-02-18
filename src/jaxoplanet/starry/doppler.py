import jax
import jax.numpy as jnp

from jaxoplanet.starry import Surface
from jaxoplanet.starry.core import basis
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.light_curves import _design_matrix_elements, surface_light_curve
from jaxoplanet.starry.utils import RSUN_DAY_TO_M_S


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

    design_matrix_p, p_y, p_u, norm = _design_matrix_elements(
        surface=surface,
        r=r,
        x=x,
        y=y,
        z=z,
        theta=theta,
        order=order,
        higher_precision=higher_precision,
        rv=True,
    )

    p_rv = Pijk.from_dense(
        jax.experimental.sparse.BCOO.from_scipy_sparse(basis.A1(surface.vdeg)).todense()
        @ rv_map_expansion(
            inc=surface._inc, obl=surface._obl, veq=surface.veq, alpha=None
        )
    )

    rv = (p_y * p_rv * p_u).todense() @ design_matrix_p
    p_yu = p_y * p_u

    n_deg = (surface.deg + 1) ** 2

    intensity = (p_yu.tosparse() @ design_matrix_p[0:n_deg]) * norm

    return surface.amplitude * RSUN_DAY_TO_M_S * rv * norm / intensity
