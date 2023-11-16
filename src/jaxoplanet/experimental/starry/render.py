import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, poly_basis
from jaxoplanet.experimental.starry.wigner import dot_rotation_matrix


def ortho_grid(res):
    x, y = jnp.meshgrid(jnp.linspace(-1, 1, res), jnp.linspace(-1, 1, res))
    z = jnp.sqrt(1 - x**2 - y**2)
    y = y + 0.0 * z  # propagate nans
    x = jnp.ravel(x)[None, :]
    y = jnp.ravel(y)[None, :]
    z = jnp.ravel(z)[None, :]
    lat = 0.5 * jnp.pi - jnp.arccos(y)
    lon = jnp.arctan2(x, z)
    return (lat, lon), (x, y, z)


def left_project(deg, M, theta, inc, obl):
    # Note that here we are using the fact that R . M = (M^T . R^T)^T
    MT = jnp.transpose(M)

    # Rotate to the polar frame
    MT = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, -0.5 * jnp.pi)(MT)
    MT = dot_rotation_matrix(deg, None, None, 1.0, -theta)(MT)

    # Rotate to the sky frame
    MT = dot_rotation_matrix(deg, 1.0, 0.0, 0.0, 0.5 * jnp.pi)(MT)
    MT = dot_rotation_matrix(deg, None, None, 1.0, -obl)(MT)
    MT = dot_rotation_matrix(
        deg, -jnp.cos(obl), -jnp.sin(obl), 0.0, 0.5 * jnp.pi - inc
    )(MT)

    return MT


def render(deg, res, theta, inc, obl, y):
    _, xyz = ortho_grid(res)
    pT = poly_basis(deg)(*xyz)
    Ry = left_project(deg, y, theta, inc, obl)
    A1Ry = A1(deg) @ Ry
    return jnp.reshape(pT @ A1Ry, (res, res))
