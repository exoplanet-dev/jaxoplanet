import jax.numpy as jnp

from jaxoplanet.experimental.starry.basis import A1, poly_basis
from jaxoplanet.experimental.starry.rotation import left_project


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


def render(deg, res, theta, inc, obl, y):
    _, xyz = ortho_grid(res)
    pT = poly_basis(deg)(*xyz)
    Ry = left_project(deg, inc, obl, theta, 0.0, y)
    A1Ry = A1(deg) @ Ry
    return jnp.reshape(pT @ A1Ry, (res, res))
