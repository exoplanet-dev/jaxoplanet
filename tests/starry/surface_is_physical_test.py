import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import pytest
from scipy.optimize import minimize

from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.surface_is_physical import surface_min_intensity
from jaxoplanet.starry.ylm import Ylm


def mollweide_grid(oversample, lmax):
    """Create an approximately uniform grid on the sphere using the Mollweide projection."""
    npts = oversample * lmax**2
    nlat = int(jnp.sqrt(npts))
    nlon = int(npts / nlat)
    lats = jnp.linspace(-jnp.pi / 2 + 1e-3, jnp.pi / 2 - 1e-3, nlat)
    lons = jnp.linspace(0, 2 * jnp.pi, nlon, endpoint=False)
    grid = [(lat, lon) for lat in lats for lon in lons]
    return grid


def scipy_surface_min_intensity(surface: Surface, oversample: int = 4, lmax: int = 4):
    """Find global minimum intensity on the surface."""
    grid = mollweide_grid(oversample, lmax)

    @jax.jit
    def objective(coord):
        lat, lon = coord
        return surface.intensity(lat, lon)

    min_val = jnp.inf
    min_coord = None

    for coord in grid:
        try:
            res = minimize(objective, np.array(coord), method="BFGS")
            if res.success and res.fun < min_val:
                min_val = res.fun
                min_coord = res.x
        except Exception as e:
            print(f"Minimization failed at {coord} with error: {e}")
            continue

    lat, lon = min_coord
    lat = np.arctan2(np.sin(lat), np.cos(lat))
    lon = lon % (2 * np.pi)
    return (lat, lon), min_val


# create a test surface with an arbitrary map
ylm_surface = np.random.rand(25)
ylm_surface[0] = 1.0  # ensure positive offset

surface = Surface(
    y=Ylm.from_dense(ylm_surface),
    inc=jnp.radians(90),
    obl=jnp.radians(0),
    period=1.0,
    amplitude=1.0,
    normalize=True,
)


@pytest.mark.parametrize(("surface"), [surface])
def test_surface_min_intensity(surface):
    """Test that the surface_min_intensity function returns the same result as the scipy version"""
    oversample = 4
    lmax = 4
    (jax_lat, jax_lon), jax_min = surface_min_intensity(surface, oversample, lmax)
    (scipy_lat, scipy_lon), scipy_min = scipy_surface_min_intensity(
        surface, oversample, lmax
    )
    assert (
        jnp.allclose(jax_lat, scipy_lat, rtol=1e-7)
        & jnp.allclose(jax_lon, scipy_lon, rtol=1e-7)
        & jnp.allclose(jax_min, scipy_min, rtol=1e-7)
    )
