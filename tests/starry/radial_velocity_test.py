import jax
import numpy as np
import pytest

from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.doppler import surface_radial_velocity
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", (2, 6))
@pytest.mark.parametrize("u", ((), (0.2, 0.6)))
@pytest.mark.parametrize("inc", (np.pi / 2, 0.5))
def test_rv(deg, u, inc):

    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    starry.config.quiet = True

    map = starry.Map(ydeg=deg, udeg=len(u), rv=True, inc=np.rad2deg(inc))
    map.veq = 1e2
    map.add_spot(-0.015, sigma=0.03, lat=0, lon=0)
    for i, ui in enumerate(u):
        map[1 + i] = ui

    theta = np.linspace(-180, 180, 100)
    expected = map.rv(theta=theta)

    def veq2period(map, radius=1):
        veq = map.veq * map.velocity_unit
        return 2 * np.pi * radius / veq.to("Rsun/day").value

    surface = Surface(y=Ylm.from_dense(map.y), u=u, period=veq2period(map), inc=inc)
    calc = jax.vmap(lambda theta: surface_radial_velocity(surface, theta=theta))(
        np.deg2rad(theta)
    )

    assert_allclose(calc, expected)
