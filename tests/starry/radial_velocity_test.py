import jax
import numpy as np
import pytest

from jaxoplanet.starry import Surface, Ylm
from jaxoplanet.starry.doppler import radial_velocity, surface_radial_velocity
from jaxoplanet.starry.orbit import Body, Central, SurfaceSystem
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", (2, 6))
@pytest.mark.parametrize("u", ((), (0.2, 0.6)))
@pytest.mark.parametrize("inc", (np.pi / 2, 0.5))
def test_rv(deg, u, inc):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    starry.config.quiet = True

    map = starry.Map(ydeg=deg, udeg=len(u), rv=True, inc=np.rad2deg(inc))
    map.velocity_unit = "Rsun/day"
    map.veq = 1e2
    map.add_spot(-0.015, sigma=0.03, lat=0, lon=0)
    for i, ui in enumerate(u):
        map[1 + i] = ui

    theta = np.linspace(-180, 180, 100)
    # expected = map.rv(theta=theta) * map.velocity_unit.to("Rsun/day").value
    expected = map.rv(theta=theta)

    def veq2period(map, radius=1):
        veq = map.veq * map.velocity_unit
        return 2 * np.pi * radius / veq.to("Rsun/day").value

    surface = Surface(y=Ylm.from_dense(map.y), u=u, period=veq2period(map), inc=inc)
    calc = jax.vmap(lambda theta: surface_radial_velocity(surface, theta=theta))(
        np.deg2rad(theta)
    )

    assert_allclose(calc, expected)


@pytest.mark.parametrize(
    "params",
    (
        (0.8, 0.1, 1e4, 0.1),
        (0.0, 0.01, 1e3, 0.01),
    ),
)
def test_system_rv(params):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False
    starry.config.quiet = True

    time = np.linspace(-0.1, 0.1, 100)

    obl, r, veq, m = params
    # starry
    A = starry.Primary(
        starry.Map(udeg=2, rv=True, veq=veq, obl=np.rad2deg(obl)), r=1.0, m=1.0
    )
    A.map[1] = 0.5
    A.map[2] = 0.25
    b = starry.Secondary(starry.Map(ydeg=2, rv=True), r=r, porb=1.0, m=m)
    sys = starry.System(A, b)
    # expected = sys.rv(time) * map.velocity_unit.to("Rsun/day").value
    expected = sys.rv(time)

    # jaxoplanet
    def veq2period(map, radius=1):
        veq = map.veq * map.velocity_unit
        return 2 * np.pi * radius / veq.to("Rsun/day").value

    star_surface = Surface(
        u=A.map[1::], obl=np.deg2rad(A.map.obl), period=veq2period(A.map)
    )
    body_surface = Surface(period=veq2period(b.map))
    system = SurfaceSystem(Central(), star_surface).add_body(
        Body(period=b.porb, mass=b.m, radius=b.r), body_surface
    )
    calc = np.sum(radial_velocity(system)(time), 1)
    assert_allclose(calc, expected)
