import jax
import numpy as np
import pytest

from jaxoplanet.experimental.starry import Map, Ylm
from jaxoplanet.experimental.starry.light_curves import light_curve, map_light_curve
from jaxoplanet.orbits import keplerian
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", [2, 5, 10])
@pytest.mark.parametrize("u", [[], [0.1], [0.2, 0.1]])
def test_compare_starry(deg, u):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    inc = np.pi / 2
    np.random.seed(deg)
    y = Ylm.from_dense(np.random.randn((deg + 1) ** 2))
    map = Map(y=y, u=u, inc=inc)

    # occultor
    yo = np.linspace(-3, 3, 1000)
    ro = 0.01
    zo = 2.0
    xo = 0.0

    # starry
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    ms[:, :] = y.todense()
    if len(u) > 0:
        ms[1:] = u
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jax.vmap(map_light_curve, in_axes=(None, None, None, 0, None, None))(
        map, ro, xo, yo, zo, 0.0
    )

    assert_allclose(calc, expected)


def test_compare_starry_system():
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # jaxoplanet system
    y = Ylm.from_dense(np.hstack([1.0, np.random.rand(10) * 0.1]))
    central_map = Map(y=y, inc=0.9, obl=0.3, period=1.2, u=[0.1, 0.1])
    central = keplerian.Central(radius=1.0, mass=1.0, map=central_map)

    y = Ylm.from_dense(np.hstack([1.0, np.random.rand(10) * 1e-1]))
    body_map = Map(y=y, inc=2.5, obl=0.3, period=-0.8, u=[0.2, 0.5])
    body = keplerian.Body(radius=0.5, mass=1.0, period=1.0, map=body_map)

    system = keplerian.System(central, bodies=[body])

    time = np.linspace(-1.5, 1.0, 300)
    jaxoplanet_flux = light_curve(system, time)

    # starry system
    pri = starry.Primary(
        starry.Map(
            ydeg=central.map.ydeg,
            udeg=central.map.udeg,
            inc=np.rad2deg(central.map.inc),
            obl=np.rad2deg(central.map.obl),
        ),
        r=central.radius.magnitude,
        m=central.mass.magnitude,
        prot=central.map.period,
    )
    if central.map.u:
        pri.map[1:] = central.map.u
        pri.map[1:, :] = central.map.y.todense().__array__()[1:]

    sec = starry.Secondary(
        starry.Map(
            ydeg=body.map.ydeg,
            udeg=body.map.udeg,
            inc=np.rad2deg(body.map.inc),
            obl=np.rad2deg(body.map.obl),
        ),
        r=body.radius.magnitude,
        m=body.mass.magnitude,
        porb=body.period.magnitude,
        prot=body.map.period,
    )
    if body.map.u:
        sec.map[1:] = body.map.u
        sec.map[1:, :] = body.map.y.todense().__array__()[1:]

    starry_system = starry.System(pri, sec)
    starry_flux = starry_system.flux(time, total=False)
    assert_allclose(jaxoplanet_flux, starry_flux)
