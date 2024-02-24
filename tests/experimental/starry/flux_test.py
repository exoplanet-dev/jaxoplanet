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


@pytest.fixture(
    params=[
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
                map=Map(
                    y=Ylm.from_dense(np.hstack([1.0, np.random.rand(10) * 0.1])),
                    inc=0.9,
                    obl=0.3,
                    period=1.2,
                    u=[0.1, 0.1],
                ),
            ),
            "body": {
                "radius": 0.5,
                "mass": 0.1,
                "period": 1.5,
                "map": Map(
                    y=Ylm.from_dense(np.hstack([1.0, np.random.rand(5) * 0.1])),
                    inc=-0.3,
                    period=0.8,
                    u=[0.2, 0.3],
                ),
            },
        },
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
                map=Map(y=1.0),
            ),
            "body": {
                "radius": 0.5,
                "mass": 0.1,
                "period": 1.5,
                "map": Map(y=0.0),
            },
        },
        {
            "central": keplerian.Central(
                mass=0.3,
                radius=0.1,
                map=Map(y=1.0),
            ),
            "body": {
                "radius": 1.5,
                "mass": 1.1,
                "period": 1.5,
                "map": Map(y=0.0),
            },
        },
    ]
)
def keplerian_system(request):
    return keplerian.System(request.param["central"]).add_body(**request.param["body"])


def test_compare_starry_system(keplerian_system):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # jaxoplanet system
    central = keplerian_system.central
    body = keplerian_system.bodies[0]

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
            amp=central.map.y[(0, 0)],
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
            amp=body.map.y[(0, 0)],
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


@pytest.mark.parametrize("deg", [2, 5, 10])
def test_compare_starry_rot(deg):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    inc = np.pi / 2
    np.random.seed(deg)
    y = Ylm.from_dense(np.random.randn((deg + 1) ** 2))
    map = Map(y=y, inc=inc)

    # phase
    theta = np.linspace(0, 2 * np.pi, 200)

    # starry
    ms = starry.Map(ydeg=deg, inc=np.rad2deg(inc))
    ms[:, :] = y.todense()
    expected = ms.flux(theta=np.rad2deg(theta))
    calc = map.flux(theta=theta)

    assert_allclose(calc, expected)
