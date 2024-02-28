import jax
import numpy as np
import pytest

from jaxoplanet.experimental.starry import Map, Ylm
from jaxoplanet.experimental.starry.light_curves import light_curve, map_light_curve
from jaxoplanet.experimental.starry.orbit import SurfaceMapSystem
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
            ),
            "surface_map": Map(
                y=Ylm.from_dense(
                    [1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03, 0.04, 0.4, 0.2, 0.1]
                ),
                inc=0.9,
                obl=0.3,
                period=1.2,
                u=(0.1, 0.1),
            ),
            "body": {
                "radius": 0.5,
                "mass": 0.1,
                "period": 1.5,
                "surface_map": Map(
                    y=Ylm.from_dense([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03]),
                    inc=-0.3,
                    period=0.8,
                    u=(0.2, 0.3),
                ),
            },
        },
        {
            "central": keplerian.Central(
                mass=1.3,
                radius=1.1,
            ),
            "surface_map": Map(
                y=Ylm.from_dense(np.hstack([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03])),
                period=1.2,
                u=(0.1, 0.1),
            ),
            "body": {
                "radius": 0.5,
                "mass": 0.1,
                "period": 1.5,
            },
        },
        {
            "central": keplerian.Central(
                mass=0.3,
                radius=0.1,
            ),
            "body": {
                "radius": 1.5,
                "mass": 1.1,
                "period": 1.5,
                "surface_map": Map(
                    y=Ylm.from_dense([1, 0.005, 0.05, 0.09, 0.0, 0.1, 0.03]),
                    period=1.2,
                    u=(0.1, 0.1),
                ),
            },
        },
    ]
)
def keplerian_system(request):
    return SurfaceMapSystem(
        request.param["central"], request.param.get("surface_map", None)
    ).add_body(**request.param["body"])


def test_compare_starry_system(keplerian_system):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # jaxoplanet system
    central = keplerian_system.central
    central_map = keplerian_system.surface_map
    body = keplerian_system.bodies[0]
    body_map = keplerian_system.surface_maps[0]

    time = np.linspace(-1.5, 1.0, 300)
    jaxoplanet_flux = light_curve(keplerian_system)(time)

    # starry system
    pri = starry.Primary(
        starry.Map(
            ydeg=central_map.ydeg,
            udeg=central_map.udeg,
            inc=np.rad2deg(central_map.inc),
            obl=np.rad2deg(central_map.obl),
            amp=central_map.amplitude,
        ),
        r=central.radius.magnitude,
        m=central.mass.magnitude,
        prot=central_map.period,
    )
    if central_map.u:
        pri.map[1:] = central_map.u
    if central_map.deg > 0:
        pri.map[1:, :] = np.asarray(central_map.y.todense())[1:]

    sec = starry.Secondary(
        starry.Map(
            ydeg=body_map.ydeg,
            udeg=body_map.udeg,
            inc=np.rad2deg(body_map.inc),
            obl=np.rad2deg(body_map.obl),
            amp=body_map.amplitude,
        ),
        r=body.radius.magnitude,
        m=body.mass.magnitude,
        porb=body.period.magnitude,
        prot=body_map.period,
    )
    if body_map.u:
        sec.map[1:] = body_map.u

    if body_map.deg > 0:
        sec.map[1:, :] = np.asarray(body_map.y.todense())[1:]

    starry_system = starry.System(pri, sec)
    starry_flux = starry_system.flux(time, total=False)
    assert_allclose(jaxoplanet_flux, starry_flux)
