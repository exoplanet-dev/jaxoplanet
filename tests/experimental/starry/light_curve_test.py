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
    def extract_starry_map_kwargs(jaxoplanet_map):
        if jaxoplanet_map is None:
            # starry's simplest case: a limb darkened map
            map_kwargs = dict(ydeg=0, rv=False, reflected=False)
        else:
            map_kwargs = dict(
                ydeg=jaxoplanet_map.ydeg,
                udeg=jaxoplanet_map.udeg,
                inc=np.rad2deg(jaxoplanet_map.inc),
                obl=np.rad2deg(jaxoplanet_map.obl),
                amp=jaxoplanet_map.amplitude,
            )
        return map_kwargs

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
    central_map_kwargs = extract_starry_map_kwargs(central_map)
    pri = starry.Primary(
        starry.Map(**central_map_kwargs),
        r=central.radius.magnitude,
        m=central.mass.magnitude,
        prot=0 if central_map is None else central_map.period,
    )
    if central_map and central_map.u:
        pri.map[1:] = central_map.u
    if central_map and central_map.deg > 0:
        pri.map[1:, :] = np.asarray(central_map.y.todense())[1:]

    body_map_kwargs = extract_starry_map_kwargs(body_map)
    sec = starry.Secondary(
        starry.Map(**body_map_kwargs),
        r=body.radius.magnitude,
        m=body.mass.magnitude,
        porb=body.period.magnitude,
        prot=0 if body_map is None else body_map.period,
    )
    if body_map and body_map.u:
        sec.map[1:] = body_map.u
    if body_map and body_map.deg > 0:
        sec.map[1:, :] = np.asarray(body_map.y.todense())[1:]

    starry_system = starry.System(pri, sec)
    starry_flux = starry_system.flux(time, total=False)

    # compare summed jaxponet fluxes with the corresponding starry fluxes
    if body_map is None:
        assert_allclose(jax.numpy.squeeze(jaxoplanet_flux), starry_flux[0])
    elif central_map is None:
        assert_allclose(jax.numpy.squeeze(jaxoplanet_flux), starry_flux[1])
    else:
        assert_allclose(jax.numpy.squeeze(jaxoplanet_flux), np.sum(starry_flux, axis=0))

    # compare individual jaxpolanet fluxes with the corresponding starry fluxes
    # for jaxoplanet_flux_item, starry_flux_item in zip(jaxoplanet_flux, starry_flux):
    #     if jaxoplanet_flux_item is not None:
    #         assert_allclose(jax.numpy.squeeze(jaxoplanet_flux_item), starry_flux_item)
