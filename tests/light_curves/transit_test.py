import numpy as np

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.keplerian import Central, System
from jaxoplanet.orbits.transit import TransitOrbit
from jaxoplanet.test_utils import assert_allclose
from jaxoplanet.units import unit_registry as ureg


def test_radius_ratio_depth():

    period = 1.0 * ureg.day
    central_radius = 0.5 * ureg.R_sun
    body_radius = 0.1 * ureg.R_sun
    radius_ratio = body_radius / central_radius

    expected_depth = radius_ratio**2

    transit_orbit = TransitOrbit(
        period=period,
        radius_ratio=radius_ratio,
        duration=0.1 * ureg.day,
    )

    system_orbit = System(
        Central(
            radius=central_radius,
            mass=0.5 * ureg.M_sun,
        )
    ).add_body(period=period, radius=body_radius)

    time = np.linspace(-0.1, 0.1, 10)

    transit_flux = limb_dark_light_curve(transit_orbit, (0,))(time)
    system_flux = limb_dark_light_curve(system_orbit, (0,))(time)

    assert_allclose(transit_flux.min(), -expected_depth.magnitude)
    assert_allclose(transit_flux.min(), system_flux.min())
