import jax.numpy as jnp

from jaxoplanet.light_curves import LimbDarkLightCurve
from jaxoplanet.orbits.keplerian import Central, System


def test_light_curve():
    # Fiducial planet parameters:
    params = {
        "period": 300.456,
        "radius": 0.1,
        "u": jnp.array([0.3, 0.2]),
        "bo": 0.8,
    }
    # The light curve calculation requires an orbit
    orbit = System().add_body(
        period=params["period"],
        radius=params["radius"],
        impact_param=params["bo"],
    )

    # Compute a limb-darkened light curve using jaxoplanet
    t = jnp.linspace(-0.3, 0.3, 1000)

    lc = LimbDarkLightCurve(params["u"]).light_curve(orbit, t=t)
    assert lc.shape == (1,) + t.shape


def test_multiplanetary():
    star = Central(mass=5, radius=2)
    system = System(star).add_body(radius=0.05, period=1).add_body(radius=0.2, period=2)

    t = jnp.linspace(-5, 5, 500)
    lc = LimbDarkLightCurve().light_curve(system, t)
    lc_texp_scalar = LimbDarkLightCurve().light_curve(system, t, texp=10)
    lc_texp_array = LimbDarkLightCurve().light_curve(
        system, t, texp=jnp.broadcast_to(10, t.size)
    )
    assert lc.shape == (system.shape[0], t.size)
    assert lc_texp_scalar.shape == (system.shape[0], t.size)
    assert lc_texp_array.shape == (system.shape[0], t.size)
