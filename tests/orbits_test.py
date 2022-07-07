# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exo4jax.orbits import KeplerianBody, KeplerianCentral


def test_sky_coords():
    _rsky = pytest.importorskip("batman._rsky")

    t = np.linspace(-100, 100, 1000)

    t0, period, a, e, omega, incl = (
        x.flatten()
        for x in np.meshgrid(
            np.linspace(-5.0, 5.0, 2),
            np.exp(np.linspace(np.log(5.0), np.log(50.0), 3)),
            np.linspace(50.0, 100.0, 2),
            np.linspace(0.0, 0.9, 5),
            np.linspace(-np.pi, np.pi, 3),
            np.arccos(np.linspace(0, 1, 5)[:-1]),
        )
    )
    r_batman = np.empty((len(t), len(t0)))

    for i in range(len(t0)):
        r_batman[:, i] = _rsky._rsky(
            t, t0[i], period[i], a[i], incl[i], e[i], omega[i], 1, 1
        )
    m = r_batman < 100.0
    assert m.sum() > 0

    def get_r(**kwargs):
        orbit = KeplerianBody.init(**kwargs)
        x, y, z = orbit.relative_position(t)
        return jnp.sqrt(x**2 + y**2), z

    r, z = jax.vmap(get_r, out_axes=1)(
        period=period,
        semimajor=a,
        time_transit=t0,
        eccentricity=e,
        omega_peri=omega,
        inclination=incl,
    )

    # Make sure that the in-transit impact parameter matches batman
    assert np.allclose(r_batman[m], r[m], atol=2e-5)

    # In-transit should correspond to positive z in our parameterization
    assert np.all(z[m] > 0)

    # Therefore, when batman doesn't see a transit we shouldn't be transiting
    no_transit = z[~m] < 0
    no_transit |= r[~m] > 2
    assert np.all(no_transit)


def test_center_of_mass():
    t = np.linspace(0, 100, 1000)
    m_planet = np.array([0.5, 0.1])
    m_star = 1.45

    orbit = KeplerianBody.init(
        central=KeplerianCentral.init(mass=m_star, radius=1.0),
        t0=np.array([0.5, 17.4]),
        period=np.array([100.0, 37.3]),
        ecc=np.array([0.1, 0.8]),
        omega=np.array([0.5, 1.3]),
        Omega=np.array([0.0, 1.0]),
        incl=np.array([0.25 * np.pi, 0.3 * np.pi]),
        m_planet=m_planet,
    )

    planet_coords = theano.function([], orbit.get_planet_position(t))()
    star_coords = theano.function([], orbit.get_star_position(t))()

    com = np.sum(
        (
            m_planet[None, :] * np.array(planet_coords)
            + m_star * np.array(star_coords)
        )
        / (m_star + m_planet)[None, :],
        axis=0,
    )
    assert np.allclose(com, 0.0)
