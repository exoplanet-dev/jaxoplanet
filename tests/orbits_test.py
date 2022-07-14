# mypy: ignore-errors

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from exo4jax.orbits import KeplerianBody, KeplerianCentral, KeplerianOrbit

assert_allclose = partial(
    np.testing.assert_allclose,
    atol=2e-5 if jax.config.jax_enable_x64 else 2e-4,
)


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
    assert_allclose(
        r_batman[m],
        r[m],
        atol=1e-5 if jax.config.jax_enable_x64 else 1e-2,
    )

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

    orbit = KeplerianOrbit.init(
        central=KeplerianCentral.init(mass=m_star, radius=1.0),
        time_transit=np.array([0.5, 17.4]),
        period=np.array([100.0, 37.3]),
        eccentricity=np.array([0.1, 0.8]),
        omega_peri=np.array([0.5, 1.3]),
        asc_node=np.array([0.0, 1.0]),
        inclination=np.array([0.25 * np.pi, 0.3 * np.pi]),
        mass=m_planet,
    )

    coords = np.asarray(orbit.position(t))
    central_coords = np.asarray(orbit.central_position(t))
    com = np.sum(
        (m_planet[..., None] * coords + m_star * central_coords)
        / (m_star + m_planet)[..., None],
        axis=0,
    )
    assert_allclose(com, 0.0)
