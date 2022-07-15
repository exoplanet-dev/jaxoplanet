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
    assert_allclose(r_batman[m], r[m])

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


def test_velocity():
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit = KeplerianOrbit.init(
        central=KeplerianCentral.init(mass=m_star, radius=1.0),
        time_transit=0.5,
        period=100.0,
        eccentricity=0.1,
        omega_peri=0.5,
        asc_node=1.0,
        inclination=0.25 * np.pi,
        mass=m_planet,
    )

    computed = orbit.central_velocity(t)
    for n in range(3):
        expected = jax.vmap(
            jax.grad(lambda t: jnp.sum(orbit.central_position(t)[n]))
        )(t)
        assert_allclose(jnp.sum(computed[n], axis=0), expected)

    computed = orbit.velocity(t)
    for n in range(3):
        for i in range(len(orbit)):
            expected = jax.vmap(jax.grad(lambda t: orbit.position(t)[n][i]))(t)
            assert_allclose(computed[n][i], expected)

    computed = orbit.relative_velocity(t)
    for n in range(3):
        for i in range(len(orbit)):
            expected = jax.vmap(
                jax.grad(lambda t: orbit.relative_position(t)[n][i])
            )(t)
            assert_allclose(computed[n][i], expected)


def test_radial_velocity():
    t = np.linspace(0, 100, 1000)
    m_planet = 0.1
    m_star = 1.3
    orbit = KeplerianOrbit.init(
        central=KeplerianCentral.init(mass=m_star, radius=1.0),
        time_transit=0.5,
        period=100.0,
        eccentricity=0.1,
        omega_peri=0.5,
        asc_node=1.0,
        inclination=0.25 * np.pi,
        mass=m_planet,
    )

    computed = orbit.radial_velocity(t)
    expected = orbit.radial_velocity(
        t,
        semiamplitude=orbit.bodies._baseline_rv_semiamplitude
        * orbit.bodies.mass
        * orbit.bodies.sin_inclination,
    )
    assert_allclose(expected, computed)


def test_small_star():
    _rsky = pytest.importorskip("batman._rsky")

    m_star = 0.151
    r_star = 0.189
    period = 0.4626413
    t0 = 0.2
    b = 0.5
    ecc = 0.1
    omega = 0.1
    t = np.linspace(0, period, 500)

    orbit = KeplerianOrbit.init(
        central=KeplerianCentral.init(radius=r_star, mass=m_star),
        period=period,
        time_transit=t0,
        impact_param=b,
        eccentricity=ecc,
        omega_peri=omega,
    )
    a = orbit.semimajor
    incl = jnp.arctan2(
        orbit.bodies.sin_inclination, orbit.bodies.cos_inclination
    )

    r_batman = _rsky._rsky(t, t0, period, a, incl, ecc, omega, 1, 1)
    m = r_batman < 100.0
    assert m.sum() > 0

    x, y, _ = orbit.relative_position(t)
    r = np.sqrt(x**2 + y**2)

    # Make sure that the in-transit impact parameter matches batman
    assert_allclose(r_batman[m], r[m])


def test_impact():
    m_star = 0.151
    r_star = 0.189
    period = 0.4626413
    t0 = 0.2
    b = 0.5
    ecc = 0.8
    omega = 0.1

    orbit = KeplerianOrbit.init(
        central=KeplerianCentral.init(radius=r_star, mass=m_star),
        period=period,
        time_transit=t0,
        impact_param=b,
        eccentricity=ecc,
        omega_peri=omega,
    )
    x, y, z = orbit.relative_position(t0)
    assert_allclose((jnp.sqrt(x**2 + y**2) / r_star), b)
    assert jnp.all(z > 0)
