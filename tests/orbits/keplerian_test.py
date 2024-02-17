import jax
import jax.experimental
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as np
import pytest

from jaxoplanet.orbits import keplerian
from jaxoplanet.test_utils import assert_allclose, assert_quantity_allclose
from jaxoplanet.units import unit_registry as ureg


@pytest.fixture(
    params=[
        {
            "central": keplerian.Central(mass=1.3, radius=1.1),
            "mass": 0.1,
            "time_transit": 0.1,
            "period": 12.5,
            "inclination": 0.3,
        },
        {
            "central": keplerian.Central(mass=1.3, radius=1.1),
            "mass": 0.1,
            "time_transit": 0.1,
            "period": 12.5,
            "inclination": 0.3,
            "eccentricity": 0.3,
            "omega_peri": -1.5,
            "asc_node": 0.3,
        },
    ]
)
def keplerian_body(request):
    return keplerian.Body(**request.param)


@pytest.fixture
def time():
    return jnp.linspace(-50.0, 50.0, 500) * ureg.day


def test_keplerian_central_density():
    star = keplerian.Central()
    assert_quantity_allclose(
        star.density, 1.4 * ureg.g / ureg.cm**3, atol=0.01, convert=True
    )


def test_keplerian_body_keplers_law():
    orbit = keplerian.Body(semimajor=1.0 * ureg.au)
    assert_quantity_allclose(orbit.period, 1.0 * ureg.year, atol=0.01, convert=True)

    orbit = keplerian.Body(period=1.0 * ureg.year)
    assert_quantity_allclose(orbit.semimajor, 1.0 * ureg.au, atol=0.01, convert=True)


@pytest.mark.parametrize("prefix", ["", "central_", "relative_"])
def test_keplerian_body_velocity(time, keplerian_body, prefix):
    v = getattr(keplerian_body, f"{prefix}velocity")(time)
    for i, v_ in enumerate(v):
        pos_func = getattr(keplerian_body, f"{prefix}position")
        assert_allclose(
            v_.magnitude,
            jax.vmap(jax.grad(lambda t: pos_func(t)[i].magnitude))(time).magnitude,
        )


def test_keplerian_body_radial_velocity(time, keplerian_body):
    computed = keplerian_body.radial_velocity(time)
    assert computed.units == ureg.R_sun / ureg.d
    computed.to(ureg.m / ureg.s)

    computed = keplerian_body.radial_velocity(time, semiamplitude=1.0)
    assert not hasattr(computed, "_magnitude")
    assert not hasattr(computed, "_units")


def test_keplerian_body_impact_parameter(keplerian_body):
    x, y, z = keplerian_body.relative_position(keplerian_body.time_transit)
    assert_quantity_allclose(
        (jnpu.sqrt(x**2 + y**2) / keplerian_body.central.radius),
        keplerian_body.impact_param,
    )
    assert jnpu.all(z > 0)


def test_keplerian_body_coordinates_match_batman(time, keplerian_body):
    _rsky = pytest.importorskip("batman._rsky")
    with jax.experimental.enable_x64(True):
        r_batman = _rsky._rsky(
            np.array(time.magnitude, dtype=np.float64),
            float(keplerian_body.time_transit.magnitude),
            float(keplerian_body.period.magnitude),
            float(keplerian_body.semimajor.magnitude),
            float(keplerian_body.inclination.magnitude),
            (
                float(keplerian_body.eccentricity.magnitude)
                if keplerian_body.eccentricity
                else 0.0
            ),
            (
                float(keplerian_body.omega_peri.magnitude)
                if keplerian_body.omega_peri
                else 0.0
            ),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, z = keplerian_body.relative_position(time)
        r = jnpu.sqrt(x**2 + y**2)

        # Make sure that the in-transit impact parameter matches batman
        assert_allclose(r_batman[m], r.magnitude[m])

        # In-transit should correspond to positive z in our parameterization
        assert np.all(z.magnitude[m] > 0)

        # Therefore, when batman doesn't see a transit we shouldn't be transiting
        no_transit = z.magnitude[~m] < 0
        no_transit |= r.magnitude[~m] > 2
        assert np.all(no_transit)


def test_keplerian_body_positions_small_star(time):
    _rsky = pytest.importorskip("batman._rsky")
    with jax.experimental.enable_x64(True):
        keplerian_body = keplerian.Body(
            central=keplerian.Central(radius=0.189, mass=0.151),
            period=0.4626413,
            time_transit=0.2,
            impact_param=0.5,
            eccentricity=0.1,
            omega_peri=0.1,
        )

        r_batman = _rsky._rsky(
            np.array(time.magnitude, dtype=np.float64),
            float(keplerian_body.time_transit.magnitude),
            float(keplerian_body.period.magnitude),
            float(keplerian_body.semimajor.magnitude),
            float(keplerian_body.inclination.magnitude),
            (
                float(keplerian_body.eccentricity.magnitude)
                if keplerian_body.eccentricity
                else 0.0
            ),
            (
                float(keplerian_body.omega_peri.magnitude)
                if keplerian_body.omega_peri
                else 0.0
            ),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, _ = keplerian_body.relative_position(time)
        r = jnpu.sqrt(x**2 + y**2)
        assert_allclose(r_batman[m], r[m].magnitude)


def test_keplerian_system_stack_construction():
    sys = keplerian.System().add_body(period=0.1).add_body(period=0.2)
    assert_quantity_allclose(
        sys._body_stack.stack.period, jnp.array([0.1, 0.2]) * ureg.day
    )

    sys = (
        keplerian.System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.1, omega_peri=-1.5)
    )
    assert sys._body_stack is None


def test_keplerian_system_radial_velocity():
    sys1 = keplerian.System().add_body(period=0.1).add_body(period=0.2)
    sys2 = (
        keplerian.System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.0, omega_peri=0.0)
    )
    assert sys1._body_stack is not None
    assert sys2._body_stack is None
    with pytest.raises(ValueError):
        sys1._body_stack.stack.radial_velocity(0.0)

    for t in [0.0, jnp.linspace(0, 1, 5)]:
        assert_quantity_allclose(
            sys1.radial_velocity(t),
            sys2.radial_velocity(t),
            # TODO(dfm): I'm not sure why we need to loosen the tolerance here,
            # but the ecc=0 model doesn't give the same results as the ecc=None
            # model otherwise.
            atol={jnp.float32: 5e-6, jnp.float64: 1e-12},
        )


def test_keplerian_system_position():
    sys1 = keplerian.System().add_body(period=0.1).add_body(period=0.2)
    sys2 = (
        keplerian.System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.0, omega_peri=0.0)
    )
    for t in [0.0, jnp.linspace(0, 1, 5)]:
        x1, y1, z1 = sys1.position(t)
        x2, y2, z2 = sys2.position(t)
        assert_quantity_allclose(x1, x2)
        assert_quantity_allclose(y1, y2)
        assert_quantity_allclose(z1, z2)


def test_body_vmap():
    # _body_stack is not None
    central = keplerian.Central()
    bodies = (
        keplerian.Body(radius=0.5, period=1.0),
        keplerian.Body(radius=0.8, period=1.0),
    )
    system_bs = keplerian.System(central, bodies=bodies)

    # _body_stack is None
    central = keplerian.Central()
    bodies = (
        keplerian.Body(radius=0.5, period=1.0, eccentricity=0.1, omega_peri=0.1),
        keplerian.Body(radius=0.8, period=1.0),
    )
    system_no_bs = keplerian.System(central, bodies=bodies)

    x = jnp.array([0.0, 1.0, 2.0])

    def func1(body, x):
        return body.radius * x

    # _body_stack is not None
    assert system_bs.body_vmap(func1, in_axes=None)(x).shape == (2, 3)
    assert system_bs.body_vmap(func1, in_axes=0)(jnp.array([0.0, 1.0])).shape == (2,)

    # _body_stack is None
    assert system_no_bs.body_vmap(func1, in_axes=None)(x).shape == (2, 3)
    assert system_no_bs.body_vmap(func1, in_axes=0)(jnp.array([0.0, 1.0])).shape == (2,)

    # both equal
    assert_quantity_allclose(
        system_no_bs.body_vmap(func1, in_axes=None)(x),
        system_bs.body_vmap(func1, in_axes=None)(x),
    )

    def func2(body, x, y):
        return body.radius * x * y

    # _body_stack is not None
    assert system_bs.body_vmap(func2, in_axes=(None, 0))(
        x, jnp.array([0.0, 1.0])
    ).shape == (2, 3)

    # _body_stack is None
    assert system_no_bs.body_vmap(func2, in_axes=(None, 0))(
        x, jnp.array([0.0, 1.0])
    ).shape == (2, 3)

    # both equal
    assert_quantity_allclose(
        system_no_bs.body_vmap(func2, in_axes=(None, 0))(x, jnp.array([0.0, 1.0])),
        system_bs.body_vmap(func2, in_axes=(None, 0))(x, jnp.array([0.0, 1.0])),
    )
