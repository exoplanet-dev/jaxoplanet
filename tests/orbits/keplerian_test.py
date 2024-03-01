import jax
import jax.experimental
import jax.numpy as jnp
import jpu.numpy as jnpu
import numpy as np
import pytest

from jaxoplanet.orbits.keplerian import Body, Central, System
from jaxoplanet.test_utils import (
    assert_allclose,
    assert_quantity_allclose,
    assert_quantity_pytree_allclose,
)
from jaxoplanet.units import unit_registry as ureg


@pytest.fixture(
    params=[
        {
            "central": Central(mass=1.3, radius=1.1),
            "bodies": [
                Body(
                    mass=0.1,
                    time_transit=0.1,
                    period=12.5,
                    inclination=0.3,
                )
            ],
        },
        {
            "central": Central(mass=1.3, radius=1.1),
            "bodies": [
                Body(
                    mass=0.1,
                    time_transit=0.1,
                    period=12.5,
                    inclination=0.3,
                    eccentricity=0.3,
                    omega_peri=-1.5,
                    asc_node=0.3,
                )
            ],
        },
    ]
)
def system(request):
    return System(**request.param)


@pytest.fixture
def time():
    return jnp.linspace(-50.0, 50.0, 500) * ureg.day


def test_keplerian_central_density():
    star = Central()
    assert_quantity_allclose(
        star.density, 1.4 * ureg.g / ureg.cm**3, atol=0.01, convert=True
    )


def test_keplerian_central_from_orbit():
    sun = Central.from_orbital_properties(
        period=1.0 * ureg.yr,
        semimajor=1.0 * ureg.au,
        radius=1.0 * ureg.R_sun,
        body_mass=1.0 * ureg.M_earth,
    )
    # add large tolerance to account for lack of precision in ureg.yr
    assert_quantity_allclose(sun.mass, 1.0 * ureg.M_sun, atol=5e-5)


def test_keplerian_body_keplers_law():
    orbit = System().add_body(semimajor=1.0 * ureg.au)
    assert_quantity_allclose(
        orbit.bodies[0].period, 1.0 * ureg.year, atol=0.01, convert=True
    )

    orbit = System().add_body(period=1.0 * ureg.year)
    assert_quantity_allclose(
        orbit.bodies[0].semimajor, 1.0 * ureg.au, atol=0.01, convert=True
    )


@pytest.mark.parametrize("prefix", ["", "central_", "relative_"])
def test_keplerian_body_velocity(time, system, prefix):
    body = system.bodies[0]
    v = getattr(body, f"{prefix}velocity")(time)
    for i, v_ in enumerate(v):
        pos_func = getattr(body, f"{prefix}position")
        assert_allclose(
            v_.magnitude,
            jax.vmap(jax.grad(lambda t: pos_func(t)[i].magnitude))(time).magnitude,
        )


def test_keplerian_body_radial_velocity(time, system):
    body = system.bodies[0]
    computed = body.radial_velocity(time)
    assert computed.units == ureg.R_sun / ureg.d
    computed.to(ureg.m / ureg.s)

    computed = body.radial_velocity(time, semiamplitude=1.0)
    assert not hasattr(computed, "_magnitude")
    assert not hasattr(computed, "_units")


def test_keplerian_body_impact_parameter(system):
    body = system.bodies[0]
    x, y, z = body.relative_position(body.time_transit)
    assert_quantity_allclose(
        (jnpu.sqrt(x**2 + y**2) / system.central.radius),
        body.impact_param,
    )
    assert jnpu.all(z > 0)


def test_keplerian_body_coordinates_match_batman(time, system):
    _rsky = pytest.importorskip("batman._rsky")
    body = system.bodies[0]
    with jax.experimental.enable_x64(True):
        r_batman = _rsky._rsky(
            np.array(time.magnitude, dtype=np.float64),
            float(body.time_transit.magnitude),
            float(body.period.magnitude),
            float(body.semimajor.magnitude),
            float(body.inclination.magnitude),
            (float(body.eccentricity.magnitude) if body.eccentricity else 0.0),
            (float(body.omega_peri.magnitude) if body.omega_peri else 0.0),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, z = body.relative_position(time)
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
        body = (
            System(
                Central(radius=0.189, mass=0.151),
            )
            .add_body(
                period=0.4626413,
                time_transit=0.2,
                impact_param=0.5,
                eccentricity=0.1,
                omega_peri=0.1,
            )
            .bodies[0]
        )

        r_batman = _rsky._rsky(
            np.array(time.magnitude, dtype=np.float64),
            float(body.time_transit.magnitude),
            float(body.period.magnitude),
            float(body.semimajor.magnitude),
            float(body.inclination.magnitude),
            (float(body.eccentricity.magnitude) if body.eccentricity else 0.0),
            (float(body.omega_peri.magnitude) if body.omega_peri else 0.0),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, _ = body.relative_position(time)
        r = jnpu.sqrt(x**2 + y**2)
        assert_allclose(r_batman[m], r[m].magnitude)


def test_keplerian_system_stack_construction():
    sys = System().add_body(period=0.1).add_body(period=0.2)
    assert sys._body_stack.stack is not None
    assert_quantity_allclose(
        sys._body_stack.stack.period, jnp.array([0.1, 0.2]) * ureg.day
    )

    sys = (
        System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.1, omega_peri=-1.5)
    )
    assert sys._body_stack.stack is None


def test_keplerian_system_radial_velocity():
    sys1 = System().add_body(period=0.1).add_body(period=0.2)
    sys2 = (
        System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.0, omega_peri=0.0)
    )
    assert sys1._body_stack.stack is not None
    assert sys2._body_stack.stack is None
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
    sys1 = System().add_body(period=0.1).add_body(period=0.2)
    sys2 = (
        System()
        .add_body(period=0.1)
        .add_body(period=0.2, eccentricity=0.0, omega_peri=0.0)
    )
    for t in [0.0, jnp.linspace(0, 1, 5)]:
        x1, y1, z1 = sys1.position(t)
        x2, y2, z2 = sys2.position(t)
        assert_quantity_allclose(x1, x2)
        assert_quantity_allclose(y1, y2)
        assert_quantity_allclose(z1, z2)


def body_vmap_func1(body, x):
    return body.radius * x


def body_vmap_func2(body, x):
    return body.radius * x["x"]


def body_vmap_func3(body, x):
    return {"y": body.radius * x}


def body_vmap_func4(body, x):
    del body
    return {"y": x}


@pytest.mark.parametrize(
    "func, in_axes, out_axes, args",
    [
        (body_vmap_func1, None, 0, (jnp.linspace(0.0, 2.0, 3),)),
        (body_vmap_func1, 0, 0, (jnp.linspace(0.0, 2.0, 6).reshape(2, 3),)),
        (body_vmap_func1, (0,), 0, (jnp.linspace(0.0, 2.0, 6).reshape(2, 3),)),
        (body_vmap_func1, (1,), 0, (jnp.linspace(0.0, 2.0, 6).reshape(3, 2),)),
        (body_vmap_func2, None, 0, ({"x": jnp.linspace(0.0, 2.0, 3)},)),
        (body_vmap_func2, {"x": None}, 0, ({"x": jnp.linspace(0.0, 2.0, 3)},)),
        (
            body_vmap_func2,
            {"x": 0},
            0,
            ({"x": jnp.linspace(0.0, 2.0, 6).reshape(2, 3)},),
        ),
        (body_vmap_func3, None, 0, (jnp.linspace(0.0, 2.0, 3),)),
        (body_vmap_func3, None, 1, (jnp.linspace(0.0, 2.0, 3),)),
        (body_vmap_func3, None, {"y": 1}, (jnp.linspace(0.0, 2.0, 3),)),
        (body_vmap_func4, None, None, (jnp.linspace(0.0, 2.0, 3),)),
    ],
)
def test_body_vmap(func, in_axes, out_axes, args):
    central = Central()
    vmap_sys = (
        System(central)
        .add_body(radius=0.5, period=1.0)
        .add_body(radius=0.8, period=1.0)
    )
    no_vmap_sys = (
        System(central)
        .add_body(radius=0.5, period=1.0, eccentricity=0.1, omega_peri=0.1)
        .add_body(radius=0.8, period=1.0)
    )

    result1 = vmap_sys.body_vmap(func, in_axes=in_axes, out_axes=out_axes)(*args)
    result2 = no_vmap_sys.body_vmap(func, in_axes=in_axes, out_axes=out_axes)(*args)
    assert_quantity_pytree_allclose(result1, result2)
