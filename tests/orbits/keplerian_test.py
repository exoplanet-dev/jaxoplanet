import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet import constants
from jaxoplanet.orbits.keplerian import Body, Central, System
from jaxoplanet.test_utils import assert_allclose, assert_pytree_allclose


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
    return jnp.linspace(-50.0, 50.0, 500)


def test_keplerian_central_density():
    star = Central()

    # ((1 * u.Msun / u.Rsun**3).to(u.g / u.cm**3)).value
    conversion = 5.905271918964842

    assert_allclose(star.density * conversion, 1.4, atol=0.01)


def test_keplerian_central_from_orbit():
    sun = Central.from_orbital_properties(
        period=365.25,
        semimajor=constants.au,
        radius=1.0,
        body_mass=constants.M_earth,
    )

    # add large tolerance to account for lack of precision in ureg.yr
    assert_allclose(sun.mass, 1.0, atol=5e-5)


def test_keplerian_body_keplers_law():
    orbit = System().add_body(
        semimajor=constants.au,
    )
    assert_allclose(orbit.bodies[0].period, 365.25, atol=0.01)

    orbit = System().add_body(period=365.25)
    assert_allclose(orbit.bodies[0].semimajor, constants.au, atol=0.01)


@pytest.mark.parametrize("prefix", ["", "central_", "relative_"])
def test_keplerian_body_velocity(time, system, prefix):
    body = system.bodies[0]
    v = getattr(body, f"{prefix}velocity")(time)
    for i, v_ in enumerate(v):
        pos_func = getattr(body, f"{prefix}position")
        assert_allclose(
            v_,
            jax.vmap(jax.grad(lambda t: pos_func(t)[i]))(time),
        )


def test_keplerian_body_radial_velocity(time, system):
    body = system.bodies[0]
    _ = body.radial_velocity(time)
    _ = body.radial_velocity(time, semiamplitude=1.0)


def test_keplerian_body_impact_parameter(system):
    body = system.bodies[0]
    x, y, z = body.relative_position(body.time_transit)
    assert_allclose(
        (jnp.sqrt(x**2 + y**2) / system.central.radius),
        body.impact_param,
    )
    assert jnp.all(z > 0)


def test_keplerian_body_coordinates_match_batman(time, system):
    _rsky = pytest.importorskip("batman._rsky")
    body = system.bodies[0]
    with jax.experimental.enable_x64(True):
        r_batman = _rsky._rsky(
            np.array(time, dtype=np.float64),
            float(body.time_transit),
            float(body.period),
            float(body.semimajor),
            float(body.inclination),
            (float(body.eccentricity) if body.eccentricity else 0.0),
            (float(body.omega_peri) if body.omega_peri else 0.0),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, z = body.relative_position(time)
        r = jnp.sqrt(x**2 + y**2)

        # Make sure that the in-transit impact parameter matches batman
        assert_allclose(r_batman[m], r[m])

        # In-transit should correspond to positive z in our parameterization
        assert np.all(z[m] > 0)

        # Therefore, when batman doesn't see a transit we shouldn't be transiting
        no_transit = z[~m] < 0
        no_transit |= r[~m] > 2
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
            np.array(time, dtype=np.float64),
            float(body.time_transit),
            float(body.period),
            float(body.semimajor),
            float(body.inclination),
            (float(body.eccentricity) if body.eccentricity else 0.0),
            (float(body.omega_peri) if body.omega_peri else 0.0),
            1,
            1,
        )
        m = r_batman < 100.0
        assert m.sum() > 0

        x, y, _ = body.relative_position(time)
        r = jnp.sqrt(x**2 + y**2)
        assert_allclose(r_batman[m], r[m])


def test_keplerian_system_stack_construction():
    sys = System().add_body(period=0.1).add_body(period=0.2)
    assert sys._body_stack.stack is not None
    assert_allclose(sys._body_stack.stack.period, jnp.array([0.1, 0.2]))

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
    # with pytest.raises(ValueError):
    #     sys1._body_stack.stack.radial_velocity(0.0)

    for t in [0.0, jnp.linspace(0, 1, 5)]:
        assert_allclose(
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
        assert_allclose(x1, x2)
        assert_allclose(y1, y2)
        assert_allclose(z1, z2)


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
    assert_pytree_allclose(result1, result2)
