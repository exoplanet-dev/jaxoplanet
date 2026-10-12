import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet.light_curves import limb_dark_light_curve
from jaxoplanet.orbits.keplerian import System


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
    lc = limb_dark_light_curve(orbit, params["u"])(t)
    assert lc.shape == t.shape + (1,)


def _systems():
    single = System().add_body(period=3.0, radius=0.1, impact_param=0.3)
    double = single.add_body(
        period=7.3, radius=0.03, impact_param=0.5, time_transit=1.1
    )
    return {"one body": single, "two bodies": double}


@pytest.mark.parametrize("name", ["one body", "two bodies"])
@pytest.mark.parametrize("u", [[0.3, 0.2], [0.3, 0.2, 0.05], []])
def test_methods_agree(name, u):
    orbit = _systems()[name]
    t = jnp.linspace(-1.0, 8.0, 3000)
    args = [jnp.asarray(u)] if u else []
    expect = limb_dark_light_curve(orbit, *args, method="agol")(t)
    calc = limb_dark_light_curve(orbit, *args, method="pizzetti")(t)
    assert calc.shape == expect.shape
    depth = np.max(np.abs(expect))
    tol = 2e-10 if jax.config.jax_enable_x64 else 5e-5  # type: ignore
    np.testing.assert_allclose(calc, expect, atol=tol * depth, rtol=0)


def test_pizzetti_time_shapes():
    orbit = _systems()["two bodies"]
    lc = limb_dark_light_curve(orbit, jnp.asarray([0.3, 0.2]), method="pizzetti")
    assert lc(jnp.asarray(0.0)).shape == (2,)
    assert lc(jnp.zeros(5)).shape == (5, 2)
    assert lc(jnp.zeros((3, 4))).shape == (3, 4, 2)
    body = limb_dark_light_curve(
        orbit.bodies[0] if hasattr(orbit, "bodies") else orbit,
        jnp.asarray([0.3, 0.2]),
        method="pizzetti",
    )
    assert body(jnp.zeros(5)).shape[0] == 5


def test_pizzetti_grad_through_orbit():
    t = jnp.linspace(-0.3, 0.3, 500)
    u = jnp.asarray([0.3, 0.2])

    def loss(method, radius):
        orbit = System().add_body(period=3.0, radius=radius, impact_param=0.3)
        return jnp.sum(limb_dark_light_curve(orbit, u, method=method)(t))

    expect = jax.grad(loss, argnums=1)("agol", 0.1)
    calc = jax.grad(loss, argnums=1)("pizzetti", 0.1)
    rtol = 1e-8 if jax.config.jax_enable_x64 else 1e-3  # type: ignore
    np.testing.assert_allclose(calc, expect, rtol=rtol)


def test_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        limb_dark_light_curve(_systems()["one body"], jnp.asarray([0.1]), method="x")
