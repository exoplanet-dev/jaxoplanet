# mypy: ignore-errors

import warnings

import jax
import numpy as np
import pytest
from jax.test_util import check_grads

from jaxoplanet.core.limb_dark import light_curve
from jaxoplanet.test_utils import assert_allclose

exoplanet_core = pytest.importorskip("exoplanet_core")


@pytest.mark.parametrize("r", [0.01, 0.1, 1.1, 2.0])
def test_compare_exoplanet(r):
    u1 = 0.2
    u2 = 0.3
    b = np.linspace(-1 - 2 * r, 1 + 2 * r, 5001)
    expect = exoplanet_core.quad_limbdark_light_curve(u1, u2, b, r)
    calc = jax.jit(light_curve)(np.array([u1, u2]), b, r)
    assert_allclose(calc, expect)


@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 0.9, 1.0, 1.5])
def test_compare_exoplanet_precise(r):
    """The light curve should match exoplanet-core to near machine precision
    relative to the transit depth, including right next to the contact points
    (worst case ~4e-12 of the depth, at the interior contact for small r)"""
    if not jax.config.jax_enable_x64:  # type: ignore
        pytest.skip("requires float64")
    u1 = 0.2
    u2 = 0.3
    b = np.sort(
        np.concatenate(
            [
                np.linspace(0, 1 + r, 1001),
                np.abs(1 - r) + np.linspace(-1e-3, 1e-3, 1001),
                np.abs(1 - r) + np.logspace(-15, -4, 100),
                np.abs(1 - r) - np.logspace(-15, -4, 100),
                r + np.linspace(-1e-3, 1e-3, 1001),
                1 + r - np.logspace(-15, -4, 100),
            ]
        )
    )
    b = b[(0 <= b) & (b <= 1 + r)]
    expect = exoplanet_core.quad_limbdark_light_curve(u1, u2, b, r)
    calc = jax.jit(light_curve)(np.array([u1, u2]), b, r)
    depth = np.max(np.abs(expect))
    np.testing.assert_allclose(calc, expect, atol=1e-12 + 1e-10 * depth, rtol=0)


@pytest.mark.parametrize("r", [0.1, 0.5, 1.0])
def test_deficit_relative_precision_f32(r):
    """The light curve is computed as a flux deficit, so even in single
    precision it should be accurate relative to the transit depth, not just
    relative to the out-of-transit flux."""
    if jax.config.jax_enable_x64:  # type: ignore
        pytest.skip("requires float32")
    u1 = 0.3
    u2 = 0.2
    b = np.sort(
        np.concatenate(
            [
                np.linspace(0, 1 + r, 3001),
                np.abs(1 - r) + np.linspace(-2e-2, 2e-2, 2001),
                r + np.linspace(-1e-3, 1e-3, 501),
            ]
        )
    ).astype(np.float32)
    b = b[(0 <= b) & (b <= np.float32(1 + r))]
    expect = exoplanet_core.quad_limbdark_light_curve(u1, u2, b.astype(np.float64), r)
    depth = np.max(np.abs(expect))
    calc = jax.jit(light_curve)(np.array([u1, u2], dtype=np.float32), b, np.float32(r))
    assert np.max(np.abs(np.asarray(calc, dtype=np.float64) - expect)) < 5e-5 * depth


@pytest.mark.parametrize("u", [[0.2], [0.2, 0.3], [0.2, 0.3, 0.1, 0.5, 0.02]])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 1.1, 2.0])
def test_edge_cases(u, r):
    u = np.array(u)
    for b in [0.0, 0.5, 1.0, r, np.abs(1 - r), 1 + r]:
        calc = jax.jit(light_curve)(u, b, r)
        assert np.isfinite(calc)
        if len(u) == 2:
            expect = exoplanet_core.quad_limbdark_light_curve(*u, b, r)
            assert_allclose(calc, expect[0])

        for n in range(3):
            g = jax.grad(light_curve, argnums=n)(u, b, r)
            assert np.all(np.isfinite(g))

    if jax.config.jax_enable_x64:  # type: ignore
        for b in [0.0, 0.5, 1.0, r, 1 + 2 * r]:
            if np.allclose(b, r) or np.allclose(np.abs(b - r), 1):
                continue
            check_grads(light_curve, (u, b, r), order=1)


@pytest.mark.parametrize("u", [[0.2, 0.3], [0.2, 0.3, 0.1, 0.5, 0.02]])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 1.0, 1.5])
def test_second_derivatives(u, r):
    """Second derivatives should be finite in every forward/reverse combination,
    including exactly at the contact points, and agree between modes away from
    them (reverse mode is sensitive to NaNs in discarded jnp.where branches)"""
    u = np.array(u)

    def f(x):
        return light_curve(u, x[0], x[1])

    modes = [
        jax.jacfwd(jax.jacfwd(f)),
        jax.hessian(f),
        jax.jacrev(jax.jacrev(f)),
    ]
    contacts = [0.0, np.abs(1 - r), r, 1.0, 1 + r]
    regular = [0.5 * np.abs(1 - r), 0.5 * (np.abs(1 - r) + 1 + r), 1 + 2 * r]
    for b in contacts + regular:
        hessians = [np.asarray(mode(np.array([b, r]))) for mode in modes]
        for H in hessians:
            assert np.all(np.isfinite(H)), (b, r)
        if b in regular:
            for H in hessians[1:]:
                assert_allclose(H, hessians[0])


# Gradients (d/db, d/dr) of the quadratic (u = [0.4, 0.26]) light curve close to the
# contact points, where the derivative is a sum of terms that each diverge like
# 1 / sqrt(distance to contact). Reference values from an independent float
# computation: a 1D integral of the occulted intensity over radius, at 80 digits
# with mpmath, differentiated numerically
CONTACT_GRAD_REFERENCE = [
    # (b, r, dF/db, dF/dr)
    # just outside the inner contact (b = 1 - r), 1 ulp and 1e-10 away
    (0.9900000000000001, 0.01, 0.0008812058544017708, -0.010965251678630546),
    (0.9900000001, 0.01, 0.0008815792074333273, -0.010964878312733642),
    (0.9000000000000001, 0.1, 0.022770375091276256, -0.15658642307776174),
    (0.9000000001, 0.1, 0.02277161318423, -0.15658518495444335),
    (0.5000000000000001, 0.5, 0.1581741701434329, -0.9664253287685911),
    (0.50000001, 0.5, 0.15821140831571054, -0.9663880874323337),
    # just inside the inner contact
    (0.8999999999999999, 0.1, 0.022770373631327986, -0.15658642453771007),
    # just inside the outer contact (b = 1 + r)
    (1.0999999999, 0.1, 1.1210238904990195e-06, -1.1210238908387212e-06),
    (1.4999999999, 0.5, 2.146598216530912e-06, -2.146598216626316e-06),
    # r = 1, b -> 0, and the full occultation edge (b = r - 1) for r > 1
    (1e-10, 1.0, 0.26290440728868386, -0.4129675190822041),
    (0.5000000001, 1.5, 6.439794648734097e-06, -6.439794649020308e-06),
]


@pytest.mark.parametrize("b, r, dfdb, dfdr", CONTACT_GRAD_REFERENCE)
def test_gradient_near_contact(b, r, dfdb, dfdr):
    if not jax.config.jax_enable_x64:  # type: ignore
        pytest.skip("requires float64")
    u = np.array([0.4, 0.26])
    g = jax.grad(lambda x: light_curve(u, x[0], x[1]))(np.array([b, r]))
    np.testing.assert_allclose(g, [dfdb, dfdr], rtol=1e-4)


@pytest.mark.parametrize("u", [[0.2], [0.2, 0.3], [0.2, 0.3, 0.1, 0.5, 0.02]])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 1.1, 2.0])
def test_compare_starry(u, r):
    u = np.array(u)
    starry = pytest.importorskip("starry")
    theano = pytest.importorskip("theano")
    theano.config.gcc__cxxflags += " -fexceptions"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        m = starry.Map(udeg=len(u))
        m[1:] = u
        b_ = theano.tensor.dscalar()
        func = theano.function([b_], m.flux(xo=b_, yo=0.0, zo=1.0, ro=r) - 1)
        for b in [0.0, 0.5, 1.0, r, np.abs(1 - r), 1 + r]:
            expect = func(b)[0]
            if not np.isfinite(expect):
                continue  # hack because starry doesn't handle all edge cases properly
            calc = light_curve(u, b, r)
            assert_allclose(calc, expect)

        b = np.linspace(-1 - 2 * r, 1 + 2 * r, 5001)
        expect = m.flux(xo=b, yo=0.0, zo=1.0, ro=r).eval() - 1
        calc = light_curve(u, b, r)
        assert_allclose(calc, expect)
