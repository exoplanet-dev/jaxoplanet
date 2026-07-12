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
    relative to the transit depth; the worst case is a narrow sliver around the
    interior contact point where the quadrature is only accurate to ~1e-7 of
    the depth."""
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
    np.testing.assert_allclose(calc, expect, atol=1e-12 + 1e-7 * depth, rtol=0)


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
