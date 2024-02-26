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
