# mypy: ignore-errors

import jax
import exoplanet_core
import numpy as np
import pytest
from jax.test_util import check_grads

from exo4jax._src.experimental.starry.limb_dark import light_curve
from exo4jax.test_utils import assert_allclose


@pytest.mark.parametrize("r", [0.01, 0.1, 1.1, 2.0])
def test_compare_exoplanet(r):
    u1 = 0.2
    u2 = 0.3
    b = np.linspace(-1 - 2 * r, 1 + 2 * r, 5001)
    expect = exoplanet_core.quad_limbdark_light_curve(u1, u2, b, r)
    calc = jax.jit(light_curve)([u1, u2], b, r)
    assert_allclose(calc, expect)


@pytest.mark.parametrize("u", [[0.2], [0.2, 0.3], [0.2, 0.3, 0.1, 0.5, 0.01]])
@pytest.mark.parametrize("r", [0.01, 0.1, 0.5, 1.1, 2.0])
def test_edge_cases(u, r):

    for b in [0.0, 0.5, 1.0, r, np.abs(1 - r), 1 + r]:
        calc = jax.jit(light_curve)(u, b, r)
        assert np.isfinite(calc)
        if len(u) == 2:
            expect = exoplanet_core.quad_limbdark_light_curve(*u, b, r)
            assert_allclose(calc, expect)

        for n in range(3):
            g = jax.grad(light_curve, argnums=n)(u, b, r)
            assert np.all(np.isfinite(g))

    for b in [0.0, 0.5, 1.0, r, 1 + 2 * r]:
        if np.allclose(b, r) or np.allclose(np.abs(b - r), 1):
            continue
        print(b)
        check_grads(light_curve, (u, b, r), order=1)
