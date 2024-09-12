import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from jaxoplanet.experimental.starry import solution
from jaxoplanet.experimental.starry.multiprecision import solution as mp_solution, utils
from jaxoplanet.test_utils import assert_allclose

TOLERANCE = 1e-15


@pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_sT(r, l_max=10, order=500):

    # We know that these are were the errors are the highest
    b = 1 - r if r < 1 else r

    expect = utils.to_numpy(mp_solution.sT(l_max, b, r))
    calc = solution.solution_vector(l_max, order=order)(b, r)

    for n in range(expect.size):
        # For logging/debugging purposes, work out the case id
        l = np.floor(np.sqrt(n)).astype(int)  # noqa
        m = n - l**2 - l
        mu = l - m
        nu = l + m
        if mu == 1 and l == 1:
            case = 1
        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 2
        elif mu == 1 and l % 2 == 0:
            case = 3
        elif mu == 1:
            case = 4
        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 5
        else:
            case = 0

        assert_allclose(
            calc[n],
            expect[n],
            err_msg=f"n={n}, l={l}, m={m}, mu={mu}, nu={nu}, case={case}",
            atol=TOLERANCE,
        )
