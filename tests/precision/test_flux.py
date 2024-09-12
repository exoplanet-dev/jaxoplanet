import jax
import numpy as np
import pytest

from jaxoplanet.experimental.starry import light_curves
from jaxoplanet.experimental.starry.multiprecision import flux as mp_flux, mp
from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.test_utils import assert_allclose

TOLERANCE = 1e-15


@pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_flux(r, l_max=5, order=500):

    # We know that these are were the errors are the highest
    b = 1 - r if r < 1 else r

    n = (l_max + 1) ** 2
    expect = np.zeros(n)
    calc = np.zeros(n)
    ys = np.eye(n, dtype=np.float64)
    ys[:, 0] = 1.0

    @jax.jit
    def light_curve(y):
        surface = Surface(y=Ylm.from_dense(y, normalize=False), normalize=False)
        return light_curves.surface_light_curve(surface, y=b, z=10.0, r=r, order=order)

    for i, y in enumerate(ys):
        expect[i] = float(mp_flux.flux_function(l_max, mp.pi / 2, 0.0)(y, b, r, 0.0))
        calc[i] = light_curve(y)

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
