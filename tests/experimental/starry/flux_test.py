import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet.experimental.starry.light_curves import light_curve
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", [2, 5, 10])
@pytest.mark.parametrize("u", [[], [0.1], [0.2, 0.1]])
def test_compare_starry(deg, u):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    np.random.seed(deg)
    y = np.random.randn((deg + 1) ** 2)
    y[0] = 1.0
    y[1:] *= 1e-2
    inc = np.pi / 2

    # occultor
    yo = np.linspace(-3, 3, 1000)
    ro = 0.01
    zo = 2.0
    xo = 0.0

    # starry
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    ms[:, :] = y
    if len(u) > 0:
        ms[1:] = u
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jnp.vectorize(
        light_curve, signature="(n),(),(),(),(),(),(),()->()", excluded={0, 2}
    )(deg, y, u, inc, 0.0, 0.0, xo, yo, zo, ro)

    assert_allclose(calc, expected)
