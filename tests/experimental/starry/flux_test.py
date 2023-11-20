import pytest
import numpy as np
import jax.numpy as jnp
from jaxoplanet.experimental.starry.flux import flux


@pytest.mark.parametrize("deg", [2, 5, 10])
def test_compare_starry(deg):
    starry = pytest.importorskip("starry")

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
    ms = starry.Map(ydeg=deg, udeg=0, inc=np.rad2deg(inc))
    ms[:, :] = y
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jnp.vectorize(
        flux, signature="(),(),(),(),(),(),(),(n),(),()->()", excluded={0}
    )(deg, 0.0, xo, yo, zo, ro, inc, 0.0, y, 0, 0.0)

    np.assert_allclose(calc, expected)
