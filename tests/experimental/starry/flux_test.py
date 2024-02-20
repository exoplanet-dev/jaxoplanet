import jax
import numpy as np
import pytest

from jaxoplanet.experimental.starry import Map, Ylm
from jaxoplanet.experimental.starry.light_curves import map_light_curve
from jaxoplanet.test_utils import assert_allclose


@pytest.mark.parametrize("deg", [2, 5, 10])
@pytest.mark.parametrize("u", [[], [0.1], [0.2, 0.1]])
def test_compare_starry(deg, u):
    starry = pytest.importorskip("starry")
    starry.config.lazy = False

    # map
    inc = np.pi / 2
    np.random.seed(deg)
    y = Ylm.from_dense(np.random.randn((deg + 1) ** 2))
    map = Map(y=y, u=u, inc=inc)

    # occultor
    yo = np.linspace(-3, 3, 1000)
    ro = 0.01
    zo = 2.0
    xo = 0.0

    # starry
    ms = starry.Map(ydeg=deg, udeg=len(u), inc=np.rad2deg(inc))
    ms[:, :] = y.todense()
    if len(u) > 0:
        ms[1:] = u
    expected = ms.flux(xo=xo, yo=yo, ro=ro, zo=zo)

    # jaxoplanet
    calc = jax.vmap(map_light_curve, in_axes=(None, None, None, 0, None, None))(
        map, ro, xo, yo, zo, 0.0
    )

    assert_allclose(calc, expected)
