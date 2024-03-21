import numpy as np

from jaxoplanet.experimental.starry.orbit import SurfaceBody
from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.experimental.starry.visualization import show_map
from jaxoplanet.experimental.starry.ylm import Ylm


def test_show_map():
    y = Ylm.from_dense(np.hstack([1.0, np.random.rand(10) * 1e-1]))
    surface = Surface(y=y, inc=0.9, obl=-0.3, period=1.2, u=[0.5, 0.5])
    surface_body = SurfaceBody(period=1.0, surface=surface)

    show_map(surface_body)
    show_map(surface)
    show_map(y)
