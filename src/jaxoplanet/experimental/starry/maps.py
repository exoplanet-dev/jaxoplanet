from dataclasses import dataclass
import matplotlib.pyplot as plt
import math

import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.visualization import (
    lon_lat_lines,
    rotate_lines,
    plot_lines,
)
from jaxoplanet.experimental.starry.basis import poly_basis, A1, U0
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.render import ortho_grid
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.ylm import Ylm


@dataclass
class Map:
    y: Ylm = Ylm({(0, 0): 1})
    inc: float = math.pi / 2
    obl: float = 0.0
    u: tuple = ()

    def __post_init__(self):
        self._poly_basis = jax.jit(poly_basis(self.deg))

    @property
    def udeg(self):
        return len(self.u)

    @property
    def ydeg(self):
        return self.y.ell_max

    @property
    def deg(self):
        return self.ydeg + self.udeg

    def render(self, theta=0, res=400):
        _, xyz = ortho_grid(res)
        pT = self._poly_basis(*xyz)
        Ry = left_project(self.ydeg, self.inc, self.obl, theta, 0.0, self.y.todense())
        A1Ry = A1(self.ydeg) @ Ry
        ydeg = np.max([self.ydeg, self.udeg])
        p_y = Pijk.from_dense(A1Ry, degree=ydeg)
        U = np.array([1, *self.u])
        p_u = Pijk.from_dense(U @ U0(self.udeg, ydeg), degree=self.udeg)
        p = (p_y * p_u).todense()
        return jnp.reshape(pT @ p, (res, res))

    def show(self, theta=0, res=400, graticule=6, **kwargs):
        plt.imshow(
            self.render(theta, res), origin="lower", **kwargs, extent=(-1, 1, -1, 1)
        )
        if graticule is not None:
            self.graticule(theta)
        plt.axis(False)

    def graticule(self, theta=0, pts=100, **kwargs):
        kwargs.setdefault("c", kwargs.pop("color", "k"))
        kwargs.setdefault("lw", kwargs.pop("linewidth", 1))
        kwargs.setdefault("alpha", 0.5)

        # plot lines
        lat, lon = lon_lat_lines(pts=pts)
        lat = rotate_lines(lat, self.inc, self.obl, theta)
        plot_lines(lat, **kwargs)
        lon = rotate_lines(lon, self.inc, self.obl, theta)
        plot_lines(lon, **kwargs)
        theta = np.linspace(0, 2 * np.pi, 2 * pts)
        plt.plot(np.cos(theta), np.sin(theta), **kwargs)
