from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from jaxoplanet.experimental.starry.basis import A1, U0, poly_basis
from jaxoplanet.experimental.starry.light_curves import light_curve
from jaxoplanet.experimental.starry.pijk import Pijk
from jaxoplanet.experimental.starry.rotation import left_project
from jaxoplanet.experimental.starry.utils import (
    lon_lat_lines,
    ortho_grid,
    plot_lines,
    rotate_lines,
)
from jaxoplanet.experimental.starry.ylm import Ylm


class Map(eqx.Module):
    y: Optional[Ylm] = None
    inc: Optional[float] = None
    obl: Optional[float] = None
    u: Optional[tuple] = None
    period: Optional[float] = None

    def __init__(
        self,
        *,
        y: Optional[Ylm] = None,
        inc: Optional[float] = None,
        obl: Optional[float] = None,
        u: Optional[tuple] = None,
        period: Optional[float] = None,
    ):

        if y is None:
            y = Ylm({(0, 0): 1})
        if inc is None:
            inc = jnp.pi / 2
        if obl is None:
            obl = 0.0
        if u is None:
            u = ()

        self.y = y
        self.inc = inc
        self.obl = obl
        self.u = u
        self.period = period

    @property
    def poly_basis(self):
        return jax.jit(poly_basis(self.deg))

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
        pT = self.poly_basis(*xyz)
        Ry = left_project(self.ydeg, self.inc, self.obl, theta, 0.0, self.y.todense())
        A1Ry = A1(self.ydeg) @ Ry
        p_y = Pijk.from_dense(A1Ry, degree=self.ydeg)
        U = np.array([1, *self.u])
        p_u = Pijk.from_dense(U @ U0(self.udeg), degree=self.udeg)
        p = (p_y * p_u).todense()
        return jnp.reshape(pT @ p, (res, res))

    def show(self, theta=0, res=400, graticule=6, **kwargs):
        import matplotlib.pyplot as plt

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

    def light_curve(self, system, time):
        theta = 0.0 if self.period is None else 2.0 * np.pi * time / self.period
        lc_func = jnp.vectorize(
            partial(
                light_curve,
                self.ydeg,
                self.y.todense(),
                self.u,
                self.inc,
                self.obl,
                theta,
            ),
            signature="(),(),(),()->()",
        )

        def body_lc(body):
            x, y, z = body.relative_position(time)
            return lc_func(x.magnitude, y.magnitude, z.magnitude, body.radius.magnitude)

        lc = jax.vmap(body_lc)(system._body_stack)

        return lc
