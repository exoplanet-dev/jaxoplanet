import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO
from scipy.special import legendre as LegendreP

from jaxoplanet.experimental.starry.rotation import dot_rotation_matrix
from jaxoplanet.experimental.starry.wigner3j import Wigner3jCalculator
from jaxoplanet.types import Array


class Ylm(eqx.Module):
    """Ylm object containing the spherical harmonic coefficients.

    Args:
        data (Mapping[tuple[int, int], Array], optional): dictionary of
            spherical harmonic coefficients. Defaults to {(0, 0): 1.0}.
    """

    # coefficients of the spherical harmonic expansion of the map in the form
    # {(l, m): coefficient}
    data: dict[tuple[int, int], Array]

    # maximum degree of the spherical harmonic expansion
    ell_max: int = eqx.field(static=True)

    # whether the spherical harmonic expansion is diagonal (all m=0)
    diagonal: bool = eqx.field(static=True)

    def __init__(
        self,
        data: Optional[Mapping[tuple[int, int], Array]] = None,
    ):
        if data is None:
            data = {(0, 0): 1.0}

        self.data = dict(data)
        self.ell_max = max(ell for ell, _ in data.keys())
        self.diagonal = all(m == 0 for _, m in data.keys())

    @property
    def shape(self) -> tuple[int, ...]:
        """The number of coefficients in the expansion. This sets the shape of
        the output of `todense`."""
        return (self.ell_max**2 + 2 * self.ell_max + 1,)

    @property
    def indices(self) -> list[tuple[int, int]]:
        return list(self.data.keys())

    def index(self, ell: Array, m: Array) -> Array:
        """Convert the degree and order of the spherical harmonic to the
        corresponding index in the coefficient array."""
        return ell * (ell + 1) + m

    def normalize(self) -> "Ylm":
        """Return a new Ylm instance with normalized coefficients.

        Returns:
            Ylm instance with normalized coefficients.

        Raises:
            ValueError: if the (0, 0) coefficient is zero.
        """

        assert self.data[(0, 0)] != 0.0, ValueError(
            "The (0, 0) coefficient must be non-zero to normalize"
        )
        data = {k: v / self.data[(0, 0)] for k, v in self.data.items()}
        return Ylm(data=data)

    def tosparse(self) -> BCOO:
        indices, values = zip(*self.data.items())
        idx = jnp.array([self.index(ell, m) for ell, m in indices])[:, None]
        return BCOO((jnp.asarray(values), idx), shape=self.shape)

    def todense(self) -> Array:
        return self.tosparse().todense()

    @classmethod
    def from_dense(cls, y: Array, normalize: bool = True) -> "Ylm":
        data = {}
        for i, ylm in enumerate(y):
            ell = int(np.floor(np.sqrt(i)))
            m = i - ell * (ell + 1)
            data[(ell, m)] = ylm
        ylm = cls(data)
        if normalize:
            return ylm.normalize()
        else:
            return ylm

    def __mul__(self, other: Any) -> "Ylm":
        if isinstance(other, Ylm):
            return _mul(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x * other, self)

    def __rmul__(self, other: Any) -> "Ylm":
        assert not isinstance(other, Ylm)
        return jax.tree_util.tree_map(lambda x: other * x, self)

    def __getitem__(self, key) -> Array:
        assert isinstance(key, tuple)
        return self.todense()[self.index(*key)]


def _mul(f: Ylm, g: Ylm) -> Ylm:
    """
    Based closely on the implementation from the MIT-licensed spherical package:

    https://github.com/moble/spherical/blob/0aa81c309cac70b90f8dfb743ce35d2cc9ae6dee/spherical/multiplication.py
    """
    ellmax_f = f.ell_max
    ellmax_g = g.ell_max
    ellmax_fg = ellmax_f + ellmax_g
    fg = defaultdict(lambda *_: 0.0)
    m_calculator = Wigner3jCalculator(ellmax_f, ellmax_g)
    for ell1 in range(ellmax_f + 1):
        sqrt1 = math.sqrt((2 * ell1 + 1) / (4 * math.pi))
        for m1 in range(-ell1, ell1 + 1):
            idx1 = (ell1, m1)
            if idx1 not in f.data:
                continue
            sum1 = sqrt1 * f.data[idx1]
            for ell2 in range(ellmax_g + 1):
                sqrt2 = math.sqrt(2 * ell2 + 1)
                # w3j_s = s_calculator.calculate(ell1, ell2, s_f, s_g)
                for m2 in range(-ell2, ell2 + 1):
                    idx2 = (ell2, m2)
                    if idx2 not in g.data:
                        continue
                    w3j_m = m_calculator.calculate(ell1, ell2, m1, m2)
                    sum2 = sqrt2 * g.data[idx2]
                    m3 = m1 + m2
                    for ell3 in range(
                        max(abs(m3), abs(ell1 - ell2)), min(ell1 + ell2, ellmax_fg) + 1
                    ):
                        # Could loop over same (ell3, m3) more than once, so add all
                        # contributions together
                        fg[(ell3, m3)] += (
                            (
                                math.pow(-1, ell1 + ell2 + ell3 + m3)
                                * math.sqrt(2 * ell3 + 1)
                                * w3j_m[ell3]  # Wigner3j(ell1, ell2, ell3, m1, m2, -m3)
                            )
                            * sum1
                            * sum2
                        )
    return Ylm(fg)


def Bp(ydeg, npts: int = 1000, eps: float = 1e-9, smoothing=None):
    """
    Return the matrix B+. This expands the
    spot profile `b` in Legendre polynomials. From https://github.com/rodluger/
    mapping_stellar_surfaces/blob/paper2-arxiv/paper2/figures/spot_profile.py and
    _spot_setup in starry/_core/core.py.
    """
    if smoothing is None:
        if ydeg < 4:
            smoothing = 0.5
        else:
            smoothing = 2.0 / ydeg

    theta = jnp.linspace(0, jnp.pi, npts)
    cost = jnp.cos(theta)
    B = jnp.hstack(
        [
            jnp.sqrt(2 * l + 1) * LegendreP(l)(cost).reshape(-1, 1)
            for l in range(ydeg + 1)
        ]
    )
    _Bp = jnp.linalg.solve(B.T @ B + eps * jnp.eye(ydeg + 1), B.T)
    l = jnp.arange(ydeg + 1)
    indices = l * (l + 1)
    S = jnp.exp(-0.5 * indices * smoothing**2)
    return (S[:, None] * _Bp, theta, indices)


def spot_profile(theta, radius, spot_fac=300):
    """
    The sigmoid spot profile.

    """
    z = spot_fac * (theta - radius)
    return 1 / (1 + jnp.exp(-z)) - 1


def ylm_spot(ydeg: int) -> callable:
    """spot expansion in the spherical harmonics basis.

    Args:
        ydeg (int): max degree of the spherical harmonics

    Returns:
        callable: function that returns the spherical harmonics coefficients of the spot
    """
    B, theta, indices = Bp(ydeg)

    def func(contrast: float, r: float, lat: float = 0.0, lon: float = 0.0):
        """spot expansion in the spherical harmonics basis.

        Args:
            contrast (float): spot contrast, defined as (1-c) where c is the intensity of
            the center of the spot relative to the unspotted surface. A contrast of 1.
            means that the spot intensity drops to zero at the center, 0. means that
            the intensity at the center of the spot is the same as the intensity of the
            unspotted surface.
            r (float): radius of the spot.
            lat (float, optional): latitude of the spot, assuming that the center of a
            star with an inclination of pi/2 has a latitude of 0. Defaults to 0.0.
            lon (float, optional): longitude of the spot, assuming that the center of a
            star with an inclination of pi/2 has a longitude of 0. Defaults to 0.0.

        Returns:
            Ylm: Ylm object containing the spherical harmonics coefficients of the spot
        """
        b = spot_profile(theta, r)
        y = jnp.zeros((ydeg + 1) * (ydeg + 1))
        y = y.at[indices].set(B @ b * contrast)
        y = y.at[0].set(y[0] + 1.0)
        y = dot_rotation_matrix(ydeg, 1.0, 0.0, 0.0, lat)(y)
        y = dot_rotation_matrix(ydeg, 0.0, 1.0, 0.0, -lon)(y)

        return Ylm.from_dense(y, normalize=False)

    return func


# TODO
def ring_y(l_max, pts=1000, eps=1e-9, smoothing=None):
    Bp, theta, idxs = None, None, None
    n_max = l_max**2 + 2 * l_max + 1

    def _y(contrast: float, width: float, latitude: float):
        b = 1 - jnp.array((theta > latitude - width) & (theta < latitude + width))
        y = jnp.zeros(n_max)
        y = y.at[idxs].set(Bp @ b)
        return y * contrast

    return _y
