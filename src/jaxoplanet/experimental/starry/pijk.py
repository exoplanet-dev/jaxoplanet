from collections import defaultdict
from collections.abc import Mapping
from functools import reduce
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from jaxoplanet.types import Array


class Pijk(eqx.Module):
    r"""A class to represent and manipulate spherical harmonics in the
    polynomial basis. Several indices are used throughout the class:

    * Indices :math:`(i, j, k)` represent the order of the polynomials of
      :math:`(x, y, z)`, for example :math:`(1, 0, 2)` represents :math:`x\,z^2`.

    * Indices :math:`(l, m)` represent the orders of the spherical harmonics.

    * Index n represent the index of the polynomial in the flattened array.

    Flattened array ``todense`` and ``from_dense`` follow the convention from
    Luger et al. (2019). More specifically:

    .. math::

        \tilde{p} =
        \begin{pmatrix}
            1 & x & y & z & x^2 & xz & xy & yz & y^2 &
            \cdot\cdot\cdot
        \end{pmatrix}^\mathsf{T}

    """

    data: dict[tuple[int, int, int], Array]
    degree: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, data: Mapping[tuple[int, int, int], Array], degree: int = None):
        self.data = dict(data)
        self.degree = (
            max(self.ijk2lm(*ijk)[0] for ijk in data.keys())
            if degree is None
            else degree
        )

        dense_data = jnp.array(
            [data.get(Pijk.index_to_ijk(i), 0.0) for i in range((self.degree + 1) ** 2)]
        )
        all_ijk = set(map(Pijk.index_to_ijk, range((self.degree + 1) ** 2)))
        non_radially_symmetric = all_ijk.difference(Pijk.diagonal_ijk(self.degree))
        non_radially_symmetric_indices = jnp.array(
            [Pijk.ijk_to_index(*ijk) for ijk in non_radially_symmetric]
        ).astype(int)
        self.diagonal = jnp.allclose(dense_data[non_radially_symmetric_indices], 0.0)

    @staticmethod
    def z_to_ijk(n: int):
        """Converts the degree z to the polynomial indices (i, j, k).

        Args:
            n (int): degree of z

        Returns:
            set: set of polynomial indices (i, j, k)
        """
        if n == 0:
            res = [(0, 0, 0)]
        elif n == 1:
            res = [(0, 0, 1)]
        elif n % 2 == 0:
            res = [(2 * i, 2 * (n // 2 - i), 0) for i in range(n // 2 + 1)]
        else:
            res = [(2 * i, 2 * (n // 2 - i), 1) for i in range(n // 2 + 1)]

        return set(res)

    @staticmethod
    def diagonal_ijk(degree: int):
        ijk = list(reduce(set.union, [Pijk.z_to_ijk(z) for z in range(degree + 1)]))
        # order them
        idx = np.argsort(([Pijk.ijk_to_index(*i) for i in ijk]))
        ijk = [ijk[i] for i in idx]
        return ijk

    @property
    def shape(self):
        return (self.degree + 1) ** 2

    @property
    def indices(self):
        return self.data.keys()

    @staticmethod
    def ijk2lm(i: int, j: int, k: int):
        """Converts the polynomial indices (i, j, k) to the spherical harmonic indices
           (l, m).

        Args:
            i (int): degree on x
            j (int): degree on y
            k (int): degree on z

        Returns:
            tuple: (l: int, m: int)
        """
        assert k in (0, 1)

        if k == 0:
            nu = 2 * j
            mu = 2 * i
        else:
            nu = 2 * j + 1
            mu = 2 * i + 1

        m = int((nu - mu) / 2)
        l = int((nu + mu) / 2)

        return l, m

    @staticmethod
    def lm_to_ijk(l: int, m: int):
        """Converts the spherical harmonic indices (l, m) to the polynomial indices
           (i, j, k).

        Args:
            l (int): degree of the spherical harmonic
            m (int): order of the spherical harmonic

        Returns:
            tuple: (i: int, j: int, k: int)
        """
        mu = l - m
        nu = l + m

        if nu % 2 == 0:
            return mu // 2, nu // 2, 0
        else:
            return (mu - 1) // 2, (nu - 1) // 2, 1

    @staticmethod
    def index_to_lm(n: int):
        """Converts the index n of the flattened SH basis to the spherical harmonic
           indices (l, m).

        Args:
            n (int): index of the flattened SH basis

        Returns:
            tuple: (l: int, m: int)
        """
        l = int(np.floor(np.sqrt(n)))
        m = n - l * (l + 1)
        return l, m

    @staticmethod
    def index_to_ijk(n: int):
        """Converts the index n of the flattened SH basis to the
           polynomial indices (i, j, k).

        Args:
            n (int): index of the flattened SH basis

        Returns:
            tuple: (i: int, j: int, k: int)
        """
        l, m = Pijk.index_to_lm(n)
        return Pijk.lm_to_ijk(l, m)

    @staticmethod
    def ijk_to_index(i: int, j: int, k: int):
        """Converts the polynomial indices (i, j, k) to the index n of the flattened
           SH basis.

        Args:
            i (int): degree on x
            j (int): degree on y
            k (int): degree on z

        Returns:
            int: index n of the flattened SH basis
        """
        l, m = Pijk.ijk2lm(i, j, k)
        return int((l + 1) * l + m)

    def tosparse(self, diagonal: bool = False) -> BCOO:
        if diagonal:
            indices = self.diagonal_ijk(self.degree)
            values = [self.data[i] for i in indices]
            return jnp.asarray(values)
        else:
            indices, values = zip(*self.data.items(), strict=False)
            idx = np.array([self.ijk_to_index(i, j, k) for i, j, k in indices])[:, None]
            return BCOO((jnp.asarray(values), idx), shape=(self.shape,))

    def todense(self) -> Array:
        return self.tosparse().todense()

    @classmethod
    def from_dense(cls, x: Array, degree: int = None) -> "Pijk":
        data = defaultdict(float)
        for i in range(x.size):
            data[Pijk.index_to_ijk(i)] = x[i]

        return cls(data, degree=degree)

    def __mul__(self, other: Any) -> "Pijk":
        if isinstance(other, Pijk):
            return _mul(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x * other, self)

    def __rmul__(self, other: Any) -> "Pijk":
        assert not isinstance(other, Pijk)
        return jax.tree_util.tree_map(lambda x: other * x, self)

    def __getitem__(self, key) -> Array:
        assert isinstance(key, tuple)
        return self.todense()[self.ijk_to_index(*key)]


def _mul(u: Pijk, v: Pijk) -> Pijk:
    uv = defaultdict(float)
    for (i1, j1, k1), v1 in u.data.items():
        for (i2, j2, k2), v2 in v.data.items():
            v1v2 = v1 * v2
            if (k1 + k2) == 2:
                uv[(i1 + i2, j1 + j2, 0)] += v1v2
                uv[(i1 + i2 + 2, j1 + j2, 0)] -= v1v2
                uv[(i1 + i2, j1 + j2 + 2, 0)] -= v1v2
            else:
                uv[(i1 + i2, j1 + j2, k1 + k2)] += v1v2

    return Pijk(uv)
