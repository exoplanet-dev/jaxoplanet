from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import math
from jax.experimental.sparse import BCOO

from jaxoplanet.types import Array


class Pijk(eqx.Module):
    data: dict[tuple[int, int, int], Array]
    degree: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, data: Mapping[tuple[int, int, int], Array]):
        self.data = dict(data)
        self.degree = max(self.ijk2lm(*ijk)[0] for ijk in data.keys())
        self.diagonal = all(self.ijk2lm(*ijk)[1] == 0.0 for ijk in data.keys())

    @property
    def shape(self):
        return (self.degree + 1) ** 2

    @property
    def indices(self):
        return self.data.keys()

    @staticmethod
    def ijk2lm(i, j, k):
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
    def n2lm(n):
        l = int(np.floor(np.sqrt(n)))
        m = n - l**2 - l
        return l, m

    @staticmethod
    def lm2n(l, m):
        return l**2 + l + m

    @staticmethod
    def lm2ijk(l, m):
        mu = l - m
        nu = l + m

        if nu % 2 == 0:
            return mu // 2, nu // 2, 0
        else:
            return (mu - 1) // 2, (nu - 1) // 2, 1

    @staticmethod
    def n2ijk(n):
        return Pijk.lm2ijk(*Pijk.n2lm(n))

    def index(self, i, j, k):
        l, m = self.ijk2lm(i, j, k)
        return int((l + 1) * l + m)

    def tosparse(self) -> BCOO:
        indices, values = zip(*self.data.items())
        idx = np.array([self.index(i, j, k) for i, j, k in indices])[:, None]
        return BCOO((jnp.asarray(values), idx), shape=(self.shape,))

    def todense(self) -> Array:
        return self.tosparse().todense()

    @classmethod
    def from_dense(cls, x: Array, degree: int) -> "Pijk":
        data = defaultdict(float)
        n = min((degree + 1) ** 2, len(x))
        for i in range(n):
            data[Pijk.n2ijk(i)] = x[i]

        return cls(data)

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
        return self.todense()[self.index(*key)]


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
