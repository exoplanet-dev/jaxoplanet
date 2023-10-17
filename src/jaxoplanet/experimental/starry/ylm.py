from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from jaxoplanet.types import Array


class Ylm(eqx.Module):
    coeffs: Array
    indices: list[tuple[int, int]] = eqx.field(static=True)
    ell_max: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, data: Mapping[tuple[int, int], Array]):
        # TODO(dfm): Think about how we want to handle the case of (0, 0). In
        # starry, that is always forced to be 1, and all the entries will be
        # normalized if an explicit value is provided. Do we want to enforce
        # that here too?
        indices, coeffs = zip(*dict(data).items())
        self.coeffs = jnp.asarray(coeffs)
        self.indices = indices
        self.ell_max = max(ell for ell, _ in indices)
        self.diagonal = all(m == 0 for _, m in indices)

    @property
    def degree(self):
        return self.ell_max**2 + 2 * self.ell_max + 1

    def index(self, ell: Array, m: Array) -> Array:
        return ell * (ell + 1) + m

    def tosparse(self) -> BCOO:
        idx = np.array([self.index(ell, m) for ell, m in self.indices])[:, None]
        return BCOO((self.coeffs, idx), shape=(self.degree,))

    def todense(self) -> Array:
        return self.tosparse().todense()

    def __mul__(self, other: Any) -> "Ylm":
        if isinstance(other, Ylm):
            # TODO(dfm): Implement product function!
            raise NotImplementedError
        else:
            return eqx.tree_at(lambda t: t.coeffs, self, self.coeffs * other)

    def __rmul__(self, other: Any) -> "Ylm":
        assert not isinstance(other, Ylm)
        return eqx.tree_at(lambda t: t.coeffs, self, other * self.coeffs)
