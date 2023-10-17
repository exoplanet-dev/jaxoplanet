import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

from jaxoplanet.experimental.starry.wigner3j import Wigner3jCalculator
from jaxoplanet.types import Array


class Ylm(eqx.Module):
    data: dict[tuple[int, int], Array]
    ell_max: int = eqx.field(static=True)
    diagonal: bool = eqx.field(static=True)

    def __init__(self, data: Mapping[tuple[int, int], Array]):
        # TODO(dfm): Think about how we want to handle the case of (0, 0). In
        # starry, that is always forced to be 1, and all the entries will be
        # normalized if an explicit value is provided. Do we want to enforce
        # that here too?
        self.data = dict(data)
        self.ell_max = max(ell for ell, _ in data.keys())
        self.diagonal = all(m == 0 for _, m in data.keys())

    @property
    def degree(self):
        return self.ell_max**2 + 2 * self.ell_max + 1

    @property
    def indices(self):
        return self.data.keys()

    def index(self, ell: Array, m: Array) -> Array:
        if np.any(np.abs(m) > ell):
            raise ValueError(
                "All spherical harmonic orders 'm' must be in the range [-ell, ell]"
            )
        return ell * (ell + 1) + m

    def tosparse(self) -> BCOO:
        indices, values = zip(*self.data.items())
        idx = np.array([self.index(ell, m) for ell, m in indices])[:, None]
        return BCOO((jnp.asarray(values), idx), shape=(self.degree,))

    def todense(self) -> Array:
        return self.tosparse().todense()

    def __mul__(self, other: Any) -> "Ylm":
        if isinstance(other, Ylm):
            return _mul(self, other)
        else:
            return jax.tree_util.tree_map(lambda x: x * other, self)

    def __rmul__(self, other: Any) -> "Ylm":
        assert not isinstance(other, Ylm)
        return jax.tree_util.tree_map(lambda x: other * x, self)


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
