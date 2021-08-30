# -*- coding: utf-8 -*-

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from exoplanet_jax import ops

config.parse_flags_with_absl()


def get_mean_and_true_anomaly(eccentricity, eccentric_anomaly):
    M = eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)
    f = 2 * np.arctan(
        np.sqrt((1 + eccentricity) / (1 - eccentricity))
        * np.tan(0.5 * eccentric_anomaly)
    )
    return M, f


class KeplerTest(jtu.JaxTestCase):
    def _check(self, e, E):
        M, f = get_mean_and_true_anomaly(e, E)
        sinf0, cosf0 = ops.kepler(M, e)
        self.assertArraysAllClose(sinf0, jnp.sin(f), atol=1e-6)
        self.assertArraysAllClose(cosf0, jnp.cos(f), atol=1e-6)

    def test_basic(self):
        e = jnp.linspace(0, 1, 500)[:-1]
        E = jnp.linspace(-300, 300, 1001)
        e = e[None, :] + jnp.zeros((E.shape[0], e.shape[0]))
        E = E[:, None] + jnp.zeros_like(e)
        self._check(e, E)

    def test_edge_cases(self):
        E = jnp.array([0.0, 2 * jnp.pi, -226.2, -170.4])
        e = (1 - 1e-6) * jnp.ones_like(E)
        e = jnp.append(e[:-1], 0.9939879759519037)
        self._check(e, E)

    def test_pi(self):
        e = jnp.linspace(0, 1.0, 100)[:-1]
        E = jnp.pi + jnp.zeros_like(e)
        self._check(e, E)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
