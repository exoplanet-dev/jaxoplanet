# -*- coding: utf-8 -*-

import numpy as np
from absl.testing import parameterized
from jax import config, test_util

from exoplanet_jax.ops import kepler

config.update("jax_enable_x64", True)

float_types = [np.dtype(np.float32), np.dtype(np.float64)]


def get_M_and_f(ecc, E):
    M = E - ecc * np.sin(E)
    f = 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(0.5 * E))
    return M, f


class KeplerTest(test_util.JaxTestCase):
    def _check_one(self, dtype, ecc, E):
        M, f = get_M_and_f(ecc, E)
        args_maker = lambda: (M.astype(dtype), ecc.astype(dtype))  # noqa
        expected = lambda *args: (  # noqa
            np.sin(f).astype(dtype),
            np.cos(f).astype(dtype),
        )
        self._CheckAgainstNumpy(expected, kepler, args_maker)
        # self._CompileAndCheck(kepler, args_maker)

    @parameterized.named_parameters(
        test_util.cases_from_list(
            {
                "testcase_name": "_dtype={}".format(dtype.name),
                "dtype": dtype,
            }
            for dtype in float_types
        )
    )
    def test_edge_cases(self, dtype):
        E = np.array([0.0, 2 * np.pi, -226.2, -170.4])
        ecc = (1 - 1e-6) * np.ones_like(E)
        ecc[-1] = 0.9939879759519037
        self._check_one(dtype, ecc, E)

        # sinf0, cosf0 = kepler(M, ecc)
        # assert np.all(np.isfinite(sinf0))
        # assert np.all(np.isfinite(cosf0))
        # np.testing.assert_allclose(np.sin(f), sinf0)
        # np.testing.assert_allclose(np.cos(f), cosf0)


# import aesara_theano_fallback.tensor as tt
# import numpy as np
# from aesara_theano_fallback import aesara as theano

# from exoplanet.theano_ops.kepler import Kepler
# from exoplanet.theano_ops.test_tools import InferShapeTester


# class TestKeplerSolver(InferShapeTester):
#     def setup_method(self):
#         super().setup_method()
#         self.op_class = Kepler
#         self.op = Kepler()

#     def _get_M_and_f(self, e, E):
#         M = E - e * np.sin(E)
#         f = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(0.5 * E))
#         return M, f

#     def test_edge(self):
#         E = np.array([0.0, 2 * np.pi, -226.2, -170.4])
#         e = (1 - 1e-6) * np.ones_like(E)
#         e[-1] = 0.9939879759519037
#         M, f = self._get_M_and_f(e, E)

#         M_t = tt.dvector()
#         e_t = tt.dvector()
#         func = theano.function([M_t, e_t], self.op(M_t, e_t))
#         sinf0, cosf0 = func(M, e)

#         assert np.all(np.isfinite(sinf0))
#         assert np.all(np.isfinite(cosf0))
#         assert np.allclose(np.sin(f), sinf0)
#         assert np.allclose(np.cos(f), cosf0)

#     def test_pi(self):
#         e = np.linspace(0, 1.0, 100)
#         E = np.pi + np.zeros_like(e)
#         M, f = self._get_M_and_f(e, E)

#         M_t = tt.dvector()
#         e_t = tt.dvector()
#         func = theano.function([M_t, e_t], self.op(M_t, e_t))
#         sinf0, cosf0 = func(M, e)

#         assert np.all(np.isfinite(sinf0))
#         assert np.all(np.isfinite(cosf0))
#         assert np.allclose(np.sin(f), sinf0)
#         assert np.allclose(np.cos(f), cosf0)

#     def test_twopi(self):
#         e = np.linspace(0, 1.0, 100)[:-1]
#         M = 2 * np.pi + np.zeros_like(e)

#         M_t = tt.dvector()
#         e_t = tt.dvector()
#         func = theano.function([M_t, e_t], self.op(M_t, e_t))
#         sinf0, cosf0 = func(np.zeros_like(M), e)
#         sinf, cosf = func(M, e)

#         assert np.all(np.isfinite(sinf0))
#         assert np.all(np.isfinite(cosf0))
#         assert np.allclose(sinf, sinf0)
#         assert np.allclose(cosf, cosf0)

#     def test_solver(self):
#         e = np.linspace(0, 1, 500)[:-1]
#         E = np.linspace(-300, 300, 1001)
#         e = e[None, :] + np.zeros((len(E), len(e)))
#         E = E[:, None] + np.zeros_like(e)
#         M, f = self._get_M_and_f(e, E)

#         M_t = tt.matrix()
#         e_t = tt.matrix()
#         func = theano.function([M_t, e_t], self.op(M_t, e_t))
#         sinf0, cosf0 = func(M, e)

#         assert np.all(np.isfinite(sinf0))
#         assert np.all(np.isfinite(cosf0))
#         assert np.allclose(np.sin(f), sinf0)
#         assert np.allclose(np.cos(f), cosf0)

#     def test_infer_shape(self):
#         np.random.seed(42)
#         M = tt.dvector()
#         e = tt.dvector()
#         M_val = np.linspace(-10, 10, 50)
#         e_val = np.random.uniform(0, 0.9, len(M_val))
#         self._compile_and_check(
#             [M, e], self.op(M, e), [M_val, e_val], self.op_class
#         )

#     def test_grad(self):
#         np.random.seed(1234)
#         M_val = np.concatenate(
#             (
#                 np.linspace(-10, 10, 100),
#                 [
#                     0.0,
#                     -np.pi + 1e-3,
#                     np.pi - 1e-3,
#                     0.5 * np.pi,
#                     -0.5 * np.pi,
#                     1.5 * np.pi,
#                     2 * np.pi + 1e-3,
#                 ],
#             )
#         )
#         e_val = np.random.uniform(0, 0.9, len(M_val))

#         a = lambda *args: tt.arctan2(*self.op(*args))  # NOQA
#         theano.gradient.verify_grad(a, [M_val, e_val], eps=1e-8, rng=np.random)
