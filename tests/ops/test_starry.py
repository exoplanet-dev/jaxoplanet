import numpy as np
import jax
import pytest

from exo4jax.ops.starry import kappas, solution_vector


def test_kappas():
    def _phi(b, r):
        arg = 1 - r**2 - b**2
        m = (r > 0) & (b > 0)
        arg[m] /= 2 * r[m] * b[m]
        result = np.arcsin(np.clip(arg, -1, 1))
        result[~m] = 0.5 * np.pi * np.sign(arg[~m])
        return result

    def _lam(b, r):
        arg = 1 - r**2 + b**2
        m = (r > 0) & (b > 0)
        arg[m] /= 2 * b[m]
        result = np.arcsin(np.clip(arg, -1, 1))
        result[~m] = 0.5 * np.pi * np.sign(arg[~m])
        return result

    b, r = np.meshgrid(np.linspace(0, 2, 100), np.linspace(0.01, 1.5, 13))
    b = b.flatten()
    r = r.flatten()
    kappa0, kappa1 = jax.vmap(kappas)(b, r)
    phi = _phi(b, r)
    lam = _lam(b, r)

    np.testing.assert_allclose(kappa0, phi + 0.5 * np.pi)
    np.testing.assert_allclose(kappa1, 0.5 * np.pi - lam)


@pytest.mark.parametrize("r", [0.1, 1.1])
def test_compare_starry(r, l_max=10, order=20):
    starry = pytest.importorskip("starry")

    b = np.linspace(0, 1 + r, 501)[:-1]
    m = starry.Map(l_max)
    s_expect = m.ops.sT(b, r)
    s_calc = solution_vector(l_max, order=order)(b, r)

    for n in range(s_expect.shape[1]):
        # For logging/debugging purposes, work out the case id
        l = np.floor(np.sqrt(n)).astype(int)
        m = n - l**2 - l
        mu = l - m
        nu = l + m
        if mu == 1 and l == 1:
            case = 1
        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 2
        elif mu == 1 and l % 2 == 0:
            case = 3
        elif mu == 1:
            case = 4
        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 5
        else:
            case = 0

        np.testing.assert_allclose(
            s_calc[:, n],
            s_expect[:, n],
            atol=1e-6,
            err_msg=f"n={n}, l={l}, m={m}, mu={mu}, nu={nu}, case={case}",
        )
