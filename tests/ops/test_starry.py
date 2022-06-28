from functools import partial

import numpy as np
from scipy.integrate import quad
import jax
import pytest

from exo4jax.ops.starry import p, q, kappas


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


def test_q(l_max=10):
    def _numerical_q(l, m, lam):
        def calc(u, v, lam):
            sol, _ = quad(
                lambda phi: np.cos(phi) ** u * np.sin(phi) ** v,
                np.pi - lam,
                2 * np.pi + lam,
            )
            return sol

        mu = l - m
        nu = l + m
        if mu // 2 % 2 == 0:
            u = (mu + 4) // 2
            v = nu // 2
            return calc(u, v, lam)
        return np.zeros_like(lam)

    lam = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100)
    q_calc = jax.vmap(q, in_axes=(None, 0))(l_max, lam)
    assert np.all(np.isfinite(q_calc)), f"n_nan = {np.sum(np.isnan(q_calc))}"

    n = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            q_expect = np.array([_numerical_q(l, m, lam_) for lam_ in lam])
            np.testing.assert_allclose(
                q_calc[:, n], q_expect, atol=1e-12, err_msg=f"l={l}, m={m}"
            )
            n += 1


@pytest.mark.parametrize(
    "l,m", [(l, m) for l in range(10) for m in range(-l, l + 1)]
)
def test_p(l, m):
    def _numerical_p(b, r, kappa):
        cond = (b > 0) & (r > 0)
        delta = (b - r) / (2 * r)
        k2 = (1 - r**2 - b**2 + 2 * b * r) / np.where(cond, 4 * b * r, 1)
        if (mu / 2) % 2 == 0:
            func = (
                lambda x, l, mu, nu, delta, k2, b, r: 2
                * (2 * r) ** (l + 2)
                * (np.sin(x) ** 2 - np.sin(x) ** 4) ** (0.25 * (mu + 4))
                * (delta + np.sin(x) ** 2) ** (0.5 * nu)
            )
            cond = np.ones_like(cond)
        elif (mu == 1) and (l % 2 == 0):
            func = (
                lambda x, l, mu, nu, delta, k2, b, r: (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (np.sin(x) ** 2 - np.sin(x) ** 4) ** (0.5 * (l - 2))
                * np.maximum(0, k2 - np.sin(x) ** 2) ** (3.0 / 2.0)
                * (1 - 2 * np.sin(x) ** 2)
            )
        elif (mu == 1) and (l != 1) and (l % 2 != 0):
            func = (
                lambda x, l, mu, nu, delta, k2, b, r: (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (np.sin(x) ** 2 - np.sin(x) ** 4) ** (0.5 * (l - 3))
                * (delta + np.sin(x) ** 2)
                * np.maximum(0, k2 - np.sin(x) ** 2) ** (3.0 / 2.0)
                * (1 - 2 * np.sin(x) ** 2)
            )
        elif ((mu - 1) % 2) == 0 and ((mu - 1) // 2 % 2 == 0) and (l != 1):
            func = (
                lambda x, l, mu, nu, delta, k2, b, r: 2
                * (2 * r) ** (l - 1)
                * (4 * b * r) ** (3.0 / 2.0)
                * (np.sin(x) ** 2 - np.sin(x) ** 4) ** (0.25 * (mu - 1))
                * (delta + np.sin(x) ** 2) ** (0.5 * (nu - 1))
                * np.maximum(0, k2 - np.sin(x) ** 2) ** (3.0 / 2.0)
            )
        elif (mu == 1) and (l == 1):
            raise ValueError("This case is treated separately.")
        else:
            return 0
        res, _ = quad(
            func, -kappa / 2, kappa / 2, args=(l, mu, nu, delta, k2, b, r)
        )
        return np.where(cond, res, 0)

    mu = l - m
    nu = l + m
    order = 20
    b, r = np.meshgrid(np.linspace(0, 1e-2, 100), np.linspace(0.01, 1.5, 13))
    b = np.concatenate((b.flatten(), [1.1, 0.9, 1e-3 / 3]))
    r = np.concatenate((r.flatten(), [0.1, 0.1, 1 + 1e-3 / 3]))
    kappa0, _ = jax.vmap(kappas)(b, r)
    p_calc = jax.vmap(partial(p, order, l))(b, r, kappa0)[:, l**2 + l + m]
    p_expect = np.array(
        [_numerical_p(b_, r_, k_) for b_, r_, k_ in zip(b, r, kappa0)]
    )

    assert np.all(np.isfinite(p_calc)), f"n_nan = {np.sum(np.isnan(p_calc))}"
    np.testing.assert_allclose(
        p_calc, p_expect, atol=5e-5, err_msg=f"l={l}, m={m}, mu={mu}, nu={nu}"
    )
