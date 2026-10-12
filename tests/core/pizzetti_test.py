# mypy: ignore-errors

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoplanet.core import pizzetti
from jaxoplanet.core.limb_dark import light_curve as agol_light_curve

LAWS = [[0.6], [0.4, 0.25], [0.4, 0.25, 0.1], [0.3, 0.2, 0.1, -0.05]]


def _x64() -> bool:
    return bool(jax.config.jax_enable_x64)  # type: ignore


def contact_points(r, n=801):
    d = np.geomspace(1e-12, 0.2, 10)
    z = np.concatenate(
        [
            np.linspace(0, 1 + r, n),
            abs(1 - r) + d,
            abs(1 - r) - d,
            1 + r - d,
            [r, 0.5 * (1 - r) if r < 1 else 0.0],
        ]
    )
    return np.sort(z[(z >= 0) & (z < 1 + r)])


@pytest.mark.parametrize("u", LAWS)
@pytest.mark.parametrize("r", [0.0092, 0.1, 0.3, 0.7, 1.5])
def test_matches_agol(u, r):
    """The deficit agrees with the solution of Agol et al. (2020) to within the
    requested tolerance relative to the depth (in float32, to its precision)"""
    z = contact_points(r)
    u = jnp.asarray(u)
    expect = np.asarray(agol_light_curve(u, z, r))
    calc = np.asarray(pizzetti.light_curve(u, z, r))
    depth = np.max(np.abs(expect))
    if _x64():
        np.testing.assert_allclose(calc, expect, atol=2e-10 * depth + 1e-14, rtol=0)
    else:
        np.testing.assert_allclose(calc, expect, atol=5e-5 * depth, rtol=0)


@pytest.mark.parametrize("tol", [1e-6, 1e-9, 1e-12])
def test_series_tol(tol):
    if not _x64():
        pytest.skip("requires float64")
    u = jnp.asarray([0.4, 0.25])
    r = 0.05
    z = contact_points(r, 3001)
    expect = np.asarray(agol_light_curve(u, z, r))
    calc = np.asarray(pizzetti.light_curve(u, z, r, series_tol=tol))
    depth = np.max(np.abs(expect))
    assert np.max(np.abs(calc - expect)) <= 1.1 * tol * depth + 1e-13


@pytest.mark.parametrize("u", [[0.4, 0.25], [0.3, 0.2, 0.1, -0.05]])
@pytest.mark.parametrize("r", [0.05, 0.2])
def test_against_mpmath(u, r):
    """An independent 40-digit reference (Green's theorem with adaptive quadrature),
    using neither the series nor any of the closed forms"""
    mpmath = pytest.importorskip("mpmath")
    if not _x64():
        pytest.skip("requires float64")
    from .pizzetti_reference import flux_ref

    z = contact_points(r, 9)
    c = np.asarray(pizzetti.mu_coefficients(jnp.asarray(u)))
    alphas = np.arange(c.shape[0])
    expect = np.array([flux_ref(zi, r, alphas, c) - 1.0 for zi in z])
    calc = np.asarray(pizzetti.light_curve(jnp.asarray(u), z, r))
    depth = np.max(np.abs(expect))
    assert mpmath is not None
    np.testing.assert_allclose(calc, expect, atol=2e-10 * depth + 1e-14, rtol=0)


@pytest.mark.parametrize("r", [0.01, 0.1, 0.3])
def test_truncation_bound_is_rigorous(r):
    """The truncated series is never below the exact disk mean, never above it by
    more than the bound, and the bound is tight (within a factor two)"""
    mpmath = pytest.importorskip("mpmath")
    if not _x64():
        pytest.skip("requires float64")
    from .pizzetti_reference import disk_mean_ref

    # I(mu) = mu: the deficit is -(3 r^2 / 2) D_{1/2}
    c = jnp.asarray([0.0, 1.0])
    for zf in (0.0, 0.3, 0.6, 0.9, 0.97):
        z = zf * (1 - r)
        exact = float(disk_mean_ref(z, r, 0.5))
        delta = float(pizzetti.interior_deficit(c, jnp.asarray(z), jnp.asarray(r)))
        series = -delta * 2 / (3 * r**2)
        bound = float(
            np.exp(pizzetti._log_flux_bound(c, jnp.asarray(z), jnp.asarray(r)))
        )
        err = series - exact
        assert mpmath is not None
        assert err >= -1e-14
        assert err <= bound * (1 + 1e-9) + 1e-14
        if zf <= 0.9 and err > 1e-12:
            assert bound / err < 2


@pytest.mark.parametrize("u", LAWS)
@pytest.mark.parametrize("r", [0.01, 0.1, 0.3])
def test_cut_is_the_largest_bounded_separation(u, r):
    if not _x64():
        pytest.skip("requires float64")
    tol = 1e-10
    c = pizzetti.mu_coefficients(jnp.asarray(u))
    z_c = float(pizzetti.series_cut(c, jnp.asarray(r), jnp.asarray(tol)))
    assert 0 < z_c < 1 - r
    log_tol = np.log(tol)
    below = pizzetti._log_flux_bound(c, jnp.asarray(z_c), jnp.asarray(r))
    above = pizzetti._log_flux_bound(c, jnp.asarray(z_c + 1e-9), jnp.asarray(r))
    assert float(below) <= log_tol < float(above)
    # the bound increases with z
    zs = jnp.linspace(1e-3, (1 - r) * 0.999, 200)
    bounds = jax.vmap(lambda z: pizzetti._log_flux_bound(c, z, jnp.asarray(r)))(zs)
    assert np.all(np.diff(np.asarray(bounds)) >= -1e-9)


def test_no_series_for_large_radius():
    c = pizzetti.mu_coefficients(jnp.asarray([0.4, 0.25]))
    assert float(pizzetti.series_cut(c, jnp.asarray(1.2), jnp.asarray(1e-10))) == -1.0
    assert float(pizzetti.series_cut(c, jnp.asarray(0.0), jnp.asarray(1e-10))) == -1.0


@pytest.mark.parametrize("u", [[0.4, 0.25], [0.3, 0.2, 0.1, -0.05]])
def test_gradients_match_agol(u):
    if not _x64():
        pytest.skip("requires float64")
    u = jnp.asarray(u)
    r = jnp.asarray(0.1)
    z = jnp.asarray(contact_points(0.1, 200))
    z = z[(z > 1e-3) & (z < 1.1)]

    def loss(fn):
        return lambda u, z, r: jnp.sum(fn(u, z, r) * jnp.cos(3.0 * z))

    expect = jax.grad(loss(agol_light_curve), argnums=(0, 1, 2))(u, z, r)
    calc = jax.grad(loss(pizzetti.light_curve), argnums=(0, 1, 2))(u, z, r)
    for a, b in zip(expect, calc, strict=True):
        a_, b_ = np.asarray(a), np.asarray(b)
        np.testing.assert_allclose(b_, a_, atol=1e-8 * np.max(np.abs(a_)), rtol=0)

    # forward mode
    tangent = jnp.ones_like(u)
    _, jvp_expect = jax.jvp(lambda u: agol_light_curve(u, z, r), (u,), (tangent,))
    _, jvp_calc = jax.jvp(lambda u: pizzetti.light_curve(u, z, r), (u,), (tangent,))
    np.testing.assert_allclose(
        jvp_calc, jvp_expect, atol=1e-8 * np.max(np.abs(jvp_expect)), rtol=0
    )


def test_gradients_are_finite_f32_and_f64():
    u = jnp.asarray([0.4, 0.25])
    z = jnp.asarray(contact_points(0.1, 300))
    grads = jax.grad(lambda u, z, r: jnp.sum(pizzetti.light_curve(u, z, r)), (0, 1, 2))(
        u, z, jnp.asarray(0.1)
    )
    assert all(np.all(np.isfinite(np.asarray(g))) for g in grads)


def test_order_independence():
    """Gathering the limb points must not depend on how the samples are ordered"""
    u = jnp.asarray([0.4, 0.25])
    r = 0.1
    z = np.abs(np.linspace(-1.5, 1.5, 5000))
    rng = np.random.default_rng(1234)
    perm = rng.permutation(z.shape[0])
    ordered = np.asarray(pizzetti.light_curve(u, z, r))
    shuffled = np.asarray(pizzetti.light_curve(u, z[perm], r))
    np.testing.assert_allclose(
        shuffled, ordered[perm], atol=1e-6 if not _x64() else 1e-13
    )


@pytest.mark.parametrize("n", [1, 3, 17, 64, 65, 1000])
def test_lengths(n):
    u = jnp.asarray([0.4, 0.25])
    z = np.linspace(0.0, 1.2, n)
    expect = agol_light_curve(u, z, 0.1)
    calc = pizzetti.light_curve(u, z, 0.1)
    assert calc.shape == (n,)
    np.testing.assert_allclose(
        calc,
        expect,
        atol=1e-5 * float(jnp.max(jnp.abs(expect))) if not _x64() else 1e-13,
    )


def test_shapes_and_broadcasting():
    u = jnp.asarray([0.4, 0.25])
    z = np.abs(np.linspace(-1.2, 1.2, 600)).reshape(20, 30)
    out = pizzetti.light_curve(u, z, 0.1)
    assert out.shape == z.shape
    np.testing.assert_allclose(
        out.reshape(-1), pizzetti.light_curve(u, z.reshape(-1), 0.1), atol=1e-6
    )

    # array of radii: elementwise evaluation, also under vmap
    r = jnp.linspace(0.02, 0.2, 30)
    out = pizzetti.light_curve(u, z, r)
    assert out.shape == z.shape
    expect = agol_light_curve(u, z, r)
    np.testing.assert_allclose(
        out,
        expect,
        atol=5e-5 * float(jnp.max(jnp.abs(expect))) if not _x64() else 2e-10,
    )
    out = jax.vmap(lambda ri: pizzetti.light_curve(u, z[0], ri))(r)
    np.testing.assert_allclose(
        out,
        jax.vmap(lambda ri: agol_light_curve(u, z[0], ri))(r),
        atol=1e-5 if not _x64() else 2e-10,
    )


def test_edge_cases():
    u = jnp.asarray([0.4, 0.25])
    # no occultation, no occultor, full occultation
    assert float(pizzetti.light_curve(u, jnp.asarray([2.0]), 0.1)[0]) == 0.0
    assert float(pizzetti.light_curve(u, jnp.asarray([0.3]), 0.0)[0]) == 0.0
    np.testing.assert_allclose(
        pizzetti.light_curve(u, jnp.asarray([0.1]), 1.5),
        agol_light_curve(u, jnp.asarray([0.1]), 1.5),
    )
    # centered occultor, negative separation and radius
    np.testing.assert_allclose(
        pizzetti.light_curve(u, jnp.asarray([0.0, -0.3]), -0.1),
        agol_light_curve(u, jnp.asarray([0.0, 0.3]), 0.1),
        atol=1e-5 if not _x64() else 2e-10,
    )
    # the uniform law, and laws beyond the order for which the bound is proven
    uniform = jnp.asarray([])
    z = jnp.linspace(0.0, 1.2, 50)
    np.testing.assert_allclose(
        pizzetti.light_curve(uniform, z, 0.1), agol_light_curve(uniform, z, 0.1)
    )
    high = jnp.full(pizzetti.MAX_COEFFS + 2, 0.01)
    np.testing.assert_allclose(
        pizzetti.light_curve(high, z, 0.1), agol_light_curve(high, z, 0.1)
    )


def test_method_dispatch():
    u = jnp.asarray([0.4, 0.25])
    z = jnp.linspace(0.0, 1.2, 200)
    np.testing.assert_allclose(
        agol_light_curve(u, z, 0.1, method="pizzetti"),
        pizzetti.light_curve(u, z, 0.1),
    )
    np.testing.assert_allclose(
        agol_light_curve(u, z, 0.1, method="agol"), agol_light_curve(u, z, 0.1)
    )
    with pytest.raises(ValueError, match="Unknown method"):
        agol_light_curve(u, z, 0.1, method="nope")
