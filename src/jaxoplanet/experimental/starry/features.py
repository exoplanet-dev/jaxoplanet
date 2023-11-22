import numpy as np
from scipy.special import legendre as LegendreP
from jax import numpy as jnp


def _Bp_expansion(l_max, pts=1000, eps=1e-9, smoothing=None):
    # From the starry_process paper (Luger 2022) Bplus equation C36
    # The default smoothing depends on `l_max`
    if smoothing is None:
        if l_max < 4:
            smoothing = 0.5
        else:
            smoothing = 2.0 / l_max

    # Pre-compute the linalg stuff
    theta = jnp.linspace(0, np.pi, pts)
    cost = jnp.cos(theta)
    B = jnp.hstack(
        [
            jnp.sqrt(2 * l + 1) * LegendreP(l)(cost).reshape(-1, 1)
            for l in range(l_max + 1)
        ]
    )
    A = jnp.linalg.solve(B.T @ B + eps * jnp.eye(l_max + 1), B.T)
    l = jnp.arange(l_max + 1)
    idxs = l * (l + 1)
    S = jnp.exp(-0.5 * idxs * smoothing**2)
    Bp = S[:, None] * A

    return Bp, theta, idxs


def spot_y(l_max, pts=1000, eps=1e-9, fac=300, smoothing=None):
    """Spot expansion

    Parameters
    ----------
    l_max : int
        degree of the spherical harmonics
    pts : int, optional
        _description_, by default 1000
    eps : float, optional
        _description_, by default 1e-9
    fac : int, optional
        _description_, by default 300
    smoothing : float, optional
        _description_, by default None

    Returns
    -------
    Callable
        spherical harmonics coefficients of the spot expansion
    """
    Bp, theta, idxs = _Bp_expansion(l_max, pts=pts, eps=eps, smoothing=smoothing)
    n_max = l_max**2 + 2 * l_max + 1

    def _y(contrast: float, radius: float):
        """Return the spherical harmonics coefficients of the spot expansion

        Parameters
        ----------
        contrast : float
            spot contrast
        radius : float
            spot radius

        Returns
        -------
        Array
            spherical harmonics coefficients of the spot expansion
        """

        # Compute unit-intensity spot at (0, 0)
        z = fac * (theta - radius)
        b = 1.0 / (1.0 + jnp.exp(-z)) - 1.0
        y = jnp.zeros(n_max)
        y = y.at[idxs].set(Bp @ b)
        y = y.at[0].set(1.0)
        return y * contrast

    return _y


def ring_y(l_max, pts=1000, eps=1e-9, smoothing=None):
    """Ring expansion

    Parameters
    ----------
    l_max : int
        degree of the spherical harmonics
    pts : int, optional
        _description_, by default 1000
    eps : float, optional
        _description_, by default 1e-9
    fac : int, optional
        _description_, by default 300
    smoothing : float, optional
        _description_, by default None

    Returns
    -------
    Callable
        spherical harmonics coefficients of the ring expansion
    """
    Bp, theta, idxs = _Bp_expansion(l_max, pts=pts, eps=eps, smoothing=smoothing)
    n_max = l_max**2 + 2 * l_max + 1

    def _y(contrast: float, width: float, latitude: float):
        b = 1 - jnp.array((theta > latitude - width) & (theta < latitude + width))
        y = jnp.zeros(n_max)
        y = y.at[idxs].set(Bp @ b)
        return y * contrast

    return _y
