from jaxoplanet._src.experimental.starry.basis import Y
import numpy as np


def lm(n: int) -> tuple:
    """degree and order of n-th spherical harmonics
    coefficient

    Parameters
    ----------
    n : int
        _description_

    Returns
    -------
    tuple
        l, m
    """
    l = int(np.floor(np.sqrt(n)))
    m = int(n - l**2 - l)
    return l, m


def n(l: int, m: int) -> int:
    """flattened index of the spherical harmonics of
    degree l and order m

    Parameters
    ----------
    l : int
        degree
    m : int
        order

    Returns
    -------
    int
        n
    """
    return l**2 + l + m


def Y_basis(l_max, n=400):
    """Spherical harmonics basis projection on the xy plane

    Parameters
    ----------
    l_max : _type_
        maximum degree
    n : int, optional
        width of the basis in pixels, by default 400

    Returns
    -------
    np.array
        spherical harmonics basis
    """
    coeffs = [Y(*lm(i)) for i in range(l_max * (l_max + 2) + 1)]
    pixels = np.linspace(-1, 1, n)
    x, y = np.meshgrid(pixels, pixels)
    mask = x**2 + y**2 < 1
    x[~mask] = np.nan
    y[~mask] = np.nan
    z = np.sqrt(1 - x**2 - y**2)
    B = np.array(
        [
            np.sum([c * x**i * y**j * z**k for (i, j, k), c in b.items()], 0)
            for b in coeffs
        ]
    )
    return B
