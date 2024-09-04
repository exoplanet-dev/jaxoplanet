from jaxoplanet.experimental.starry.multiprecision import mp
from jaxoplanet.experimental.starry.multiprecision.utils import (
    fac as fac_function,
    kron_delta,
)

global FAC_CACHE
FAC_CACHE = {}


def fac(n):
    if n not in FAC_CACHE:
        FAC_CACHE[n] = fac_function(n)

    return FAC_CACHE[n]


def A(l, m):
    """A spherical harmonic normalization constant."""
    return mp.sqrt(
        (2 - kron_delta(m, 0)) * (2 * l + 1) * fac(l - m) / (4 * mp.pi * fac(l + m))
    )


def B(l, m, j, k):
    """Another spherical harmonic normalization constant."""

    ratio = fac((l + m + k - 1) / 2) / fac((-l + m + k - 1) / 2)

    return 2**l * fac(m) / (fac(j) * fac(k) * fac(m - j) * fac(l - m - k)) * ratio


def C(p, q, k):
    """Return the binomial theorem coefficient `C`."""
    return fac(k / 2) / (fac(q / 2) * fac((k - p) / 2) * fac((p - q) / 2))


from collections import defaultdict


def Y(l, m):
    """Return the spherical harmonic of degree `l` and order `m`."""
    coeffs = defaultdict(lambda: {})

    def get(function, *indices):
        fun_name = function.__name__
        if indices not in coeffs[fun_name]:
            coeffs[fun_name][indices] = function(*indices)

        return coeffs[fun_name][indices]

    res = defaultdict(lambda: 0)
    if m >= 0:
        for j in range(0, m + 1, 2):
            for k in range(0, l - m + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p) // 2)
                            * get(A, l, m)
                            * get(B, l, m, j, k)
                            * get(C, p, q, k)
                        )
                        x_order = m - j + p - q
                        y_order = j + q
                        z_order = 0
                        res[(x_order, y_order, z_order)] += coeff
            for k in range(1, l - m + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p) // 2)
                            * get(A, l, m)
                            * get(B, l, m, j, k)
                            * get(C, p, q, k - 1)
                        )
                        x_order = m - j + p - q
                        y_order = j + q
                        z_order = 1
                        res[(x_order, y_order, z_order)] += coeff
    else:
        for j in range(1, abs(m) + 1, 2):
            for k in range(0, l - abs(m) + 1, 2):
                for p in range(0, k + 1, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p - 1) // 2)
                            * get(A, l, abs(m))
                            * get(B, l, abs(m), j, k)
                            * get(C, p, q, k)
                        )
                        x_order = abs(m) - j + p - q
                        y_order = j + q
                        z_order = 0
                        res[(x_order, y_order, z_order)] += coeff
            for k in range(1, l - abs(m) + 1, 2):
                for p in range(0, k, 2):
                    for q in range(0, p + 1, 2):
                        coeff = (
                            (-1) ** ((j + p - 1) // 2)
                            * get(A, l, abs(m))
                            * get(B, l, abs(m), j, k)
                            * get(C, p, q, k - 1)
                        )
                        x_order = abs(m) - j + p - q
                        y_order = j + q
                        z_order = 1
                        res[(x_order, y_order, z_order)] += coeff

    return res


def p_coeffs(n):
    l = mp.floor(mp.sqrt(n))
    m = n - l * l - l
    mu = int(l - m)
    nu = int(l + m)
    if nu % 2 == 0:
        i = mu // 2
        j = nu // 2
        k = 0
    else:
        i = (mu - 1) // 2
        j = (nu - 1) // 2
        k = 1
    return (i, j, k)


def A1(l_max):
    n = (l_max + 1) ** 2
    p = {m: p_coeffs(m) for m in range(n)}
    res = mp.zeros(n, n)

    k = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            y = Y(l, m)
            res[:, k] = mp.matrix([y[p[i]] for i in range(n)])
            k += 1

    res = res * 2 / mp.sqrt(mp.pi)
    return res


def gtilde(n):
    l = mp.floor(mp.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        I = [mp.floor(mu / 2)]
        J = [mp.floor(nu / 2)]
        K = [0]
        C = [mp.floor((mu + 2) / 2)]
    elif (l == 1) and (m == 0):
        I = [0]
        J = [0]
        K = [1]
        C = [1]
    elif (mu == 1) and (l % 2 == 0):
        I = [l - 2]
        J = [1]
        K = [1]
        C = [3]
    elif mu == 1:
        I = [l - 3, l - 1, l - 3]
        J = [0, 0, 2]
        K = [1, 1, 1]
        C = [-1, 1, 4]
    else:
        I = [mp.floor((mu - 5) / 2), mp.floor((mu - 5) / 2), mp.floor((mu - 1) / 2)]
        J = [mp.floor((nu - 1) / 2), mp.floor((nu + 3) / 2), mp.floor((nu - 1) / 2)]
        K = [1, 1, 1]
        C = [mp.floor((mu - 3) / 2), -mp.floor((mu - 3) / 2), -mp.floor((mu + 3) / 2)]
    res = {}
    for i, j, k, c in zip(I, J, K, C, strict=False):
        res[(i, j, k)] = c
    return res


def A2_inv(l_max):
    n = (l_max + 1) ** 2
    p = {m: p_coeffs(m) for m in range(n)}
    res = mp.zeros(n, n)

    k = 0
    for l in range(l_max + 1):
        for _ in range(-l, l + 1):
            y = gtilde(k)
            res[:, k] = mp.matrix([y[p[i]] if p[i] in y else 0.0 for i in range(n)])
            k += 1

    return res
