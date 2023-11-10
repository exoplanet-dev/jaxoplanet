import math
from collections import defaultdict
from functools import partial

import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg
from jax.experimental.sparse import BCOO
from scipy.special import gamma, comb
from scipy.special import comb


try:
    from scipy.sparse import csc_array
except ImportError:
    # With older versions of scipy, the data structures were called "matrices"
    # not "arrays"; this allows us to support either.
    from scipy.sparse import csc_matrix as csc_array


def basis(lmax):
    matrix = scipy.sparse.linalg.spsolve(A2_inv(lmax), A1(lmax))
    if lmax > 0:
        return BCOO.from_scipy_sparse(matrix)
    else:
        return BCOO.fromdense(np.squeeze(matrix)[None, None])


def A1(lmax):
    return _A_impl(lmax, p_Y) * 2 / np.sqrt(np.pi)


def A2_inv(lmax):
    return _A_impl(lmax, p_G)


def _A_impl(lmax, func):
    n = (lmax + 1) ** 2
    data = []
    row_ind = []
    col_ind = []
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            idx, val = func(p, l, m, n)
            data.extend(val)
            row_ind.extend(idx)
            col_ind.extend([n] * len(idx))
            n += 1
    return csc_array((np.array(data), (row_ind, col_ind)), shape=(n, n))


def ptilde(n):
    l = math.floor(math.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        i = mu // 2
        j = nu // 2
        k = 0
    else:
        i = (mu - 1) // 2
        j = (nu - 1) // 2
        k = 1
    return (i, j, k)


def Alm(l, m):
    return math.sqrt(
        (2 - int(m == 0))
        * (2 * l + 1)
        * math.factorial(l - m)
        / (4 * math.pi * math.factorial(l + m))
    )


def Blmjk(l, m, j, k):
    a = l + m + k - 1
    b = -l + m + k - 1
    if (b < 0) and (b % 2 == 0):
        return 0
    else:
        ratio = gamma(0.5 * a + 1) / gamma(0.5 * b + 1)
    return (
        2**l
        * math.factorial(m)
        / (
            math.factorial(j)
            * math.factorial(k)
            * math.factorial(m - j)
            * math.factorial(l - m - k)
        )
        * ratio
    )


def Cpqk(p, q, k):
    return math.factorial(k // 2) / (
        math.factorial(q // 2)
        * math.factorial((k - p) // 2)
        * math.factorial((p - q) // 2)
    )


def Ylm(l, m):
    res = defaultdict(lambda: 0)
    A = Alm(l, abs(m))
    for j in range(int(m < 0), abs(m) + 1, 2):
        for k in range(0, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k + 1, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 0)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k)
                    )
        for k in range(1, l - abs(m) + 1, 2):
            B = Blmjk(l, abs(m), j, k)
            if not B:
                continue
            factor = A * B
            for p in range(0, k, 2):
                for q in range(0, p + 1, 2):
                    ind = (abs(m) - j + p - q, j + q, 1)
                    res[ind] += (
                        (-1) ** ((j + p - (m < 0)) // 2) * factor * Cpqk(p, q, k - 1)
                    )

    return dict(res)


def p_Y(p, l, m, n):
    del n
    indicies = []
    data = []
    for k, v in Ylm(l, m).items():
        if k not in p:
            continue
        indicies.append(p[k])
        data.append(v)
    indicies = np.array(indicies, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indicies)
    return indicies[idx], data[idx]


def gtilde(n):
    l = math.floor(math.sqrt(n))
    m = n - l * l - l
    mu = l - m
    nu = l + m
    if nu % 2 == 0:
        I = [mu // 2]
        J = [nu // 2]
        K = [0]
        C = [(mu + 2) // 2]
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
        I = [(mu - 5) // 2, (mu - 5) // 2, (mu - 1) // 2]
        J = [(nu - 1) // 2, (nu + 3) // 2, (nu - 1) // 2]
        K = [1, 1, 1]
        C = [(mu - 3) // 2, -(mu - 3) // 2, -(mu + 3) // 2]
    res = {}
    for i, j, k, c in zip(I, J, K, C):
        res[(i, j, k)] = c
    return res


def p_G(p, l, m, n):
    del l, m
    indicies = []
    data = []
    for k, v in gtilde(n).items():
        if k not in p:
            continue
        indicies.append(p[k])
        data.append(v)
    indicies = np.array(indicies, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indicies)
    return indicies[idx], data[idx]


def poly_basis(deg):
    N = (deg + 1) * (deg + 1)

    @partial(jnp.vectorize, signature=f"(),(),()->({N})")
    def impl(x, y, z):
        xarr = [None for _ in range(N)]
        yarr = [None for _ in range(N)]

        # Ensures we get `nan`s off the disk
        xterm = 1.0 + 0.0 * z
        yterm = 1.0 + 0.0 * z

        i0 = 0
        di0 = 3
        j0 = 0
        dj0 = 2
        for n in range(deg + 1):
            i = i0
            di = di0
            xarr[i] = xterm
            j = j0
            dj = dj0
            yarr[j] = yterm
            i = i0 + di - 1
            j = j0 + dj - 1
            while i + 1 < N:
                xarr[i] = xterm
                xarr[i + 1] = xterm
                di += 2
                i += di
                yarr[j] = yterm
                yarr[j + 1] = yterm
                dj += 2
                j += dj - 1
            xterm *= x
            i0 += 2 * n + 1
            di0 += 2
            yterm *= y
            j0 += 2 * (n + 1) + 1
            dj0 += 2

        assert all(v is not None for v in xarr)
        assert all(v is not None for v in yarr)

        inds = []
        n = 0
        for ell in range(deg + 1):
            for m in range(-ell, ell + 1):
                if (ell + m) % 2 != 0:
                    inds.append(n)
                n += 1

        p = jnp.array(xarr) * jnp.array(yarr)
        if len(inds):
            return p.at[np.array(inds)].multiply(z)
        else:
            return p

    return impl


def utilde(n):
    res = defaultdict(lambda: 1.0)
    L = []

    if n == 0:
        res[(0, 0, 0)] = 1.0
        return res

    # binomial coefficients for (1-z)^n and using z^2 = x^2 + y^2
    # for every even power of z
    for k in range(n + 1):
        L.append((-1) ** k * comb(n, k))

    res[(0, 0, 0)] = 1.0
    res[(0, 0, 1)] = -L[1]

    for j in range(2, n + 1):
        n_2 = j // 2
        for k in range(0, n_2 + 1):
            res[(2 * (n_2 - k), 2 * k, j % 2)] *= -L[j] * comb(n_2, k)

    return res


def u_p(p, l, m, n):
    # this is very similar to gtilde, might be a more general
    # way of doing the _A_impl function without dummy variables
    del l, m
    indicies = []
    data = []
    for k, v in utilde(n).items():
        if k not in p:
            continue
        indicies.append(p[k])
        data.append(v)
    indicies = np.array(indicies, dtype=int)
    data = np.array(data, dtype=float)
    idx = np.argsort(indicies)
    return indicies[idx], data[idx]


def U0(udeg, ydeg):
    assert udeg < ydeg
    n = (ydeg + 1) ** 2
    p = {ptilde(m): m for m in range(n)}
    P = np.zeros((udeg + 1, (ydeg + 1) ** 2))
    for i in range(udeg + 1):
        idxs, values = u_p(p, None, None, i)
        for j, v in zip(idxs, values):
            P[i, j] += v

    return P
