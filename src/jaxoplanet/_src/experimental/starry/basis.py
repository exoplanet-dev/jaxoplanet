import math
from collections import defaultdict

import numpy as np
from scipy.special import gamma


def A1(lmax):
    """Note: The normalization here matches the starry paper, but not the
    code. To get the code's normalization, multiply the result by 2 /
    sqrt(pi).
    """
    n = (lmax + 1) ** 2
    res = np.zeros((n, n))
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            p_Y(p, l, m, res[:, n])
            n += 1
    return res


def A2_inv(lmax):
    n = (lmax + 1) ** 2
    res = np.zeros((n, n))
    p = {ptilde(m): m for m in range(n)}
    n = 0
    for l in range(lmax + 1):
        for _ in range(-l, l + 1):
            p_G(p, n, res[:, n])
            n += 1
    return res


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


def p_Y(p, l, m, res):
    for k, v in Ylm(l, m).items():
        if k not in p:
            continue
        res[p[k]] = v
    return res


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


def p_G(p, n, res):
    for k, v in gtilde(n).items():
        if k not in p:
            continue
        res[p[k]] = v
    return res
