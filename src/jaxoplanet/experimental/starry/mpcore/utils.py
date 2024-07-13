from jaxoplanet.experimental.starry.mpcore import mp
import numpy as np


def to_numpy(x):
    array = np.array(x.tolist(), dtype=np.float64)
    if array.shape[1] == (1):
        return array.T[0]
    else:
        return array


def to_mp(x):
    import mpmath as mp

    return mp.matrix(x.tolist())


def diff_mp(mp_matrix, np_array):
    return to_numpy(to_mp(np_array) - mp.matrix(mp_matrix))


def kron_delta(m, n):
    return 1 if m == n else 0


def fac(n):
    if n == 0:
        return mp.mpf(1.0)
    else:
        try:
            return mp.fac(n)
        except ValueError:
            return mp.inf
