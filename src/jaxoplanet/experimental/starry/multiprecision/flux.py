from collections import defaultdict

from jaxoplanet.experimental.starry.multiprecision import mp
from jaxoplanet.experimental.starry.multiprecision.basis import A1, A2
from jaxoplanet.experimental.starry.multiprecision.rotation import (
    R,
    dot_rotation_matrix,
    dot_rz,
)
from jaxoplanet.experimental.starry.multiprecision.solution import rT, sT

CACHED_MATRICES = defaultdict(
    lambda: {
        "R_obl": {},
        "R_inc": {},
    }
)


def get_A(A, l_max, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    name = A.__name__
    if name not in cache[l_max]:
        cache[l_max][name] = A(l_max)
    return cache[l_max][name]


def get_R(name, l_max, obl=None, inc=None, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    if obl is not None and inc is not None:
        if name == "R_obl":
            if obl not in CACHED_MATRICES[l_max][name]:
                CACHED_MATRICES[l_max][name][obl] = R(l_max, (0.0, 0.0, 1.0), -obl)
            return CACHED_MATRICES[l_max][name][obl]
        elif name == "R_inc":
            if (inc, obl) not in CACHED_MATRICES[l_max][name]:
                CACHED_MATRICES[l_max][name][(inc, obl)] = R(
                    l_max, (-mp.cos(obl), -mp.sin(obl), 0.0), (0.5 * mp.pi - inc)
                )
            return CACHED_MATRICES[l_max][name][(inc, obl)]
    else:
        if name not in CACHED_MATRICES[l_max]:
            if name == "R_px":
                CACHED_MATRICES[l_max][name] = R(l_max, (1.0, 0.0, 0.0), -0.5 * mp.pi)
            elif name == "R_mx":
                CACHED_MATRICES[l_max][name] = R(l_max, (1.0, 0.0, 0.0), 0.5 * mp.pi)

        return CACHED_MATRICES[l_max][name]


def get_sT(l_max, b, r, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    br = (b, r)
    if br not in CACHED_MATRICES[l_max]:
        CACHED_MATRICES[l_max][br] = sT(l_max, b, r)
    return CACHED_MATRICES[l_max][br]


def flux_function(l_max, inc, obl, cache=None):
    if cache is None:
        cache = CACHED_MATRICES

    _rT = rT(l_max)
    _A1 = get_A(A1, l_max, cache=cache)
    _A2 = get_A(A2, l_max, cache=cache)

    R_px = get_R("R_px", l_max, cache=cache)
    R_mx = get_R("R_mx", l_max, cache=cache)
    R_obl = get_R("R_obl", l_max, obl=obl, inc=inc, cache=cache)
    R_inc = get_R("R_inc", l_max, obl=obl, inc=inc, cache=cache)

    def rotate_y(y, phi):
        y_rotated = dot_rotation_matrix(l_max, None, None, R_px)(y)
        y_rotated = dot_rz(l_max, phi)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_mx)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_obl)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_inc)(y_rotated)
        return y_rotated

    def rot_flux(y, phi):
        y = mp.matrix(y.tolist())
        y_rotated = rotate_y(y, phi)
        return ((_A1 @ y_rotated).T @ _rT)[0]

    def occ_flux(y, b, r, phi, rotate=True):
        y = mp.matrix(y.tolist())
        xo = 0.0
        yo = b
        theta_z = mp.atan2(xo, yo)
        _sT = get_sT(l_max, b, r)
        x = _sT.T @ _A2
        if rotate:
            y_rotated = rotate_y(y, phi)
            y_rotated = dot_rz(l_max, theta_z)(y_rotated)
        else:
            y_rotated = y
        py = (_A1 @ y_rotated).T
        return py @ x.T

    def impl(y, b, r, phi, rotate=True):
        if abs(b) >= (r + 1):
            return rot_flux(y, phi)
        elif r > 1 and abs(b) <= (r - 1):
            return mp.mpf(0.0)
        else:
            return occ_flux(y, b, r, phi, rotate=rotate)[0]

    return impl
