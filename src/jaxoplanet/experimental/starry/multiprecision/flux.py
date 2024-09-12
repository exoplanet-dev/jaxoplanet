from collections import defaultdict

from jaxoplanet.experimental.starry.multiprecision import mp
from jaxoplanet.experimental.starry.multiprecision.basis import A1, A2
from jaxoplanet.experimental.starry.multiprecision.rotation import (
    dot_rotation_matrix,
    dot_rz,
    get_R,
)
from jaxoplanet.experimental.starry.multiprecision.solution import get_sT, rT


def flux_function(l_max, inc, obl, cache=None):
    if cache is None:
        cache = defaultdict(
            lambda: {
                "R_obl": {},
                "R_inc": {},
                "sT": {},
            }
        )

    _rT = rT(l_max)
    _A1 = A1(l_max, cache=cache)
    _A2 = A2(l_max, cache=cache)

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
        y_rotated = rotate_y(y, phi, cache=cache)
        return ((_A1 @ y_rotated).T @ _rT)[0]

    def occ_flux(y, b, r, phi, rotate=True):
        y = mp.matrix(y.tolist())
        xo = 0.0
        yo = b
        theta_z = mp.atan2(xo, yo)
        _sT = get_sT(l_max, b, r, cache=cache)
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
