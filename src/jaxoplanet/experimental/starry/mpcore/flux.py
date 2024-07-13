from jaxoplanet.experimental.starry.mpcore import mp
from jaxoplanet.experimental.starry.mpcore.basis import A1, A2inv
from jaxoplanet.experimental.starry.mpcore.solution import rT, sT
from jaxoplanet.experimental.starry.mpcore.rotation import (
    R,
    dot_rotation_matrix,
    dot_rz,
)


def flux_function(l_max, inc, obl):
    _rT = rT(l_max)
    _A1 = A1(l_max)
    _A2 = A2inv(l_max) ** -1

    R_px = R(l_max, (1.0, 0.0, 0.0), -0.5 * mp.pi)
    R_mx = R(l_max, (1.0, 0.0, 0.0), 0.5 * mp.pi)
    R_obl = R(l_max, (0.0, 0.0, 1.0), -obl)
    R_inc = R(l_max, (-mp.cos(obl), -mp.sin(obl), 0.0), (0.5 * mp.pi - inc))

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

    def occ_flux(y, b, r, phi):
        y = mp.matrix(y.tolist())
        xo = 0.0
        yo = b
        zo = 10
        theta_z = mp.atan2(xo, yo)
        _sT = sT(l_max, b, r)
        x = _sT.T @ _A2
        y_rotated = rotate_y(y, phi)
        y_rotated = dot_rz(l_max, theta_z)(y_rotated)
        py = (_A1 @ y_rotated).T
        return py @ x.T

    def impl(y, b, r, phi):
        if abs(b) >= (r + 1):
            return rot_flux(y, phi)
        else:
            if r > 1 and abs(b) <= (r - 1):
                return mp.mpf(0.0)
            else:
                return occ_flux(y, b, r, phi)[0]

    return impl
