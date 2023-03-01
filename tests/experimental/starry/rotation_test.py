import numpy as np
import pytest
from jax import config
from jaxoplanet._src.experimental.starry.rotation import R

config.update("jax_enable_x64", True)


@pytest.mark.parametrize("lmax", [10, 7, 5, 4, 3, 2, 1, 0])
@pytest.mark.parametrize("u", [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.1, 0)])
def test_R(lmax, u):
    pytest.importorskip("sympy")
    theta = np.pi / 4
    expected = np.array(R_symbolic(lmax, u, theta)).astype(float)
    calc = R(lmax, u)(np.array([theta]))[0]
    np.testing.assert_allclose(calc, expected, atol=5e-12)


def R_symbolic(lmax, u, theta):
    import sympy as sm
    from sympy.functions.special.tensor_functions import KroneckerDelta

    def Dmn(l, m, n, alpha, beta, gamma):
        """Compute the (m, n) term of the Wigner D matrix."""
        sumterm = 0
        # Expression diverges when beta = 0
        if beta == 0:
            beta = 1e-16
        for k in range(l + m + 1):
            sumterm += (
                (-1) ** k
                * sm.cos(beta / 2) ** (2 * l + m - n - 2 * k)
                * sm.sin(beta / 2) ** (-m + n + 2 * k)
                / (
                    sm.factorial(k)
                    * sm.factorial(l + m - k)
                    * sm.factorial(l - n - k)
                    * sm.factorial(n - m + k)
                )
            )
        return (
            sumterm
            * sm.exp(-sm.I * (alpha * n + gamma * m))
            * (-1) ** (n + m)
            * sm.sqrt(
                sm.factorial(l - m)
                * sm.factorial(l + m)
                * sm.factorial(l - n)
                * sm.factorial(l + n)
            )
        )

    def D(l, alpha, beta, gamma):
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Dmn(l, m, n, alpha, beta, gamma)
        return res

    def Umn(l, m, n):
        """Compute the (m, n) term of the transformation matrix from complex to real Ylms."""
        if n < 0:
            term1 = sm.I
        elif n == 0:
            term1 = sm.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return (
            term1
            * term2
            * 1
            / sm.sqrt(2)
            * (KroneckerDelta(m, n) + KroneckerDelta(m, -n))
        )

    def U(l):
        """Compute the U transformation matrix."""
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res[m + l, n + l] = Umn(l, m, n)
        return res

    def REuler(l, alpha, beta, gamma):
        """Return the rotation matrix for a single degree `l`."""
        res = sm.zeros(2 * l + 1, 2 * l + 1)
        if l == 0:
            res[0, 0] = 1
            return res
        foo = sm.re(U(l).inv() * D(l, alpha, beta, gamma) * U(l))
        for m in range(2 * l + 1):
            for n in range(2 * l + 1):
                if abs(foo[m, n]) < 1e-15:
                    res[m, n] = 0
                else:
                    res[m, n] = foo[m, n]
        return res

    def RAxisAngle(l, u1, u2, u3, theta):
        """Axis-angle rotation matrix."""
        # Numerical tolerance
        tol = 1e-16
        if theta == 0:
            theta = tol
        if u1 == 0 and u2 == 0:
            u1 = tol
            u2 = tol

        # Elements of the transformation matrix
        costheta = sm.cos(theta)
        sintheta = sm.sin(theta)
        RA01 = u1 * u2 * (1 - costheta) - u3 * sintheta
        RA02 = u1 * u3 * (1 - costheta) + u2 * sintheta
        RA11 = costheta + u2 * u2 * (1 - costheta)
        RA12 = u2 * u3 * (1 - costheta) - u1 * sintheta
        RA20 = u3 * u1 * (1 - costheta) - u2 * sintheta
        RA21 = u3 * u2 * (1 - costheta) + u1 * sintheta
        RA22 = costheta + u3 * u3 * (1 - costheta)

        # Determine the Euler angles
        if (RA22 < -1 + tol) and (RA22 > -1 - tol):
            cosbeta = -1
            sinbeta = 0
            cosgamma = RA11
            singamma = RA01
            cosalpha = 1
            sinalpha = 0
        elif (RA22 < 1 + tol) and (RA22 > 1 - tol):
            cosbeta = 1
            sinbeta = 0
            cosgamma = RA11
            singamma = -RA01
            cosalpha = 1
            sinalpha = 0
        else:
            cosbeta = RA22
            sinbeta = sm.sqrt(1 - cosbeta**2)
            norm1 = sm.sqrt(RA20 * RA20 + RA21 * RA21)
            norm2 = sm.sqrt(RA02 * RA02 + RA12 * RA12)
            cosgamma = -RA20 / norm1
            singamma = RA21 / norm1
            cosalpha = RA02 / norm2
            sinalpha = RA12 / norm2
        alpha = sm.atan2(sinalpha, cosalpha)
        beta = sm.atan2(sinbeta, cosbeta)
        gamma = sm.atan2(singamma, cosgamma)

        return REuler(l, alpha, beta, gamma)

    def R(lmax, u, theta):
        """Return the full axis-angle rotation matrix up to degree `lmax`."""
        blocks = [RAxisAngle(l, *u, theta) for l in range(lmax + 1)]
        return sm.BlockDiagMatrix(*blocks)

    return R(lmax, u, theta)
