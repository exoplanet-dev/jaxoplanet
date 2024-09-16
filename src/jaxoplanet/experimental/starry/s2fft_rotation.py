from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from s2fft.utils.rotation import generate_rotate_dls


@partial(jax.jit, static_argnums=(0, 5))
def compute_rotation_matrices(deg, x, y, z, theta, homogeneous=False):

    def kron(m, n):
        """Kronecker delta. Can we do better?"""
        return m == n

    def Umn(m, n):
        """Compute the (m, n) term of the transformation
        matrix from complex to real Ylms. This part is from starry"""
        if n < 0:
            term1 = 1j
        elif n == 0:
            term1 = jnp.sqrt(2) / 2
        else:
            term1 = 1
        if (m > 0) and (n < 0) and (n % 2 == 0):
            term2 = -1
        elif (m > 0) and (n > 0) and (n % 2 != 0):
            term2 = -1
        else:
            term2 = 1
        return (
            term1 * term2 * 1 / jnp.sqrt(2) * jnp.complex128(kron(m, n) + kron(m, -n))
        )

    def U(l):
        """Compute the complete U transformation matrix.. This part is from starry"""
        res = jnp.zeros((2 * l + 1, 2 * l + 1)).astype(jnp.complex128)
        for m in range(-l, l + 1):
            for n in range(-l, l + 1):
                res = res.at[m + l, n + l].set(Umn(m, n))
        return res

    def euler(x, y, z, theta):
        """axis-angle to euler angles"""
        # the jnp where for theta == 0 is to avoid nans when computing grad
        axis = jnp.array([jnp.where(theta == 0.0, 1.0, x), y, z])
        _theta = jnp.where(theta == 0.0, 1.0, theta)
        axis = axis / jnp.linalg.norm(axis)
        r = Rotation.from_rotvec(axis * _theta)
        return jnp.where(theta == 0.0, jnp.array([0.0, 0.0, 0.0]), r.as_euler("zyz"))

    alpha, beta, gamma = euler(x, y, z, theta)
    dls = generate_rotate_dls(deg + 1, beta)
    N = jnp.arange(-deg, deg + 1)
    m, mp = jnp.meshgrid(N, N)
    Dlm = jnp.exp(-1j * (m * alpha + mp * gamma)) * dls

    u = U(deg)
    u_inv = jnp.conj(u.T)

    # The output of generate_rotate_dls has shape (L, 2L+1, 2L+1)
    Rs_full = jnp.real(u_inv[None, :] @ Dlm @ u[None, :])

    if homogeneous:
        Rs = Rs_full
    else:
        # reformat to a list of matrices with different shape
        # (to match current implementation of rotation.dot_rotation_matrix)
        Rs = []

        for i in range(deg + 1):
            Rs.append(Rs_full[i, deg - i : deg + i + 1, deg - i : deg + i + 1])

    return Rs
