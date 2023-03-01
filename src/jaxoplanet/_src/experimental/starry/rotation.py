import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
from scipy.special import factorial

@jax.jit
def axis_to_euler(u1, u2, u3, theta):
    """Axis-angle rotation matrix"""
    tol = 1e-15
    costheta = jnp.cos(theta)
    sintheta = jnp.sin(theta)
    P01 = u1 * u2 * (1 - costheta) - u3 * sintheta
    P02 = u1 * u3 * (1 - costheta) + u2 * sintheta
    P11 = costheta + u2 * u2 * (1 - costheta)
    P12 = u2 * u3 * (1 - costheta) - u1 * sintheta
    P20 = u3 * u1 * (1 - costheta) - u2 * sintheta
    P21 = u3 * u2 * (1 - costheta) + u1 * sintheta
    P22 = costheta + u3 * u3 * (1 - costheta)
    sinbeta = jnp.sqrt(1 - P22 ** 2)
    norm1 = jnp.sqrt(P20 * P20 + P21 * P21)
    norm2 = jnp.sqrt(P02 * P02 + P12 * P12)
    
    alpha0 = jnp.arctan2(0., 1.)
    beta0 = jnp.arctan2(0., -1.)
    gamma0 = jnp.arctan2(P01, P11)
    
    alpha1 = jnp.arctan2(0., 1.)
    beta1 = jnp.arctan2(0., 1.)
    gamma1 = jnp.arctan2(-P01, P11)
    
    alpha2 = jnp.arctan2(P12 / norm2, P02 / norm2)
    beta2 = jnp.arctan2(jnp.sqrt(1 - P22 ** 2), P22)
    gamma2 = jnp.arctan2(P21 / norm1, -P20 / norm1)
    
    case1 = jnp.logical_and((P22 < -1 + tol), (P22 > -1 - tol))
    case2 = jnp.logical_and((P22 < 1 + tol), (P22 > 1 - tol))
    
    
    alpha = jnp.where(case1, alpha0, jnp.where(case2, alpha1, alpha2))
    beta = jnp.where(case1, beta0, jnp.where(case2, beta1, beta2))
    gamma = jnp.where(case1, gamma0, jnp.where(case2, gamma1, gamma2))
    
    return alpha, beta, gamma

def Rl(l):
    # U
    U = np.zeros((2*l+1, 2*l+1), dtype=np.complex_)
    Ud1 = np.ones(2*l+1)*1j
    Ud1[l+1::] = (-1)**np.arange(1, l+1)
    np.fill_diagonal(U, Ud1)
    np.fill_diagonal(np.fliplr(U), -1j*Ud1)
    U[l, l] = np.sqrt(2)
    U *= (1/np.sqrt(2))
    U = jnp.array(U)
    
    # dlm
    m, mp = np.indices((2*l+1, 2*l+1)) - l
    k = np.arange(0, 2*l+2)[:, None, None]
    
    @jax.jit
    def _Rl(alpha, beta, gamma):
        dlm = jnp.power(-1+0j, mp+m)*jnp.sqrt(factorial(l - m) * factorial(l + m) * factorial(l - mp) *\
        factorial(l + mp)) * (-1)**k*jnp.cos(beta/2)**(2*l+m-mp-2*k)*jnp.sin(beta/2)**(-m+mp+2*k)/ \
        (factorial(k)*factorial(l+m-k)*factorial(l-mp-k)*factorial(mp-m+k))

        dlm = jnp.nansum(dlm, 0)
        Dlm = jnp.exp(-1j*(mp*alpha + m*gamma)) *  dlm
        
        return jnp.real(jnp.linalg.inv(U)@Dlm@U)
    
    return _Rl

def R(lmax, u):
    Rls = [Rl(l) for l in range(lmax+1)]

    def _R(theta):
        a, b, g = axis_to_euler(u[0], u[1], u[2], theta)
        full = block_diag(*[rl(a, b, g) for rl in Rls])
        return jnp.where(theta != 0, full, jnp.eye(lmax*(lmax+2)+1))
    
    return _R