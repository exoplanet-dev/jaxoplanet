from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def dot_rotation_matrix(ydeg, x, y, z, theta, M):
    R = compute_rotation_matrices(ydeg, x, y, z, theta)
    result = []
    for ell in range(ydeg + 1):
        result.append(M[:, ell * ell : ell * ell + 2 * ell + 1] @ R[ell])
    return jnp.concatenate(result, axis=1)


def compute_rotation_matrices(ydeg, x, y, z, theta):
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    ra01 = x * y * (1 - c) - z * s
    ra02 = x * z * (1 - c) + y * s
    ra11 = c + y * y * (1 - c)
    ra12 = y * z * (1 - c) - x * s
    ra20 = z * x * (1 - c) - y * s
    ra21 = z * y * (1 - c) + x * s
    ra22 = c + z * z * (1 - c)

    cond_neg = jnp.isclose(ra22, -1.0)
    cond_pos = jnp.isclose(ra22, 1.0)
    cond_full = jnp.logical_or(cond_pos, cond_neg)
    sign = cond_neg.astype(int) - cond_pos.astype(int)
    cos_beta = ra22
    sin_beta = jnp.where(
        cond_neg,
        1 + ra22,
        jnp.where(
            cond_pos,
            1 - ra22,
            jnp.sqrt(1 - cos_beta * cos_beta),  # type: ignore
        ),  # TODO(dfm): handle NaNs in grad
    )
    norm1 = jnp.sqrt(ra20 * ra20 + ra21 * ra21)
    norm2 = jnp.sqrt(ra02 * ra02 + ra12 * ra12)
    cos_gamma = jnp.where(
        cond_full,
        ra11,
        -ra20 / norm1,  # TODO(dfm): handle NaNs in grad
    )
    sin_gamma = jnp.where(
        cond_full,
        sign * ra01,
        ra21 / norm1,  # TODO(dfm): handle NaNs in grad
    )
    cos_alpha = jnp.where(
        cond_full,
        -sign * ra22,
        ra02 / norm2,  # TODO(dfm): handle NaNs in grad
    )
    sin_alpha = jnp.where(
        cond_full,
        1 + sign * ra22,
        ra12 / norm2,  # TODO(dfm): handle NaNs in grad
    )

    return rotar(
        ydeg,
        cos_alpha,
        sin_alpha,
        cos_beta,
        sin_beta,
        cos_gamma,
        sin_gamma,
    )[1]


def rotar(ydeg, c1, s1, c2, s2, c3, s3):
    sqrt_2 = jnp.sqrt(2.0)

    D = []
    R = []

    # D[0]; R[0 ]
    D.append(jnp.ones((1, 1)))
    R.append(jnp.ones((1, 1)))
    if ydeg == 0:
        return D, R

    # D[1]
    D1_22 = 0.5 * (1 + c2)
    D1_21 = -s2 / sqrt_2
    D1_20 = 0.5 * (1 - c2)
    D1_12 = -D1_21
    D1_11 = D1_22 - D1_20
    D1_10 = D1_21
    D1_02 = D1_20
    D1_01 = D1_12
    D1_00 = D1_22
    D.append(
        jnp.array(
            [
                [D1_00, D1_01, D1_02],
                [D1_10, D1_11, D1_12],
                [D1_20, D1_21, D1_22],
            ]
        )
    )

    # R[1]
    cosag = c1 * c3 - s1 * s3
    cosamg = c1 * c3 + s1 * s3
    sinag = s1 * c3 + c1 * s3
    sinamg = s1 * c3 - c1 * s3
    R1_11 = D1_11
    R1_21 = sqrt_2 * D1_12 * c1
    R1_01 = sqrt_2 * D1_12 * s1
    R1_12 = sqrt_2 * D1_21 * c3
    R1_10 = -sqrt_2 * D1_21 * s3
    R1_22 = D1_22 * cosag - D1_20 * cosamg
    R1_20 = -D1_22 * sinag - D1_20 * sinamg
    R1_02 = D1_22 * sinag - D1_20 * sinamg
    R1_00 = D1_22 * cosag + D1_20 * cosamg
    R.append(
        jnp.array(
            [
                [R1_00, R1_01, R1_02],
                [R1_10, R1_11, R1_12],
                [R1_20, R1_21, R1_22],
            ]
        )
    )

    # TODO(dfm): handle NaNs in gradient
    tgbet2 = jnp.where(jnp.allclose(s2, 0.0), s2, (1 - c2) / s2)

    for ell in range(2, ydeg + 1):
        D_, R_ = dlmn(ell, s1, c1, c2, tgbet2, s3, c3, D)
        D.append(D_)
        R.append(R_)

    return D, R


def dlmn(ell, s1, c1, c2, tgbet2, s3, c3, D):
    iinf = 1 - ell
    isup = -iinf

    # Last row by recurrence (Eq. 19 and 20 in Alvarez Collado et al.)
    D_ = [[0 for _ in range(2 * ell + 1)] for _ in range(2 * ell + 1)]
    D_[2 * ell][2 * ell] = 0.5 * D[-1][isup + ell - 1, isup + ell - 1] * (1 + c2)
    for m in range(isup, iinf - 1, -1):
        D_[2 * ell][m + ell] = (
            -tgbet2 * jnp.sqrt((ell + m + 1) / (ell - m)) * D_[2 * ell][m + 1 + ell]
        )
    D_[2 * ell][0] = 0.5 * D[ell - 1][isup + ell - 1, -isup + ell - 1] * (1 - c2)

    # The rows of the upper quarter triangle of the D[l;m',m) matrix
    # (Eq. 21 in Alvarez Collado et al.)
    al = ell
    al1 = al - 1
    tal1 = al + al1
    ali = 1.0 / al1
    cosaux = c2 * al * al1
    for mp in range(ell - 1, -1, -1):
        amp = mp
        laux = ell + mp
        lbux = ell - mp
        aux = ali / jnp.sqrt(laux * lbux)
        cux = jnp.sqrt((laux - 1) * (lbux - 1)) * al
        for m in range(isup, iinf - 1, -1):
            am = m
            lauz = ell + m
            lbuz = ell - m
            auz = 1.0 / jnp.sqrt(lauz * lbuz)
            fact = aux * auz
            term = tal1 * (cosaux - am * amp) * D[-1][mp + ell - 1, m + ell - 1]
            if lbuz != 1 and lbux != 1:
                cuz = jnp.sqrt((lauz - 1) * (lbuz - 1))
                term = term - D[-2][mp + ell - 2, m + ell - 2] * cux * cuz
            D_[mp + ell][m + ell] = fact * term
        iinf += 1
        isup -= 1

    # The remaining elements of the D[l;m',m) matrix are calculated
    # using the corresponding symmetry relations:
    # reflection ---> ((-1)**(m-m')) D[l;m,m') = D[l;m',m), m'<=m
    # inversion ---> ((-1)**(m-m')) D[l;-m',-m) = D[l;m',m)
    sign = 1
    iinf = -ell
    isup = ell - 1
    for m in range(ell, 0, -1):
        for mp in range(iinf, isup + 1):
            D_[mp + ell][m + ell] = sign * D_[m + ell][mp + ell]
            sign *= -1
        iinf += 1
        isup -= 1

    # Inversion
    iinf = -ell
    isup = iinf
    for m in range(ell - 1, -(ell + 1), -1):
        sign = -1
        for mp in range(isup, iinf - 1, -1):
            D_[mp + ell][m + ell] = sign * D_[-mp + ell][-m + ell]
            sign *= -1
        # iinf += 1
        isup += 1

    # Compute the real rotation matrices R from the complex ones D
    R_ = [[0 for _ in range(2 * ell + 1)] for _ in range(2 * ell + 1)]
    R_[ell][ell] = D_[ell][ell]
    cosmal = c1
    sinmal = s1
    sign = -1
    root_two = jnp.sqrt(2.0)
    for mp in range(1, ell + 1):
        cosmga = c3
        sinmga = s3
        aux = root_two * D_[ell][mp + ell]
        R_[mp + ell][ell] = aux * cosmal
        R_[-mp + ell][ell] = aux * sinmal
        for m in range(1, ell + 1):
            aux = root_two * D_[m + ell][ell]
            R_[ell][m + ell] = aux * cosmga
            R_[ell][-m + ell] = -aux * sinmga
            d1 = D_[-mp + ell][-m + ell]
            d2 = sign * D_[mp + ell][-m + ell]
            cosag = cosmal * cosmga - sinmal * sinmga
            cosagm = cosmal * cosmga + sinmal * sinmga
            sinag = sinmal * cosmga + cosmal * sinmga
            sinagm = sinmal * cosmga - cosmal * sinmga
            R_[mp + ell][m + ell] = d1 * cosag + d2 * cosagm
            R_[mp + ell][-m + ell] = -d1 * sinag + d2 * sinagm
            R_[-mp + ell][m + ell] = d1 * sinag + d2 * sinagm
            R_[-mp + ell][-m + ell] = d1 * cosag - d2 * cosagm
            aux = cosmga * c3 - sinmga * s3
            sinmga = sinmga * c3 + cosmga * s3
            cosmga = aux
        sign *= -1
        aux = cosmal * c1 - sinmal * s1
        sinmal = sinmal * c1 + cosmal * s1
        cosmal = aux

    return jnp.asarray(D_), jnp.asarray(R_)
