from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("eps",))
def light_curve(c, alpha, b, r, eps=1e-12):
    I_0 = (alpha + 2) / (jnp.pi * (alpha - c * alpha + 2))
    g = 0.5 * alpha
    b = jnp.abs(b)
    r = jnp.abs(r)

    def q1(z, p, c, alpha):
        zt = jnp.clip(abs(z), 0, 1 - p)
        s = 1 - zt**2
        c0 = 1 - c + c * s**g
        c2 = 0.5 * alpha * c * s ** (g - 2) * ((alpha - 1) * zt**2 - 1)
        return 1 - I_0 * jnp.pi * p**2 * (
            c0 + 0.25 * p**2 * c2 - 0.125 * alpha * c * p**2 * s ** (g - 1)
        )

    def q2(z, p, c, alpha):
        zt = jnp.clip(abs(z), 1 - p, 1 + p)
        d = jnp.clip((zt**2 - p**2 + 1) / (2 * zt), 0, 1)
        ra = 0.5 * (zt - p + d)
        rb = 0.5 * (1 + d)
        sa = jnp.clip(1 - ra**2, eps, 1)
        sb = jnp.clip(1 - rb**2, eps, 1)
        q = jnp.clip((zt - d) / p, -1, 1)
        w2 = p**2 - (d - zt) ** 2
        w = jnp.sqrt(jnp.clip(w2, eps, 1))
        b0 = 1 - c + c * sa**g
        b1 = -alpha * c * ra * sa ** (g - 1)
        b2 = 0.5 * alpha * c * sa ** (g - 2) * ((alpha - 1) * ra**2 - 1)
        a0 = b0 + b1 * (zt - ra) + b2 * (zt - ra) ** 2
        a1 = b1 + 2 * b2 * (zt - ra)
        aq = jnp.arccos(q)
        J1 = (
            a0 * (d - zt)
            - (2 / 3) * a1 * w2
            + 0.25 * b2 * (d - zt) * (2 * (d - zt) ** 2 - p**2)
        ) * w + (a0 * p**2 + 0.25 * b2 * p**4) * aq
        J2 = (
            alpha
            * c
            * sa ** (g - 1)
            * p**4
            * (
                0.125 * aq
                + (1 / 12) * q * (q**2 - 2.5) * jnp.sqrt(jnp.clip(1 - q**2, 0, 1))
            )
        )
        d0 = 1 - c + c * sb**g
        d1 = -alpha * c * rb * sb ** (g - 1)
        K1 = (d0 - rb * d1) * jnp.arccos(d) + (
            (rb * d + (2 / 3) * (1 - d**2)) * d1 - d * d0
        ) * jnp.sqrt(jnp.clip(1 - d**2, 0, 1))

        K2 = (1 / 3) * c * alpha * sb ** (g + 0.5) * (1 - d)
        return 1 - I_0 * (J1 - J2 + K1 - K2)

    return (
        jnp.select(
            [b <= (1 - r), abs(b - 1) < r],
            [q1(b, r, c, alpha), q2(b, r, c, alpha)],
            default=1,
        )
        - 1.0
    )
