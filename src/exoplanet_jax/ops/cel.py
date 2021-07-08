# -*- coding: utf-8 -*-

__all__ = ["cel"]

import jax
import jax.numpy as jnp
import numpy as np


def cel(k2, p, a, b):
    k2, p, a, b = jnp.broadcast_arrays(k2, p, a, b)
    return _cel(k2, jnp.sqrt(1 - k2), p, a, b)


@jnp.vectorize
def _cel(k2, kc, p, a, b):
    eps = np.finfo(k2.dtype).eps
    ca = jnp.sqrt(eps)
    kc = jnp.maximum(kc, eps)

    q = k2 * (b - a * p)
    g = 1.0 - p
    f = g - k2
    ginv = 1.0 / g

    def pos(op):
        p, a, b, f, ginv, q = op
        p = jnp.sqrt(p)
        pinv = 1.0 / p
        a = a
        b = b * pinv
        return p, pinv, a, b

    def neg(op):
        p, a, b, f, ginv, q = op
        p = jnp.sqrt(f * ginv)
        pinv = 1.0 / p
        a = (a - b) * ginv
        b = -q * jnp.square(ginv) * pinv + a * p
        return p, pinv, a, b

    p, pinv, a, b = jax.lax.cond(
        jnp.greater(p, 0),
        pos,
        neg,
        operand=(p, a, b, f, ginv, q),
    )

    ee = kc
    f = a
    a += b * pinv
    g = ee * pinv
    b += f * g
    b += b
    p += g
    g = jnp.ones_like(kc)
    m = 1.0 + kc

    def loop_cond(args):
        i, kc, p, a, b, ee, f, g, m = args
        return jnp.logical_and(jnp.abs(g - kc) > g * ca, i < 50)

    def loop_body(args):
        i, kc, p, a, b, ee, f, g, m = args
        kc = jnp.sqrt(ee)
        kc += kc
        ee = kc * m
        f = a
        pinv = 1.0 / p
        a += b * pinv
        g = ee * pinv
        b += f * g
        b += b
        p += g
        g = m
        m += kc

        return i + 1, kc, p, a, b, ee, f, g, m

    _, _, p, a, b, _, _, _, m = jax.lax.while_loop(
        loop_cond, loop_body, (0, kc, p, a, b, ee, f, g, m)
    )

    return 0.5 * jnp.pi * (a * m + b) / (m * (m + p))
