# -*- coding: utf-8 -*-

__all__ = ["cel"]

import jax
import jax.numpy as jnp
import numpy as np


@jax.custom_jvp
def cel(k2, p, a, b):
    return _cel(k2, jnp.sqrt(1 - k2), p, a, b)


@cel.defjvp
def _cel_jvp(primals, tangents):
    k2, p, a, b = primals
    k2_dot, p_dot, a_dot, b_dot = tangents

    kc = jnp.sqrt(1 - k2)
    kc2 = jnp.square(kc)
    ap = a * p
    bp = b * p
    lam = kc2 * (b + ap - 2 * bp) + p * (3 * bp - ap * p - 2 * b)

    primal_out = _cel(k2, kc, p, a, b)

    dk = -kc * (_cel(k2, kc, kc2, a, b) - primal_out) / (p - kc2)
    dk2 = -0.5 * dk / kc
    dp = (
        _cel(k2, kc, p, jnp.zeros_like(a), lam)
        + (b - ap) * _cel(k2, kc, jnp.ones_like(p), 1 - p, kc2 - p)
    ) / (2 * p * (1 - p) * (p - kc2))
    da = _cel(k2, kc, p, jnp.ones_like(a), jnp.zeros_like(b))
    db = _cel(k2, kc, p, jnp.zeros_like(a), jnp.ones_like(b))

    tangent_out = k2_dot * dk2 + p_dot * dp + a_dot * da + b_dot * db

    return primal_out, tangent_out


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
