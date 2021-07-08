# -*- coding: utf-8 -*-

__all__ = ["kite_area"]

import jax.numpy as jnp
import numpy as np
from jax import lax


def kite_area(p0, p1, p2):
    p = jnp.sort(jnp.stack(jnp.broadcast_arrays(p0, p1, p2), axis=-1), axis=-1)
    A = p[..., 2]
    B = p[..., 1]
    C = p[..., 0]
    square_area = (A + (B + C)) * (C - (A - B)) * (C + (A - B)) * (A + (B - C))
    return jnp.sqrt(jnp.maximum(square_area, 0.0))


def s0(b, r):
    area = kite_area(1.0, b, r)
    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    kcsq = -0.25 * (1.0 + bpr) * (1 - bpr) / (r * b)
    kap0 = jnp.arctan2(area, (r - 1.0) * (r + 1.0) + b2)
    kap1 = jnp.arctan2(area, (1.0 - r) * (1.0 + r) + b2)
    Alens = kap1 + r2 * kap0 - area * 0.5
    return jnp.where(
        b < r - 1.0,  # Complete occultation
        0.0,
        jnp.where(
            jnp.isclose(r, 0.0) | (b > r + 1.0),  # No occultation
            np.pi,
            jnp.where(
                jnp.isclose(b, 0.0) | (kcsq < 0.0),
                np.pi * (1 - r2),
                np.pi - Alens,
            ),
        ),
    )


def s0_select(b, r):
    area = kite_area(1.0, b, r)
    b2 = jnp.square(b)
    r2 = jnp.square(r)
    bpr = b + r
    kcsq = -0.25 * (1.0 + bpr) * (1 - bpr) / (r * b)
    kap0 = jnp.arctan2(area, (r - 1.0) * (r + 1.0) + b2)
    kap1 = jnp.arctan2(area, (1.0 - r) * (1.0 + r) + b2)
    Alens = kap1 + r2 * kap0 - area * 0.5

    return jnp.select(
        [
            b < r - 1.0,  # Complete occultation
            jnp.isclose(r, 0.0) | (b > r + 1.0),  # No occultation
            jnp.isclose(b, 0.0) | (kcsq < 0.0),
            True,
        ],
        [0.0, np.pi, np.pi * (1 - r2), np.pi - Alens],
    )


def s0_scal(b, r):
    def body_func(b, r):
        area = kite_area(1.0, b, r)
        b2 = jnp.square(b)
        r2 = jnp.square(r)
        kap0 = jnp.arctan2(area, (r - 1.0) * (r + 1.0) + b2)
        kap1 = jnp.arctan2(area, (1.0 - r) * (1.0 + r) + b2)
        Alens = kap1 + r2 * kap0 - area * 0.5
        return np.pi - Alens

    def inner_cond(b, r):
        bpr = b + r
        kcsq = -0.25 * (1.0 + bpr) * (1 - bpr) / (r * b)
        return jnp.isclose(b, 0.0) | (kcsq < 0.0)

    return lax.cond(
        b < r - 1.0,
        lambda args: jnp.zeros_like(args[0]),
        lambda args: lax.cond(
            jnp.isclose(args[1], 0.0) | (args[0] > args[1] + 1.0),
            lambda args: jnp.full_like(args[0], np.pi),
            lambda args: lax.cond(
                inner_cond(*args),
                lambda args: np.pi * (1.0 - jnp.square(args[1])),
                lambda args: body_func(*args),
                operand=args,
            ),
            operand=args,
        ),
        operand=(b, r),
    )
