# -*- coding: utf-8 -*-

__all__ = ["kite_area"]

import jax.numpy as jnp
from jax import lax

from .cel import cel


def soln(b, r):
    b, r = jnp.broadcast_arrays(b, r)

    # Otherwise case
    inds = jnp.full_like(b, 3, dtype=jnp.int_)

    # Special case for b=0
    inds = jnp.where(jnp.isclose(b, 0), 2, inds)

    def zero_b(op):
        _, r = op
        r2 = jnp.square(r)
        onemr2 = 1 - r2
        return (
            jnp.pi * onemr2,
            2 * jnp.pi * onemr2 * jnp.sqrt(onemr2) / 3.0,
            -2 * jnp.pi * onemr2 * r2,
        )

    # No occultation
    inds = jnp.where(
        jnp.logical_or(jnp.isclose(r, 0), jnp.greater(b, r + 1)), 1, inds
    )

    def none(op):
        b, _ = op
        return jnp.full_like(b, jnp.pi), jnp.full_like(2 * jnp.pi / 3, b), zero

    # Complete occultation
    inds = jnp.where(jnp.less(b, r - 1), 0, inds)

    def complete(op):
        b, _ = op
        zero = jnp.zeros_like(b)
        return zero, zero, zero

    return jnp.vectorize(lax.switch, excluded=(1,))(
        inds, [complete, none, zero_b, complete], (b, r)
    )

    conds = [
        jnp.less(b, r - 1),
        jnp.logical_or(jnp.isclose(r, 0), jnp.greater(b, r + 1)),
        jnp.isclose(b, 0),
    ]

    zero = jnp.zeros_like(b)

    def complete(_):
        return zero, zero, zero

    def none(_):
        return jnp.pi, 2 * jnp.pi / 3, zero

    return lax.cond(
        jnp.less(b, r - 1),
        complete,
        lambda _: lax.cond(
            jnp.logical_or(jnp.isclose(r, 0), jnp.greater(b, r + 1)),
            none,
            lambda _: soln_impl(b, r),
            operand=None,
        ),
        operand=None,
    )


@jnp.vectorize
def s02(b, r):
    one = jnp.ones_like(b)
    r2 = jnp.square(r)
    b2 = jnp.square(b)
    invr = 1 / r
    invb = 1 / b
    bpr = b + r
    invfourbr = 0.25 * invr * invb
    onembpr2 = (1 + bpr) * (1 - bpr)
    r2pb2 = r2 + b2
    eta2 = 0.5 * r2 * (r2pb2 + b2)

    def large(_):
        return jnp.pi * (1 - r2), 4 * jnp.pi * (eta2 - 0.5)

    def small(_):
        area = kite_area(one, b, r)
        factor = (r - 1) * (r + 1)
        kap0 = jnp.arctan2(area, b2 + factor)
        kap1 = jnp.arctan2(area, b2 - factor)
        Alens = kap1 + r2 * kap0 - area * 0.5
        s0 = jnp.pi - Alens

        ds = 2 * (
            -(jnp.pi - kap1)
            + 2 * eta2 * kap0
            - 0.25 * area * (1 + 5 * r2 + b2)
        )
        return s0, ds

    s0, ds = lax.cond(
        jnp.greater(onembpr2 * invfourbr + 1, 1), large, small, operand=None
    )
    s2 = 2 * s0 + ds
    return s0, s2


def soln_impl(b, r):
    one = jnp.ones_like(b)
    zero = jnp.zeros_like(b)
    r2 = jnp.square(r)

    def zero_b(_):
        onemr2 = 1 - r2
        return (
            jnp.pi * onemr2,
            2 * jnp.pi * onemr2 * jnp.sqrt(onemr2) / 3.0,
            -2 * jnp.pi * onemr2 * r2,
        )

    def nonzero_b(_):
        b2 = jnp.square(b)
        invr = 1 / r
        invb = 1 / b
        bmr = b - r
        bpr = b + r
        fourbr = 4 * b * r
        invfourbr = 0.25 * invr * invb
        onembmr2 = (1 + bmr) * (1 - bmr)
        onembmr2inv = 1 / onembmr2
        onembpr2 = (1 + bpr) * (1 - bpr)
        sqonembmr2 = jnp.sqrt(onembmr2)
        sqbr = jnp.sqrt(b * r)
        r2pb2 = r2 + b2
        eta2 = 0.5 * r2 * (r2pb2 + b2)

        ksq = onembpr2 * invfourbr + 1

        # === s0 and s2 ===
        def bigk(_):
            kcsq = onembpr2 * onembmr2inv
            s0 = jnp.pi * (1 - r2)
            four_pi_eta = 4 * jnp.pi * (eta2 - 0.5)
            return s0, four_pi_eta, kcsq

        def smallk(_):
            area = kite_area(one, b, r)
            kcsq = -onembpr2 * invfourbr
            factor = (r - 1) * (r + 1)
            kap0 = jnp.arctan2(area, b2 + factor)
            kap1 = jnp.arctan2(area, b2 - factor)
            Alens = kap1 + r2 * kap0 - area * 0.5
            s0 = jnp.pi - Alens

            four_pi_eta = 2 * (
                -(jnp.pi - kap1)
                + 2 * eta2 * kap0
                - 0.25 * area * (1 + 5 * r2 + b2)
            )
            return s0, four_pi_eta, kcsq

        s0, four_pi_eta, kcsq = lax.cond(
            jnp.greater(ksq, 1), bigk, smallk, operand=None
        )
        s2 = 2 * s0 + four_pi_eta

        # === s1 ===
        def eq_rb(_):
            def not_half(_):
                m = 4 * r2
                cond = jnp.greater(r, 0.5)
                m_or_minv, a1, a2 = lax.cond(
                    cond,
                    lambda _: (1 / m, -m * invr, (2 * m - 3) * invr),
                    lambda _: (m, 4 * m - 6, -2 * m),
                    operand=None,
                )
                Eofk = cel(m_or_minv, one, one, 1 - m_or_minv)
                Em1mKdm = cel(m_or_minv, one, one, zero)
                return jnp.pi + (a1 * Eofk + a2 * Em1mKdm) / 3

            return lax.cond(
                jnp.isclose(r, 0.5),
                lambda _: jnp.pi - 4.0 / 3,
                not_half,
                operand=None,
            )

        def neq_rb(_):
            def one_k(_):
                rootr1mr = jnp.sqrt(r * (1 - r))
                return (
                    2 * jnp.arccos(1.0 - 2.0 * r)
                    - 4 * (3 + 2 * r - 8 * r2) * rootr1mr / 3
                    - 2 * jnp.pi * (r > 0.5)
                )

            def notone_k(_):
                def bigk_inner(_):
                    bmrdbpr = (b - r) / (b + r)
                    mu = 3 * bmrdbpr * onembmr2inv
                    p = bmrdbpr * bmrdbpr * onembpr2 * onembmr2inv
                    return (1 / ksq, p, 1 + mu, p + mu), (
                        2 * sqonembmr2 * onembpr2,
                        -2 * sqonembmr2 * (4 - 7 * r2 - b2),
                        zero,
                    )

                def smallk_inner(_):
                    factor = onembmr2 / sqbr
                    return (
                        ksq,
                        (b - r) * (b - r) * kcsq,
                        zero,
                        3 * kcsq * (b - r) * (b + r),
                    ), (
                        factor,
                        -factor * fourbr,
                        factor
                        * (6 * r2 + 2 * b * r - 3)
                        * cel(ksq, one, one, zero),
                    )

                (a1, a2, a3, a4), (f1, f2, f3) = lax.cond(
                    jnp.greater(ksq, 1), bigk_inner, smallk_inner, operand=None
                )
                Piofk = cel(a1, a2, a3, a4)
                Eofk = cel(a1, one, one, kcsq)
                return f1 * Piofk + f2 * Eofk + f3

            return lax.cond(
                jnp.isclose(ksq, 1),
                one_k,
                notone_k,
                operand=None,
            )

        lambda1 = lax.cond(jnp.isclose(r, b), eq_rb, neq_rb, operand=None)
        s1 = lax.cond(
            jnp.greater(r, b),
            lambda _: -lambda1 / 3,
            lambda _: (2 * jnp.pi - lambda1) / 3,
            operand=None,
        )

        return s0, s1, s2

    return lax.cond(jnp.isclose(b, 0), zero_b, nonzero_b, operand=None)


def kite_area(p0, p1, p2):
    p0, p1 = lax.cond(
        jnp.less(p0, p1), lambda _: (p1, p0), lambda _: (p0, p1), operand=None
    )
    p1, p2 = lax.cond(
        jnp.less(p1, p2), lambda _: (p2, p1), lambda _: (p1, p2), operand=None
    )
    p0, p1 = lax.cond(
        jnp.less(p0, p1), lambda _: (p1, p0), lambda _: (p0, p1), operand=None
    )
    square_area = (
        (p0 + (p1 + p2))
        * (p2 - (p0 - p1))
        * (p2 + (p0 - p1))
        * (p0 + (p1 - p2))
    )
    return jnp.sqrt(jnp.maximum(square_area, 0.0))
