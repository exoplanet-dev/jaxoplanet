# mypy: ignore-errors
"""High-precision reference (mpmath) for power-law intensities by Green's theorem,
independent of the series and of any closed form: adaptive quadrature at 40 digits"""


def flux_ref(z, k, alphas, coefs, dps=40):
    """Relative flux for ``I(mu) = sum_i coefs_i mu^alphas_i``"""
    import mpmath as mp

    with mp.workdps(dps):
        z = mp.mpf(float(z))
        k = mp.mpf(float(k))
        gs = [mp.mpf(float(a)) / 2 for a in alphas]
        cs = [mp.mpf(float(c)) for c in coefs]
        total = 2 * mp.pi * sum(c / (2 * g + 2) for c, g in zip(cs, gs, strict=True))
        if z >= 1 + k:
            return 1.0
        if k >= 1 and z <= k - 1:
            return 0.0

        def h(x, g):
            if x == 0:
                return mp.mpf(1) / 2
            x = min(x, mp.mpf(1))
            return (1 - (1 - x) ** (g + 1)) / (2 * (g + 1) * x)

        blocked = mp.mpf(0)
        if z > 0:
            cs_ = (1 + z * z - k * k) / (2 * z)
            th = mp.acos(max(min(cs_, 1), -1))
        else:
            th = mp.pi if k >= 1 else mp.mpf(0)
        blocked += sum(c * 2 * th / (2 * (g + 1)) for c, g in zip(cs, gs, strict=True))
        c0 = (
            (1 - z * z - k * k) / (2 * z * k)
            if z > 0
            else (mp.mpf(1) if k < 1 else mp.mpf(-1))
        )
        if c0 > -1:
            t0 = mp.acos(min(c0, 1))

            def f(t):
                r2 = z * z + k * k + 2 * z * k * mp.cos(t)
                return (
                    2
                    * k
                    * (k + z * mp.cos(t))
                    * sum(c * h(r2, g) for c, g in zip(cs, gs, strict=True))
                )

            ten = mp.mpf(10)
            pts = [
                t0 + (mp.pi - t0) * x
                for x in (
                    0,
                    ten**-8,
                    ten**-4,
                    ten**-2,
                    mp.mpf(1) / 10,
                    mp.mpf(1) / 2,
                    1,
                )
            ]
            blocked += mp.quad(f, pts)
        return float(1 - blocked / total)


def disk_mean_ref(z, k, g, dps=40):
    """Mean of ``(1 - r^2)^g`` over the occulting disk (interior, ``z + k < 1``) from
    the Green's theorem line integral at ``dps`` digits; returns an mpf"""
    import mpmath as mp

    with mp.workdps(dps):
        z = mp.mpf(float(z))
        k = mp.mpf(float(k))
        g = mp.mpf(float(g))

        def f(t):
            r2 = z * z + k * k + 2 * z * k * mp.cos(t)
            hh = (
                mp.mpf(1) / 2
                if r2 == 0
                else (1 - (1 - r2) ** (g + 1)) / (2 * (g + 1) * r2)
            )
            return hh * (k + z * mp.cos(t))

        return 2 * mp.quad(f, [0, mp.pi / 8, mp.pi / 2, mp.pi]) / (mp.pi * k)
