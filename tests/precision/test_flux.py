import jax
import pytest
import numpy as np
import pytest
from jaxoplanet.experimental.starry import light_curves
from jaxoplanet.experimental.starry.multiprecision import flux as mp_flux, mp, utils
from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.test_utils import assert_allclose

TOLERANCE = 1e-15


@pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_flux(r, l_max=5, order=500):

    # We know that these are were the errors are the highest
    b = 1 - r if r < 1 else r

    n = (l_max + 1) ** 2
    expect = np.zeros(n)
    calc = np.zeros(n)
    ys = np.eye(n, dtype=np.float64)
    ys[:, 0] = 1.0

    @jax.jit
    def light_curve(y):
        surface = Surface(y=Ylm.from_dense(y, normalize=False), normalize=False)
        return light_curves.surface_light_curve(surface, y=b, z=10.0, r=r, order=order)

    for i, y in enumerate(ys):
        expect[i] = float(mp_flux.flux_function(l_max, mp.pi / 2, 0.0)(y, b, r, 0.0))
        calc[i] = light_curve(y)

    for n in range(expect.size):
        # For logging/debugging purposes, work out the case id
        l = np.floor(np.sqrt(n)).astype(int)  # noqa
        m = n - l**2 - l
        mu = l - m
        nu = l + m
        if mu == 1 and l == 1:
            case = 1
        elif mu % 2 == 0 and (mu // 2) % 2 == 0:
            case = 2
        elif mu == 1 and l % 2 == 0:
            case = 3
        elif mu == 1:
            case = 4
        elif (mu - 1) % 2 == 0 and ((mu - 1) // 2) % 2 == 0:
            case = 5
        else:
            case = 0

        assert_allclose(
            calc[n],
            expect[n],
            err_msg=f"n={n}, l={l}, m={m}, mu={mu}, nu={nu}, case={case}",
            atol=TOLERANCE,
        )


def plot_flux_precision(l_max=5):
    from collections import defaultdict
    from functools import partial

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    radii = [0.01, 0.1, 1.0, 10.0, 100.0]
    orders = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    result = defaultdict(dict)
    expect = np.zeros((l_max + 1) ** 2)
    calc = np.zeros((l_max + 1) ** 2)

    ys = np.eye((l_max + 1) ** 2, dtype=np.float64)
    ys[:, 0] = 1.0

    @partial(jax.jit, static_argnames=("order",))
    def light_curve(y, b, r, order):
        surface = Surface(y=Ylm.from_dense(y, normalize=False), normalize=False)
        return light_curves.surface_light_curve(surface, y=b, z=10.0, r=r, order=order)

    for r in tqdm(radii):
        b = 1 - r if r < 1 else r
        for i, y in enumerate(ys):
            expect[i] = float(
                mp_flux.flux_function(l_max, mp.pi / 2, 0.0)(y, b, r, 0.0)
            )
        diffs = []
        for order in orders:
            for i, y in enumerate(ys):
                calc[i] = light_curve(y, b, r, order)
            diff = np.abs(utils.diff_mp(expect, calc))
            diffs.append(diff)

        result[r] = diffs

    np.savez("flux_precision.npz", result=result, radii=radii, orders=orders)

    cmap = plt.get_cmap("magma")
    for i, r in enumerate(radii):
        color = cmap(i / len(radii))
        error = np.array(result[r]).max(1)
        plt.plot(orders, error, ".-", label=f"r={r}", color=color)
    plt.yscale("log")
    plt.legend()
    plt.xlabel("order")
    plt.ylabel("max error")
    plt.axhline(1e-6, ls="--", lw=1, color="k", alpha=0.2)
    plt.axhline(1e-9, ls="--", lw=1, color="k", alpha=0.2)
    plt.ylim(1e-17, 1e-4)
    plt.annotate(
        "ppm",
        xy=(0, 1e-6),
        xycoords="data",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        alpha=0.75,
    )
    plt.annotate(
        "ppb",
        xy=(0, 1e-9),
        xycoords="data",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        alpha=0.75,
    )
    plt.tight_layout()
    plt.savefig("flux_precision.png")


if __name__ == "__main__":
    plot_flux_precision()
