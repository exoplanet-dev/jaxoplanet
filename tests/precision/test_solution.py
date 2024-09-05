import pytest
import numpy as np
from jaxoplanet.experimental.starry import solution
from jaxoplanet.experimental.starry.multiprecision import solution as mp_solution, utils
from jaxoplanet.test_utils import assert_allclose

TOLERANCE = 1e-15


@pytest.mark.parametrize("r", [0.01, 0.1, 1.0, 10.0, 100.0])
def test_sT(r, l_max=5, order=500):

    # We know that these are were the errors are the highest
    b = 1 - r if r < 1 else r

    expect = utils.to_numpy(mp_solution.sT(l_max, b, r))
    calc = solution.solution_vector(l_max, order=order)(b, r)

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


def plot_sT_precision(lmax=20):
    from collections import defaultdict

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    radii = [0.01, 0.1, 1.0, 10.0, 100.0]
    orders = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    result = defaultdict(dict)
    for r in tqdm(radii):
        b = 1 - r if r < 1 else r
        expect = utils.to_numpy(mp_solution.sT(lmax, b, r))
        diffs = []
        for order in orders:
            calc = solution.solution_vector(lmax, order=order)(b, r)
            diff = np.abs(utils.diff_mp(expect, calc))
            diffs.append(diff)
        result[r] = diffs

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
    plt.savefig("log_sT_precision.png")


if __name__ == "__main__":
    plot_sT_precision()
