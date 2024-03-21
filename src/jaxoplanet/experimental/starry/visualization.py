import numpy as np

from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.experimental.starry.utils import graticule
from jaxoplanet.experimental.starry.ylm import Ylm


def show_map(
    ylm_map_or_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    **kwargs,
):
    """Show map of a

    Args:
        map_or_body (Map, Central or Body): Map or Body with a map
        theta (float, optional): Rotation angle of the map wrt its rotation axis.
        Defaults to 0.0.
        res (int, optional): Resolution of the map render. Defaults to 400.
        n (int, optional): number of latitude and longitude lines to show.
        Defaults to 6.
        ax (matplotlib.pyplot.Axes, optional): plot axes. Defaults to None.
        white_contour (bool, optional): Whether to surround the map by a white border
        (to hide border pixel aliasing). Defaults to True.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)

    if hasattr(ylm_map_or_body, "map"):
        map = ylm_map_or_body.map
        radius = ylm_map_or_body.radius.magnitude
        n = int(np.ceil(n * np.cbrt(radius)))
    # import Ylm leads to circular import
    elif isinstance(ylm_map_or_body, Ylm):
        map = Surface(y=ylm_map_or_body)
        radius = 1.0
    else:
        map = ylm_map_or_body
        radius = 1.0

    ax.imshow(
        map.render(
            theta,
            res,
        ),
        origin="lower",
        **kwargs,
        extent=(-radius, radius, -radius, radius),
    )
    if n is not None:
        graticule(
            map.inc,
            map.obl,
            theta,
            radius=radius,
            n=n,
            white_contour=white_contour,
        )
    ax.axis(False)
