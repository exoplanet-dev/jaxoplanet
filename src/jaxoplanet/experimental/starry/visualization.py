import numpy as np

from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.experimental.starry.utils import graticule
from jaxoplanet.experimental.starry.ylm import Ylm


def show_map(
    ylm_surface_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    **kwargs,
):
    """Show map of a

    Args:
        ylm_surface_body (Ylm, Surface or SurfaceBody): Map or Body with a map
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

    if hasattr(ylm_surface_body, "surface"):
        surface = ylm_surface_body.surface
        if ylm_surface_body.radius is not None:
            radius = ylm_surface_body.radius.magnitude
        else:
            radius = 1.0
        n = int(np.ceil(n * np.cbrt(radius)))
    # import Ylm leads to circular import
    elif isinstance(ylm_surface_body, Ylm):
        surface = Surface(y=ylm_surface_body)
        radius = 1.0
    else:
        surface = ylm_surface_body
        radius = 1.0

    ax.imshow(
        surface.render(
            theta,
            res,
        ),
        origin="lower",
        **kwargs,
        extent=(-radius, radius, -radius, radius),
    )
    if n is not None:
        graticule(
            surface.inc,
            surface.obl,
            theta,
            radius=radius,
            n=n,
            white_contour=white_contour,
        )
    ax.axis(False)
