import jax.numpy as jnp
import numpy as np

from jaxoplanet.starry.core.basis import A1
from jaxoplanet.starry.core.polynomials import Pijk
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.utils import graticule
from jaxoplanet.starry.ylm import Ylm


def show_surface(
    ylm_pijk_surface_body,
    theta: float = 0.0,
    res: int = 400,
    n: int = 6,
    ax=None,
    white_contour: bool = True,
    radius: float = None,
    include_phase: bool = True,
    rv=False,
    return_im=False,
    **kwargs,
):
    """Show map of a

    Args:
        ylm_pijk_surface_body (Ylm, Pijk, Surface or SurfaceBody): Ylm, Pijk, Surface or
            Body with a surface
        theta (float, optional): Rotation angle of the map wrt its rotation axis.
            Defaults to 0.0.
        res (int, optional): Resolution of the map render. Defaults to 400.
        n (int, optional): number of latitude and longitude lines to show.
            Defaults to 6.
        ax (matplotlib.pyplot.Axes, optional): plot axes. Defaults to None.
        white_contour (bool, optional): Whether to surround the map by a white border
            (to hide border pixel aliasing). Defaults to True.
        radius (float, optional): Radius of the body. Defaults to None.
        include_phase (bool, optional): Whether to add the proper phase to the map to
            the rotation angle theta. Defaults to True.
    """
    import matplotlib.pyplot as plt

    if rv:
        assert isinstance(
            ylm_pijk_surface_body, Surface
        ), "if rv is True, surface must be a Surface object"
        kwargs.setdefault("cmap", "RdBu_r")

    if ax is None:
        ax = plt.gca()
        if ax is None:
            ax = plt.subplot(111)
    if isinstance(ylm_pijk_surface_body, (np.ndarray, jnp.ndarray)):
        y = Ylm.from_dense(ylm_pijk_surface_body, normalize=False)
        surface = Surface(y=y, normalize=False)
        radius = 1.0 if radius is None else radius
    elif hasattr(ylm_pijk_surface_body, "surface"):
        surface = ylm_pijk_surface_body.surface
        if ylm_pijk_surface_body.radius is not None:
            radius = ylm_pijk_surface_body.radius.magnitude
        else:
            radius = 1.0 if radius is None else radius
        n = int(np.ceil(n * np.cbrt(radius)))
    # import Ylm leads to circular import
    elif isinstance(ylm_pijk_surface_body, Ylm):
        surface = Surface(y=ylm_pijk_surface_body)
        radius = 1.0 if radius is None else radius
    elif isinstance(ylm_pijk_surface_body, Pijk):
        A1inv = np.linalg.inv(A1(ylm_pijk_surface_body.degree).todense())
        surface = Surface(
            y=Ylm.from_dense(A1inv @ ylm_pijk_surface_body.todense(), normalize=False),
            normalize=False,
        )
        radius = 1.0 if radius is None else radius
    else:
        surface = ylm_pijk_surface_body
        radius = 1.0 if radius is None else radius

    phase = theta + (surface.phase if include_phase else 0.0)

    im = ax.imshow(
        surface.render(phase, res, rv=rv),
        origin="lower",
        **kwargs,
        extent=(-radius, radius, -radius, radius),
    )
    if n is not None:
        graticule(
            surface.inc,
            surface.obl,
            phase,
            radius=radius,
            n=n,
            white_contour=white_contour,
        )
    ax.axis(False)

    if return_im:
        return im
