from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union

from jaxoplanet.experimental.starry.surface import Surface
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody, System


class SurfaceBody(Body):
    surface: Optional[Surface] = None


class SurfaceSystem(System):
    central_surface: Optional[Surface]
    _body_surface_stack: ObjectStack[Surface]

    def __init__(
        self,
        central: Optional[Central] = None,
        central_surface: Optional[Surface] = None,
        *,
        bodies: Iterable[
            tuple[Union[Body, OrbitalBody, SurfaceBody], Optional[Surface]]
        ] = (),
    ):
        self.central = Central() if central is None else central

        if central_surface is None:
            central_surface = Surface()

        self.central_surface = central_surface

        orbital_bodies = []
        body_surfaces = []
        for body, surface in bodies:
            if isinstance(body, OrbitalBody):
                orbital_bodies.append(body)
                body_surfaces.append(surface)
            else:
                orbital_bodies.append(OrbitalBody(self.central, body))
                if surface is None:
                    body_surfaces.append(getattr(body, "surface", None))
                else:
                    body_surfaces.append(surface)

        self._body_stack = ObjectStack(*orbital_bodies)
        self._body_surface_stack = ObjectStack(*body_surfaces)

    @property
    def body_surfaces(self) -> tuple[Surface, ...]:
        return self._body_surface_stack.objects

    def add_body(
        self,
        body: Optional[Union[Body, SurfaceBody]] = None,
        surface: Optional[Surface] = None,
        **kwargs: Any,
    ) -> "SurfaceSystem":
        if body is None:
            body = Body(**kwargs)
        if surface is None:
            surface = getattr(body, "surface", None)
        bodies = list(zip(self.bodies, self.body_surfaces)) + [(body, surface)]
        return SurfaceSystem(
            central=self.central,
            central_surface=self.central_surface,
            bodies=bodies,
        )

    def surface_vmap(
        self,
        func: Callable,
        in_axes: Union[int, None, Sequence[Any]] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._body_surface_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)
