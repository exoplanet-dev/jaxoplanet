from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union

from jaxoplanet.experimental.starry.maps import Map
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody, System


class SurfaceMapBody(Body):
    surface_map: Optional[Map] = None


class SurfaceMapSystem(System):
    central_surface_map: Optional[Map]
    _body_surface_map_stack: ObjectStack[Map]

    def __init__(
        self,
        central: Optional[Central] = None,
        central_surface_map: Optional[Map] = None,
        *,
        bodies: Iterable[
            tuple[Union[Body, OrbitalBody, SurfaceMapBody], Optional[Map]]
        ] = (),
    ):
        self.central = Central() if central is None else central
        self.central_surface_map = central_surface_map

        orbital_bodies = []
        bodies_surface_maps = []
        for body, surface_map in bodies:
            if isinstance(body, OrbitalBody):
                orbital_bodies.append(body)
                bodies_surface_maps.append(surface_map)
            else:
                orbital_bodies.append(OrbitalBody(self.central, body))
                if surface_map is None:
                    bodies_surface_maps.append(getattr(body, "surface_map", None))
                else:
                    bodies_surface_maps.append(surface_map)

        self._body_stack = ObjectStack(*orbital_bodies)
        self._body_surface_map_stack = ObjectStack(*bodies_surface_maps)

    @property
    def bodies_surface_maps(self) -> tuple[Map, ...]:
        return self._body_surface_map_stack.objects

    def add_body(
        self,
        body: Optional[Union[Body, SurfaceMapBody]] = None,
        surface_map: Optional[Map] = None,
        **kwargs: Any,
    ) -> "SurfaceMapSystem":
        if body is None:
            body = Body(**kwargs)
        if surface_map is None:
            surface_map = getattr(body, "surface_map", None)
        bodies = list(zip(self.bodies, self.bodies_surface_maps)) + [
            (body, surface_map)
        ]
        return SurfaceMapSystem(
            central=self.central,
            central_surface_map=self.central_surface_map,
            bodies=bodies,
        )

    def surface_map_vmap(
        self,
        func: Callable,
        in_axes: Union[int, None, Sequence[Any]] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._body_surface_map_stack.vmap(
            func, in_axes=in_axes, out_axes=out_axes
        )
