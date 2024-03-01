from collections.abc import Iterable, Sequence
from typing import Any, Callable, Optional, Union

from jaxoplanet.experimental.starry.maps import Map
from jaxoplanet.object_stack import ObjectStack
from jaxoplanet.orbits.keplerian import Body, Central, OrbitalBody, System


class SurfaceMapBody(Body):
    surface_map: Optional[Map] = None


class SurfaceMapSystem(System):
    surface_map: Optional[Map]
    _surface_map_stack: ObjectStack[Map]

    def __init__(
        self,
        central: Optional[Central] = None,
        surface_map: Optional[Map] = None,
        *,
        bodies: Iterable[
            tuple[Union[Body, OrbitalBody, SurfaceMapBody], Optional[Map]]
        ] = (),
    ):
        self.central = Central() if central is None else central

        if surface_map is None:
            surface_map = Map()

        self.surface_map = surface_map

        orbital_bodies = []
        surface_maps = []
        for body, surface_map in bodies:
            if isinstance(body, OrbitalBody):
                orbital_bodies.append(body)
                surface_maps.append(surface_map)
            else:
                orbital_bodies.append(OrbitalBody(self.central, body))
                if surface_map is None:
                    surface_maps.append(getattr(body, "surface_map", None))
                else:
                    surface_maps.append(surface_map)

        self._body_stack = ObjectStack(*orbital_bodies)
        self._surface_map_stack = ObjectStack(*surface_maps)

    @property
    def surface_maps(self) -> tuple[Map, ...]:
        return self._surface_map_stack.objects

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
        bodies = list(zip(self.bodies, self.surface_maps)) + [(body, surface_map)]
        return SurfaceMapSystem(
            central=self.central, surface_map=self.surface_map, bodies=bodies
        )

    def surface_map_vmap(
        self,
        func: Callable,
        in_axes: Union[int, None, Sequence[Any]] = 0,
        out_axes: Any = 0,
    ) -> Callable:
        return self._surface_map_stack.vmap(func, in_axes=in_axes, out_axes=out_axes)
