from typing import Optional, Tuple

from typing_extensions import Protocol

from jaxoplanet._src.types import Array


class LightCurveBody(Protocol):
    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def radius(self) -> Array:
        ...


class LightCurveOrbit(LightCurveBody):
    def relative_position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        ...

    @property
    def central_radius(self) -> Array:
        ...


class Orbit(LightCurveOrbit):
    def position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        ...

    def central_position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        ...

    def velocity(self, t: Array) -> Tuple[Array, Array, Array]:
        ...

    def central_velocity(self, t: Array) -> Tuple[Array, Array, Array]:
        ...

    def relative_velocity(self, t: Array) -> Tuple[Array, Array, Array]:
        ...

    def radial_velocity(
        self, t: Array, semiamplitude: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        ...
