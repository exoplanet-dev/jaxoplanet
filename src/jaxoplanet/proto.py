from typing import Optional

from typing_extensions import Protocol

from jaxoplanet.types import Array


class LightCurveBody(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def radius(self) -> Array:
        ...


class LightCurveOrbit(LightCurveBody):
    def relative_position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        ...

    @property
    def central_radius(self) -> Array:
        ...


class Orbit(LightCurveOrbit):
    def position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        ...

    def central_position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        ...

    def velocity(self, t: Array) -> tuple[Array, Array, Array]:
        ...

    def central_velocity(self, t: Array) -> tuple[Array, Array, Array]:
        ...

    def relative_velocity(self, t: Array) -> tuple[Array, Array, Array]:
        ...

    def radial_velocity(
        self, t: Array, semiamplitude: Optional[Array] = None
    ) -> tuple[Array, Array, Array]:
        ...
