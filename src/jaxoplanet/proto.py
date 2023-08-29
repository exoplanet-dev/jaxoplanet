from typing import Optional, Protocol

from jaxoplanet.types import Quantity


class LightCurveBody(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def radius(self) -> Quantity:
        ...


class LightCurveOrbit(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def radius(self) -> Quantity:
        ...

    def relative_position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        ...

    @property
    def central_radius(self) -> Quantity:
        ...


class Orbit(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

    @property
    def radius(self) -> Quantity:
        ...

    def relative_position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        ...

    @property
    def central_radius(self) -> Quantity:
        ...

    def position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        ...

    def central_position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        ...

    def velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        ...

    def central_velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        ...

    def relative_velocity(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]:
        ...

    def radial_velocity(
        self, t: Quantity, semiamplitude: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        ...
