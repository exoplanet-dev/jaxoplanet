__all__ = ["LightCurveOrbit"]

from typing import Protocol

from jaxoplanet.types import Quantity


class LightCurveOrbit(Protocol):
    """An interface for orbits that can be used to compute light curves"""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def radius(self) -> Quantity: ...

    def relative_position(self, t: Quantity) -> tuple[Quantity, Quantity, Quantity]: ...

    @property
    def central_radius(self) -> Quantity: ...
