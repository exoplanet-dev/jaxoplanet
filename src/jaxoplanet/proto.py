__all__ = ["LightCurveOrbit"]

from typing import Protocol

from jaxoplanet.types import Scalar


class LightCurveOrbit(Protocol):
    """An interface for orbits that can be used to compute light curves"""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def radius(self) -> Scalar: ...

    def relative_position(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]: ...

    @property
    def central_radius(self) -> Scalar: ...
