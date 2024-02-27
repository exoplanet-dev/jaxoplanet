from typing import Any, Protocol, TypeVar

from jaxoplanet.types import Array, Quantity

Value = TypeVar("Value", Array, Quantity, covariant=True)


class LightCurveFunc(Protocol[Value]):
    def __call__(self, time: Quantity, *args: Any, **kwargs: Any) -> Value: ...
