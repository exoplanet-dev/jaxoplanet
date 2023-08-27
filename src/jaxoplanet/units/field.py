from functools import partial
from typing import Any

import equinox as eqx

from jaxoplanet.units.decorator import _is_quantity
from jaxoplanet.units.registry import unit_registry


def _convert_to(unit: unit_registry.Unit, value: Any) -> unit_registry.Quantity:
    if _is_quantity(value):
        return value.to(unit)
    else:
        return unit_registry.Quantity(value, unit)


def field(unit: unit_registry.Unit, **kwargs: Any) -> Any:
    original_converter = kwargs.pop("converter", None)

    if original_converter is None:
        return eqx.field(converter=partial(_convert_to, unit), **kwargs)
    else:

        def converter(value: Any) -> Any:
            return _convert_to(unit, original_converter(value))

        return eqx.field(converter=converter, **kwargs)
