from functools import partial
from typing import Any

import equinox as eqx
import jax

from jaxoplanet.units.decorator import _apply_units, _is_quantity


def field(*, units: Any = None, strict: bool = False, **kwargs: Any) -> Any:
    """A custom Equinox field with support for units

    This is a wrapper around :func:`equinox.field` that adds support for units.
    The ``units`` argument is used to specify the units of the field either as a
    plain ``Unit`` or string, or as a Pytree with the same structure as the
    expected input.
    """
    if units is None:
        return eqx.field(**kwargs)

    original_converter = kwargs.pop("converter", lambda x: x)

    def converter(value: Any) -> Any:
        if value is None:
            return None
        else:
            return jax.tree_util.tree_map(
                partial(_apply_units, strict=strict),
                original_converter(value),
                units,
                is_leaf=_is_quantity,
            )

    return eqx.field(converter=converter, **kwargs)
