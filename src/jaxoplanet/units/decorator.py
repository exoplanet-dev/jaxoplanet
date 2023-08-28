import inspect
from functools import partial, wraps
from typing import Any, Callable

import jax
from pint import DimensionalityError

from jaxoplanet.units.registry import unit_registry


def _is_quantity(x: Any) -> bool:
    return hasattr(x, "_magnitude") and hasattr(x, "_units")


class QuantityInput:
    def __init__(
        self,
        *,
        _strict: bool = False,
        **kwargs: Any,
    ):
        self.decorator_kwargs = kwargs
        self.strict = _strict

    def __call__(self, func: Callable) -> Callable:
        signature = inspect.signature(func)

        input_unit_map = {}
        bound_input_units = signature.bind_partial(**self.decorator_kwargs)
        for param in signature.parameters.values():
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                raise TypeError(
                    "General variable arguments and keyword arguments are not "
                    "supported"
                )
            else:
                input_unit_map[param.name] = bound_input_units.arguments.get(
                    param.name, None
                )

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            bound_args = signature.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for name, value in bound_args.arguments.items():
                # If the value and the default value is None, pass through
                # ignoring units
                if value is None and signature.parameters[name].default is None:
                    continue

                unit = input_unit_map.get(name, None)
                if unit is not None:
                    bound_args.arguments[name] = jax.tree_util.tree_map(
                        partial(self._apply_unit, name),
                        value,
                        unit,
                        is_leaf=_is_quantity,
                    )

            return func(*bound_args.args, **bound_args.kwargs)

        return wrapped

    def _apply_unit(self, name: str, value: Any, unit: Any) -> Any:
        if unit is None:
            return value
        if _is_quantity(value):
            try:
                return value.to(unit)
            except DimensionalityError as e:
                raise DimensionalityError(
                    e.units1,
                    e.units2,
                    e.dim1,
                    e.dim2,
                    f" for input '{name}'",
                ) from None
        elif self.strict:
            raise ValueError(
                f"All elements of argument '{name}' must be quantities for strict "
                "parsing"
            )
        else:
            return unit_registry.Quantity(value, unit)


def quantity_input(
    func: Callable | None = None,
    *,
    _strict: bool = False,
    **kwargs: Any,
) -> Callable | QuantityInput:
    if func is None:
        return QuantityInput(_strict=_strict, **kwargs)
    else:
        return QuantityInput(_strict=_strict, **kwargs)(func)
