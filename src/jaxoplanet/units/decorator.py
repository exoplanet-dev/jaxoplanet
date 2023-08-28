import inspect
from functools import partial, wraps
from typing import Any, Callable

import jax
from pint import DimensionalityError

from jaxoplanet.units.registry import unit_registry


def quantity_input(
    func: Callable | None = None,
    *,
    _strict: bool = False,
    **kwargs: Any,
) -> Callable | "QuantityInput":
    """A decorator to wrap functions that require quantities as inputs

    Please note, this is similar to the decorator of the same name from
    ``astropy.units``, but the behavior is slightly different, in ways that
    we'll try to highlight here.

    This decorator will wrap a function and check or convert the units of all
    inputs, such that the wrapped function can assume that all input units are
    correct.

    By default, if a non-``Quantity`` is provided, it will be assumed to be in
    the correct units, and converted to a ``Quantity``. If the ``_strict`` flag
    is instead set to ``True``, inputting a non-``Quantity`` will raise a
    ``ValueError``.

    Simple examples
    ---------------

    The following function expects a length in meters and a time in seconds, and
    it returns a speed in meters per second:

    .. code-block:: python

        from jaxoplanet.units import unit_registry as ureg

        @units.quantity_input(a=ureg.m, b=ureg.s)
        def speed(a, b):
            return a / b

    If we call this function with a length and a time, it will work as expected:

    .. code-block:: python

        speed(1.5 * ureg.m, 0.5 * ureg.s)

    And it will also handle unit conversions:

    .. code-block:: python

        speed(1.5 * ureg.AU, 0.5 * ureg.day)  # The result will still be in m/s

    To skip validating specific inputs, you can set the unit to ``None``, or
    omit it from the decorator arguments:

    .. code-block:: python

        @units.quantity_input(x=ureg.m)  # optionally include flag=None
        def condition(x, flag):
            if flag:
                return x
            else:
                return 0.0 * x

    JAX Pytree support
    ------------------

    This decorator also supports JAX Pytrees, so you can use it to wrap
    functions with structured inputs. For example, we could rewrite the
    ``speed`` example from above as:

    .. code-block:: python

        @units.quantity_input(params={"distance": ureg.m, "time": ureg.s})
        def speed(params):
            return params["distance"] / params["time"]

    This will work with arbitrary Pytrees, as long as structure of the input
    fully matches the decorator argument. In other words, since the full Pytree
    structure must be specified, you'll need to explicitly list ``None`` for any
    Pytree nodes that you want to skip during validation:

    .. code-block:: python

        # Omitting `flag` from the decorator wouldn't work here
        @units.quantity_input(params={"x": ureg.m, "flag": None})
        def condition(params):
            if params["flag"]:
                return params["x"]
            else:
                return 0.0 * params["x"]
    """
    if func is None:
        return QuantityInput(_strict=_strict, **kwargs)
    else:
        return QuantityInput(_strict=_strict, **kwargs)(func)


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


def _is_quantity(x: Any) -> bool:
    return hasattr(x, "_magnitude") and hasattr(x, "_units")
