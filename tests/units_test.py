import jax
import pytest
from pint import DimensionalityError

from jaxoplanet import units
from jaxoplanet.test_utils import assert_quantity_allclose
from jaxoplanet.units import unit_registry as ureg


def test_quantity_input():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        return a / b

    assert_quantity_allclose(func(1.5 * ureg.m, 0.5 * ureg.s), 3.0 * ureg.m / ureg.s)


def test_quantity_input_without_units():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        return a / b

    assert_quantity_allclose(func(1.5 * ureg.m, 0.5), 3.0 * ureg.m / ureg.s)
    assert_quantity_allclose(func(1.5, 0.5 * ureg.s), 3.0 * ureg.m / ureg.s)


def test_quantity_input_skip_parameter_explicit():
    @units.quantity_input(a=ureg.m, b=None)
    def func(a, b):
        return a * b

    assert_quantity_allclose(func(1.5 * ureg.m, 2.0), 3.0 * ureg.m)


def test_quantity_input_skip_parameter_implicit():
    @units.quantity_input(a=ureg.m)
    def func(a, b):
        return a * b

    assert_quantity_allclose(func(1.5 * ureg.m, 2.0), 3.0 * ureg.m)


def test_quantity_input_conversion():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        return a / b

    assert_quantity_allclose(
        func(150.0 * ureg.cm, 500.0 * ureg.ms), 3.0 * ureg.m / ureg.s
    )


def test_quantity_input_invalid():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        return a / b

    with pytest.raises(DimensionalityError):
        func(150.0 * ureg.hr, 0.5 * ureg.s)


@pytest.mark.parametrize("with_jit", [False, True])
def test_quantity_input_pytree(with_jit):
    @units.quantity_input(params={"a": ureg.m, "b": ureg.s})
    def func(params):
        return params["a"] / params["b"]

    if with_jit:
        func = jax.jit(func)

    assert_quantity_allclose(
        func({"a": 1.5 * ureg.m, "b": 0.5 * ureg.s}), 3.0 * ureg.m / ureg.s
    )
    assert_quantity_allclose(func({"a": 1.5 * ureg.m, "b": 0.5}), 3.0 * ureg.m / ureg.s)
    assert_quantity_allclose(
        func({"a": 150.0 * ureg.cm, "b": 500.0 * ureg.ms}), 3.0 * ureg.m / ureg.s
    )
