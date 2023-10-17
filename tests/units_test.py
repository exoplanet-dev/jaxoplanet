from typing import Any

import equinox as eqx
import jax
import pytest
from pint import DimensionalityError

from jaxoplanet import units
from jaxoplanet.test_utils import assert_quantity_allclose
from jaxoplanet.units import unit_registry as ureg


def test_quantity_input():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        assert a.units == ureg.m
        assert b.units == ureg.s
        return a / b

    assert_quantity_allclose(func(1.5 * ureg.m, 0.5 * ureg.s), 3.0 * ureg.m / ureg.s)


def test_quantity_input_functional():
    def func(a, b):
        assert a.units == ureg.m
        assert b.units == ureg.s
        return a / b

    func = units.quantity_input(func, a=ureg.m, b=ureg.s)
    assert_quantity_allclose(func(1.5 * ureg.m, 0.5 * ureg.s), 3.0 * ureg.m / ureg.s)


def test_quantity_input_string():
    @units.quantity_input(a="m", b="s")
    def func(a, b):
        assert a.units == ureg.m
        assert b.units == ureg.s
        return a / b

    assert_quantity_allclose(func(1.5 * ureg.m, 0.5 * ureg.s), 3.0 * ureg.m / ureg.s)


def test_quantity_input_varargs():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b, *args, **kwargs):
        assert a.units == ureg.m
        assert b.units == ureg.s
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


@pytest.mark.parametrize("with_jit", [False, True])
def test_quantity_input_pytree(with_jit):
    @units.quantity_input(params={"a": ureg.m, "b": ureg.s})
    def func(params):
        assert params["a"].units == ureg.m
        assert params["b"].units == ureg.s
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


def test_quantity_input_invalid():
    @units.quantity_input(a=ureg.m, b=ureg.s)
    def func(a, b):
        return a / b

    with pytest.raises(DimensionalityError):
        func(150.0 * ureg.hr, 0.5 * ureg.s)


def test_quantity_input_unrecognized_argument():
    with pytest.raises(TypeError, match="got an unexpected keyword argument 'x'"):

        @units.quantity_input(a=ureg.m, b=ureg.s, x=ureg.cm)
        def func(a, b):
            return a / b


def test_quantity_input_positional_unit():
    with pytest.raises(TypeError):

        @units.quantity_input(ureg.m, b=ureg.s)  # type: ignore
        def func(a, b):
            return a / b


def test_quantity_input_varargs_error():
    with pytest.raises(TypeError):

        @units.quantity_input(a=ureg.m, b=ureg.s, args=ureg.cm)
        def func(a, b, *args, **kwargs):
            assert a.units == ureg.m
            assert b.units == ureg.s
            return a / b


def test_field():
    class Model(eqx.Module):
        x: Any = units.field(units=ureg.m)

    assert_quantity_allclose(Model(x=1.5).x, 1.5 * ureg.m)
    assert_quantity_allclose(Model(x=1.5 * ureg.m).x, 1.5 * ureg.m)
    assert_quantity_allclose(Model(x=150.0 * ureg.cm).x, 1.5 * ureg.m)


def test_field_converter():
    class Model(eqx.Module):
        x: Any = units.field(units=ureg.m, converter=lambda x: 2 * x)

    assert_quantity_allclose(Model(x=1.5).x, 3 * ureg.m)


def test_field_pytree():
    class Model(eqx.Module):
        x: Any = units.field(units={"a": ureg.m, "b": ureg.s})

    model = Model(x={"a": 150.0 * ureg.cm, "b": 0.5})
    assert_quantity_allclose(model.x["a"], 1.5 * ureg.m)
    assert_quantity_allclose(model.x["b"], 0.5 * ureg.s)


def test_field_optional():
    class Model(eqx.Module):
        x: Any = units.field(units=ureg.m)

    model = Model(x=None)
    assert model.x is None
    assert_quantity_allclose(Model(x=1.5).x, 1.5 * ureg.m)
