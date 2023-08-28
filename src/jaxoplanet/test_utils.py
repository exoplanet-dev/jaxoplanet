from jax._src.public_test_util import check_close


def assert_allclose(calculated, expected, *args, **kwargs):
    check_close(calculated, expected, *args, **kwargs)


def assert_quantity_allclose(calculated, expected, *args, **kwargs):
    assert calculated.units == expected.units
    assert_allclose(calculated.magnitude, expected.magnitude, *args, **kwargs)
