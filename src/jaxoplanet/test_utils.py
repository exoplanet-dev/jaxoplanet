from jax._src.public_test_util import check_close


def assert_allclose(calculated, expected, *args, **kwargs):
    kwargs["rtol"] = kwargs.get(
        "rtol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    check_close(calculated, expected, *args, **kwargs)


def assert_quantity_allclose(calculated, expected, *args, convert=False, **kwargs):
    if convert:
        assert_allclose(
            calculated.magnitude,
            expected.to(calculated.units).magnitude,
            *args,
            **kwargs,
        )
    else:
        assert calculated.units == expected.units
        assert_allclose(calculated.magnitude, expected.magnitude, *args, **kwargs)
