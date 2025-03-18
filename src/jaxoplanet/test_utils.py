__all__ = ["assert_allclose"]

from jax import tree_util
from jax._src.public_test_util import check_close


def assert_allclose(calculated, expected, *args, **kwargs):
    """
    Check that two floating point arrays are equal within a dtype-dependent tolerance
    """
    kwargs["rtol"] = kwargs.get(
        "rtol",
        {
            "float32": 5e-4,
            "float64": 5e-7,
        },
    )
    check_close(calculated, expected, *args, **kwargs)
