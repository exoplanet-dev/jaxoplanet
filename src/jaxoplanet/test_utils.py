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


def assert_pytree_allclose(calculated, expected, *args, **kwargs):
    """
    Check that two Pytrees with floating point or arrays as leaves are equal
    within a dtype-dependent tolerance
    """
    leaves1, treedef1 = tree_util.tree_flatten(calculated)
    leaves2, treedef2 = tree_util.tree_flatten(expected)
    assert treedef1 == treedef2
    for l1, l2 in zip(leaves1, leaves2, strict=False):
        assert_allclose(l1, l2, *args, **kwargs)
