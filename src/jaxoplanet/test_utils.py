from jax import tree_util
from jax._src.public_test_util import check_close
from jpu.core import is_quantity


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
    if not is_quantity(calculated) and not is_quantity(expected):
        assert_allclose(calculated, expected, *args, **kwargs)
    elif convert:
        assert_allclose(
            calculated.magnitude,
            expected.to(calculated.units).magnitude,
            *args,
            **kwargs,
        )
    else:
        assert calculated.units == expected.units
        assert_allclose(calculated.magnitude, expected.magnitude, *args, **kwargs)


def assert_quantity_pytree_allclose(
    calculated, expected, *args, is_leaf=is_quantity, **kwargs
):
    leaves1, treedef1 = tree_util.tree_flatten(calculated, is_leaf=is_leaf)
    leaves2, treedef2 = tree_util.tree_flatten(expected, is_leaf=is_leaf)
    assert treedef1 == treedef2
    for l1, l2 in zip(leaves1, leaves2):
        assert_quantity_allclose(l1, l2, *args, **kwargs)
