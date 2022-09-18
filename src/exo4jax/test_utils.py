from functools import partial

import jax
import numpy as np

assert_allclose = partial(
    np.testing.assert_allclose,
    atol=2e-5 if jax.config.jax_enable_x64 else 2e-4,
)
