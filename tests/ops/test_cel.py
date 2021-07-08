# -*- coding: utf-8 -*-

import time

import cppimport
import jax
import jax.numpy as jnp
import numpy as np
from jax import config, test_util

from exoplanet_jax.ops.cel import cel


def test_cel():
    config.update("jax_enable_x64", True)

    limbdark = cppimport.imp("limbdark")

    args = (
        np.random.rand(50000).astype(np.float64),
        np.random.randn(50000).astype(np.float64),
        np.asarray(0.2).astype(np.float64),
        np.asarray(0.4).astype(np.float64),
    )
    f = jax.jit(cel)
    f(*args).block_until_ready()
    limbdark.cel_f64(*args)

    strt = time.time()
    for k in range(100):
        f(*args).block_until_ready()
    print(time.time() - strt)

    strt = time.time()
    for k in range(100):
        limbdark.cel_f64(*args)
    print(time.time() - strt)

    print(np.max(np.abs(f(*args) - limbdark.cel_f64(*args))))
    assert 0
