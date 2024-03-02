# jaxoplanet

_Astronomical time series analysis with JAX_

---

_jaxoplanet_ is a
[functional-programming](https://en.wikipedia.org/wiki/Functional_programming)-forward
implementation of many features from the
[exoplanet](https://docs.exoplanet.codes/en/latest/) and
[starry](https://starry.readthedocs.io/en/latest/) packages built on top of
[JAX](https://jax.readthedocs.io/en/latest/).

*jaxoplanet* includes fast and robust implementations of many exoplanet-specific
operations, including solving Kepler's equation, and computing limb-darkened
light curves. Since *jaxoplanet* is built on top of JAX it has first-class
support for hardware acceleration using GPUs and TPUs, and it also integrates
seamlessly with modeling tools like
[NumPyro](https://numpyro.readthedocs.io/en/latest/), and
[Flax](https://flax.readthedocs.io/en/latest/).

**For the most complete documentation, check out the documentation page at
[jax.exoplanet.codes](https://jax.exoplanet.codes).**

## Installation

You'll first need to install JAX following [the instructions in the JAX
docs](https://jax.readthedocs.io/en/latest/#installation). For example, to
install the CPU version of JAX, you can run:

```bash
python -m pip install "jax[cpu]"
```

Then install _jaxoplanet_ with:

```bash
python -m pip install jaxoplanet
```

If you run into issues with installing *jaxoplanet*, take a look at [the
installation instructions](https://jax.exoplanet.codes/en/latest/install).

## Quick start


## Attribution

While we don't yet have a citation for *jaxoplanet*, please reference the GitHub
repository if you find this code useful in your research. The BibTeX entry for
the repo is:

```
@software{jaxoplanet,
  author = {{Foreman-Mackey}, D. and {Garcia}, L.~J. and {Hattori}, S.
            and {Dong}, J. and {Murray}, C. and {Dholakia}, S. and {Degen}, D.},
  title = {{jaxoplanet}: Astronomical time series analysis with {JAX}},
  url = {http://github.com/exoplanet-dev/jaxoplanet},
  version = {0.0.2},
  year = {2024},
}
```

## License
Copyright (c) 2021-2024 Simons Foundation, Inc.

*jaxoplanet* is free software made available under the MIT License. For details
see the LICENSE file.
