# Jaxoplanet

_Astronomical time series analysis with JAX_

---

_jaxoplanet_ is an implementation of many features from the
[exoplanet](https://docs.exoplanet.codes/en/latest/) and
[starry](https://starry.readthedocs.io/en/latest/) packages built on top of
[JAX](https://jax.readthedocs.io/en/latest/).

````{grid} 3
---
padding: 0
margin: 0
gutter: 0
---

```{grid-item-card} Custom operations
---
columns: 12 6 6 4
class-card: sd-border-0
shadow: None
---

*jaxoplanet* includes fast and robust implementations of many exoplanet-specific
operations, including solving Kepler's equation, and computing limb-darkened
light curves.
```

```{grid-item-card} Hardware acceleration
---
columns: 12 6 6 4
class-card: sd-border-0
shadow: None
---

Since all *jaxoplanet* operations are written in JAX, they have first-class
support for hardware acceleration using GPUs and TPUs.
```

```{grid-item-card} Software ecosystem
---
columns: 12 6 6 4
class-card: sd-border-0
shadow: None
---

*jaxoplanet* is built on top of JAX, which means that it integrates seamlessly
with modeling tools like [NumPyro](https://numpyro.readthedocs.io/en/latest/),
and [Flax](https://flax.readthedocs.io/en/latest/).
```

````

## Installation

```bash
python -m pip install jaxoplanet
```

## Table of contents

```{toctree}
---
maxdepth: 2
---

start
api
```
