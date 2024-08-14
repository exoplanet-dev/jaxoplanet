(doc:commonissues)=

# Common Issues

This page includes some tips for troubleshooting common issues that you might run into
when using `jaxoplanet`. This is a work-in-progress, so if you don't see your
issue listed here, feel free to open an issue on the [GitHub repository issue
tracker](https://github.com/exoplanet-dev/jaxoplanet/issues).

## NaNs and infinities

It's not that uncommon to hit NaNs or infinities when using `jax` and
`jaxoplanet`. This is often caused by numerical precision issues, and this can
be exacerbated by the fact that, by default, `jax` disables double precision
calculations. You can enable double precision [a few different ways as described
in the `jax`
docs](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision),
and the way we do it in these docs is to add the following, when necessary:

```python
import jax

jax.config.update("jax_enable_x64", True)
```

If enabling double precision doesn't do the trick, this often means that there's
an issue with the parameter or modeling choices that you're making. `jax`'s
["debug NaNs" mode](https://jax.readthedocs.io/en/latest/debugging/flags.html)
can help diagnose issues here.
