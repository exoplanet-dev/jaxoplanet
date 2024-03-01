(troubleshooting)=

# Troubleshooting

This page includes some tips for troubleshooting issues that you might run into
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

## Installation issues on ARM Macs

Follow these steps to install `jax` and `jaxoplanet` on your system, with
special instructions for Mac M1 chip or newer.

### Step 1: Check Your Python Architecture

First, verify your Python architecture to determine if you're on an ARM or
Intel-based chip.

```bash
import platform
platform.machine()
```

- If the output is "arm64", proceed with the following steps.
- If the output is "x86-64", you're running an Intel emulator on an Apple
  silicon chip and need to switch to arm64.

For the best performance of `jax`, it will be better to install Python under
the arm64 architecture.

**Install Miniforge for ARM64.**

Download Miniforge from the official Conda Forge page (https://conda-forge.org/miniforge/).
Run the installer script:

```bash
bash Mambaforge-23.11.0-0-MacOSX-arm64.sh
```

Restart the terminal.

### Step 2: Install `jax` and `jaxoplanet`

Create a New Environment (Optional) in miniforge.

It's a good practice to create a new environment for your projects:

```bash
conda create --name jaxoplanet
conda activate jaxoplanet
conda install pip

conda install jax
# or
python -m pip install "jax[cpu]"
```

Then install `jaxoplanet` using the instructions in {ref}`install`.
