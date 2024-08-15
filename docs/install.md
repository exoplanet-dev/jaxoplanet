(doc:install)=

# Installation

`jaxoplanet` is built on top of [`jax`](https://github.com/google/jax) so that's the
primary dependency that you'll need. All of the methods below will install any
required dependencies, but if you want to take advantage of your GPU, that might
take a little more setup. `jaxoplanet` doesn't have any GPU-specific code, so it
should be enough to just [follow the installation instructions for CUDA support
in the `jax` README](https://github.com/google/jax/#installation).

## Using pip

The easiest way to install the most recent stable version of `jaxoplanet` is
with [pip](https://pip.pypa.io):

```bash
python -m pip install jaxoplanet
```

## From source

Alternatively, you can get the source:

```bash
git clone https://github.com/exoplanet-dev/jaxoplanet.git
cd jaxoplanet
python -m pip install -e .
```

## Tests

If you installed from source, you can run the unit tests. From the root of the
source directory, run:

```bash
python -m pip install nox
python -m nox -s test
```

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

Then install `jaxoplanet` using the instructions above.
