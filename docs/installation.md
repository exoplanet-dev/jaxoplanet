# Troubleshooting on `jaxoplanet` installation

Follow these steps to install `jax` and `jaxoplanet` on your system, with special instructions for Mac M1 chip or newer.

### Step 1: Check Your Python Architecture

First, verify your Python architecture to determine if you're on an ARM or Intel-based chip.

```bash
import platform
platform.machine()
```

- If the output is "arm64", proceed with the following steps.
- If the output is "x86-64", you're running an Intel emulator on an Apple silicon chip and need to switch to arm64.

For the best performance of `jax`, it will be better to install `python` under the arm64 archecture.

Install Miniforge for ARM64.

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
python -m pip install "jax[cpu]"
 ```

You have two options to install jaxoplanet:

Option 1: Direct Installation via `pip`

```bash
python -m pip install jaxoplanet
```

Option 2: Install from Source
If you prefer to install jaxoplanet from source:

Clone the jaxoplanet repository:
```bash
git clone https://github.com/exoplanet-dev/jaxoplanet.git
cd jaxoplanet
pip install .
```

You could install `jaxoplanet` and its depedencies with
```bash
pip install ".[docs]"
```

Follow these instructions to successfully install jax and jaxoplanet on your system.
