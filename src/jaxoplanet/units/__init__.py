from importlib.resources import as_file, files

import jpu

registry = jpu.UnitRegistry()
with as_file(
    files("jaxoplanet.units").joinpath("astro_constants_and_units.txt")
) as path:
    registry.load_definitions(path)
