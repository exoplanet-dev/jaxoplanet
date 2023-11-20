from importlib.resources import as_file, files

import jpu

unit_registry = jpu.UnitRegistry()
with as_file(
    files("jaxoplanet.units").joinpath("astro_constants_and_units.txt")
) as path:
    unit_registry.load_definitions(path)
