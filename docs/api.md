# API reference

These pages contain the API reference for the `jaxoplanet` package.

```{warning}
While being stable, the rest of the API is still subject to change. This is particularly true for the `experimental.starry` module.
```

## Orbital systems

The modules to define orbital systems are:

- [keplerian](jaxoplanet.orbits.keplerian) : a module to define systems made of a central object and its orbiting bodies.
- [transit](jaxoplanet.orbits.transit) : a module to define a system made of a star and its transiting planet, defined by the transit parameters.
- [starry](jaxoplanet.experimental.starry.orbit) : a module to define a system made of a central object and orbiting bodies, all with non-uniform emitting surfaces. (experimental)

## Light curves

Given an orbital system, the following modules can be used to compute different kind of light-curves light curves:

- [limb-darkened](jaxoplanet.light_curves.limb_dark) : a module to compute occultation light curve of a star with polynomial limb darkening.
- [starry](jaxoplanet.experimental.starry.surface) : a module to compute the light curve of a non-uniform star whose surface is represented by a sum of spherical harmonics.

## Surface maps

The following module can be used to create and manipulate non-uniform surface maps:

- [Ylm](jaxoplanet.experimental.starry.ylm) : a lower-level module to create and manipulate vectors in the spherical harmonic basis.
- [Pijk](jaxoplanet.experimental.starry.pijk) : a lower-level module to create and manipulate vectors in the polynomial basis.
- [Surface](jaxoplanet.experimental.starry.ylm) : a module to manipulate the oriented surface of spherical bodies, represented by a sum of spherical harmonics.
- [visualization](jaxoplanet.experimental.starry.visualization) : a module to visualize the surface of non-uniform spherical bodies.

```{toctree}
:hidden:

keplerian orbit <autoapi/jaxoplanet/orbits/keplerian/index>
transit orbit <autoapi/jaxoplanet/orbits/transit/index>
starry orbit <autoapi/jaxoplanet/experimental/starry/orbit/index>
limb-darkened light curve <autoapi/jaxoplanet/light_curves/limb_dark/index>
starry light curve <autoapi/jaxoplanet/experimental/starry/surface/index>
Ylm <autoapi/jaxoplanet/experimental/starry/ylm/index>
Pijk <autoapi/jaxoplanet/experimental/starry/pijk/index>
visualization <autoapi/jaxoplanet/experimental/starry/visualization/index>

```