(doc:api)=

# API reference

These pages contain the API reference for the `jaxoplanet` package.

## Orbital systems

The modules to define orbital systems are:

- [keplerian](jaxoplanet.orbits.keplerian) : a module to define a Keplerian system made of a central object and its orbiting bodies.
- [transit](jaxoplanet.orbits.transit) : a module to define a system made of a star and its transiting planet, defined by the transit parameters assuming the planet transits at a constant velocity.

## Limb-darkened stars

Given an orbital system, the following modules can be used to compute different kind of light curves:

- [limb-darkened](jaxoplanet.light_curves.limb_dark) : a module to compute occultation light curve of a star with polynomial limb darkening.
- [transforms](jaxoplanet.light_curves.transforms) : a module providing decorators to transform light curve functions.


## Non-uniform surfaces


```{warning}
While being stable, computing *starry* light curves of non-uniform surfaces is still experimental.
```

The following modules can be used to create systems of bodies with limb-darkened and non-uniform emitting surfaces:

- [starry orbit](jaxoplanet.experimental.starry.orbit) : a module to define a system made of a central object and orbiting bodies, all with non-uniform emitting surfaces. (experimental)
- [starry light curve](jaxoplanet.experimental.starry.light_curves) : a module to compute the light curve of a non-uniform star whose surface is represented by a sum of spherical harmonics.

And the following are lower-level modules to define and manipulate non-uniform surfaces:

- [Ylm](jaxoplanet.experimental.starry.ylm) : a lower-level module to create and manipulate vectors in the spherical harmonic basis.
- [Pijk](jaxoplanet.experimental.starry.pijk) : a lower-level module to create and manipulate vectors in the polynomial basis.
- [Surface](jaxoplanet.experimental.starry.surface) : a module to manipulate the oriented surface of spherical bodies, represented by a sum of spherical harmonics.
- [visualization](jaxoplanet.experimental.starry.visualization) : a module to visualize the surface of non-uniform spherical bodies.


```{toctree}
:hidden:

keplerian orbit <autoapi/jaxoplanet/orbits/keplerian/index>
transit orbit <autoapi/jaxoplanet/orbits/transit/index>
starry orbit <autoapi/jaxoplanet/experimental/starry/orbit/index>
limb-darkened light curve <autoapi/jaxoplanet/light_curves/limb_dark/index>
transforms <autoapi/jaxoplanet/light_curves/transforms/index>
starry light curve <autoapi/jaxoplanet/experimental/starry/light_curves/index>
Surface <autoapi/jaxoplanet/experimental/starry/surface/index>
Ylm <autoapi/jaxoplanet/experimental/starry/ylm/index>
Pijk <autoapi/jaxoplanet/experimental/starry/pijk/index>
visualization <autoapi/jaxoplanet/experimental/starry/visualization/index>
```

**Missing something?** Check the [full API reference](autoapi/jaxoplanet/index).
