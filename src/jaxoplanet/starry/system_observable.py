from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from jaxoplanet.light_curves.utils import vectorize
from jaxoplanet.starry.orbit import SurfaceSystem
from jaxoplanet.types import Array, Quantity
from jaxoplanet.units import quantity_input, unit_registry as ureg


def system_observable(surface_observable, **kwargs):
    _surface_observable = partial(surface_observable, **kwargs)

    def observable_fun(
        system: SurfaceSystem,
    ) -> Callable[[Quantity], tuple[Array | None, Array | None]]:
        # the observable function of the central given the position and radii
        # of all other bodies
        central_bodies_observable = jax.vmap(
            _surface_observable, in_axes=(None, 0, 0, 0, 0, None)
        )

        # the observable function of all bodies combined given their position to the
        # central
        @partial(system.surface_vmap, in_axes=(0, 0, 0, 0, None))
        def compute_body_observable(surface, radius, x, y, z, time):
            if surface is None:
                return 0.0
            else:
                theta = surface.rotational_phase(time.magnitude)
                return _surface_observable(
                    surface,
                    (system.central.radius / radius).magnitude,
                    (x / radius).magnitude,
                    (y / radius).magnitude,
                    (z / radius).magnitude,
                    theta,
                )

        @quantity_input(time=ureg.day)
        @vectorize
        def obsrvable_impl(time: Quantity) -> Array:
            # a function that give the array of observables for all bodies, starting
            # with the central
            if system.central_surface is None:
                central_light_curves = jnp.array([0.0])
            else:
                theta = system.central_surface.rotational_phase(time.magnitude)
                central_radius = system.central.radius
                central_phase_curve = surface_observable(
                    system.central_surface, theta=theta
                )
                if len(system.bodies) > 0:
                    xos, yos, zos = system.relative_position(time)
                    n = len(xos.magnitude)
                    central_light_curves = central_bodies_observable(
                        system.central_surface,
                        (system.radius / central_radius).magnitude,
                        (xos / central_radius).magnitude,
                        (yos / central_radius).magnitude,
                        (zos / central_radius).magnitude,
                        theta,
                    )

                    if n > 1 and central_light_curves is not None:
                        central_light_curves = central_light_curves.sum(
                            0
                        ) - central_phase_curve * (n - 1)
                        central_light_curves = jnp.expand_dims(central_light_curves, 0)

                    body_light_curves = compute_body_observable(
                        system.radius, -xos, -yos, -zos, time
                    )

                    return jnp.hstack([central_light_curves, body_light_curves])
                else:
                    return jnp.array([central_phase_curve])

        return obsrvable_impl

    return observable_fun
