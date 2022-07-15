from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from exo4jax._src.types import Array


class TransitOrbit(NamedTuple):
    period: Array
    duration: Array
    time_transit: Array
    impact_param: Array
    radius: Array

    @classmethod
    def init(
        cls,
        *,
        period: Array,
        duration: Array,
        time_transit: Optional[Array] = None,
        impact_param: Optional[Array] = None,
        radius: Optional[Array] = None,
    ) -> "TransitOrbit":
        period, duration = jnp.broadcast_arrays(
            jnp.atleast_1d(period), jnp.atleast_1d(duration)
        )
        shape = period.shape
        return cls(
            period=period,
            duration=duration,
            time_transit=jnp.zeros_like(period)
            if time_transit is None
            else jnp.broadcast_to(time_transit, shape),
            impact_param=jnp.zeros_like(period)
            if impact_param is None
            else jnp.broadcast_to(impact_param, shape),
            radius=jnp.zeros_like(period)
            if radius is None
            else jnp.broadcast_to(radius, shape),
        )

    @property
    def central_radius(self) -> Array:
        return jnp.ones_like(self.period)

    def relative_position(
        self, t: Array, parallax: Optional[Array] = None
    ) -> Tuple[Array, Array, Array]:
        del parallax

        def impl(
            period: Array,
            duration: Array,
            time_transit: Array,
            impact_param: Array,
            radius: Array,
        ) -> Tuple[Array, Array, Array]:
            x2 = (1 + radius) ** 2 - impact_param**2
            speed = 2 * jnp.sqrt(x2) / duration
            half_period = 0.5 * period
            ref_time = time_transit - half_period
            dt = (t - ref_time) % period - half_period

            x = speed * dt
            y = jnp.broadcast_to(impact_param, dt.shape)
            m = jnp.abs(dt) < 0.5 * duration
            z = m * 1.0 - (~m) * 1.0
            return x, y, z

        return jax.vmap(impl)(*self)
