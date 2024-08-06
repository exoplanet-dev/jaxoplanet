import equinox as eqx
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg


class TransitOrbit(eqx.Module):
    """An orbit parameterized by the transit signal parameters.

    Args:
        period: Orbital periods of the planets [time unit].
        duration (Optional[Quantity]): Durations of transits [time unit].
            Either this or `speed` must be provided.
        speed (Optional[Quantity]): Speeds of the planets [length/time unit].
            Either this or `duration` must be provided.
        time_transit (Optional[Quantity]): The epochs of reference transits [time unit].
            Default is 0.
        impact_param (Optional[Quantity]): Impact parameters of the transits.
            Default is 0.
        radius_ratio (Optional[Quantity]): Ratio of planet radii to star radius.
            Default is 0.

    Raises:
        ValueError: If neither `speed` nor `duration` is provided.

    Properties:
        - shape: Returns the shape of the period, i.e. the number of planets.
        - central_radius: A quantity representing the radius of the central body.
            Its value is always 1 in this class, but required for compatibility reasons.
        - radius: An alias for radius_ratio required for compatibility reasons.
        - relative_position: The relative position of the orbiting body at a given time.
    """

    period: Quantity = units.field(units=ureg.d)
    speed: Quantity = units.field(units=1 / ureg.d)
    duration: Quantity = units.field(units=ureg.d)
    time_transit: Quantity = units.field(units=ureg.d)
    impact_param: Quantity = units.field(units=ureg.dimensionless)
    radius_ratio: Quantity = units.field(units=ureg.dimensionless)

    @units.quantity_input(
        period=ureg.d,
        duration=ureg.d,
        speed=1 / ureg.d,
        time_transit=ureg.d,
        impact_param=ureg.dimensionless,
        radius_ratio=ureg.dimensionless,
    )
    def __init__(
        self,
        *,
        period: Quantity,
        duration: Quantity | None = None,
        speed: Quantity | None = None,
        time_transit: Quantity | None = None,
        impact_param: Quantity | None = None,
        radius_ratio: Quantity | None = None,
    ):
        if duration is None:
            if speed is None:
                raise ValueError("Either 'speed' or 'duration' must be provided")
            self.period = period
            self.speed = speed
        else:
            self.period = period
            self.duration = duration

        if time_transit is None:
            self.time_transit = 0.0 * ureg.d
        else:
            self.time_transit = time_transit

        if impact_param is None:
            self.impact_param = 0.0 * ureg.dimensionless
        else:
            self.impact_param = impact_param

        if radius_ratio is None:
            self.radius_ratio = 0.0 * ureg.dimensionless
        else:
            self.radius_ratio = radius_ratio

        x2 = jnpu.square(1 + self.radius_ratio) - jnpu.square(self.impact_param)
        if duration is None:
            self.duration = 2 * jnpu.sqrt(jnpu.maximum(0, x2)) / self.speed
        else:
            self.speed = 2 * jnpu.sqrt(jnpu.maximum(0, x2)) / self.duration

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.shape(self.period)

    @property
    def radius(self) -> Quantity:
        return self.radius_ratio

    @property
    def central_radius(self) -> Quantity:
        return jnp.ones_like(self.period) * ureg.dimensionless

    @units.quantity_input(t=ureg.d, parallax=ureg.arcsec)
    def relative_position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        del parallax

        half_period = 0.5 * self.period
        ref_time = self.time_transit - half_period
        dt = jnpu.mod(t - ref_time, self.period) - half_period

        x = self.speed * dt
        y = jnpu.full_like(dt, self.impact_param)
        m = jnpu.fabs(dt) < 0.5 * self.duration
        z = (m * 1.0 - (~m) * 1.0) * ureg.dimensionless

        return x, y, z
