import equinox as eqx
import jax.numpy as jnp

from jaxoplanet import units
from jaxoplanet.types import Scalar


class TransitOrbit(eqx.Module):
    """An orbit parameterized by the transit signal parameters.

    Args:
        period: Orbital periods of the planets [time unit].
        duration (Optional[Scalar]): Durations of transits [time unit].
            Either this or `speed` must be provided.
        speed (Optional[Scalar]): Speeds of the planets [length/time unit].
            Either this or `duration` must be provided.
        time_transit (Optional[Scalar]): The epochs of reference transits [time unit].
            Default is 0.
        impact_param (Optional[Scalar]): Impact parameters of the transits.
            Default is 0.
        radius_ratio (Optional[Scalar]): Ratio of planet radii to star radius.
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

    period: Scalar
    speed: Scalar
    duration: Scalar
    time_transit: Scalar
    impact_param: Scalar
    radius_ratio: Scalar

    def __init__(
        self,
        *,
        period: Scalar,
        duration: Scalar | None = None,
        speed: Scalar | None = None,
        time_transit: Scalar | None = None,
        impact_param: Scalar | None = None,
        radius_ratio: Scalar | None = None,
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
            self.time_transit = 0.0
        else:
            self.time_transit = time_transit

        if impact_param is None:
            self.impact_param = 0.0
        else:
            self.impact_param = impact_param

        if radius_ratio is None:
            self.radius_ratio = 0.0
        else:
            self.radius_ratio = radius_ratio

        x2 = jnp.square(1 + self.radius_ratio) - jnp.square(self.impact_param)
        if duration is None:
            self.duration = 2 * jnp.sqrt(jnp.maximum(0, x2)) / self.speed
        else:
            self.speed = 2 * jnp.sqrt(jnp.maximum(0, x2)) / self.duration

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.shape(self.period)

    @property
    def radius(self) -> Scalar:
        return self.radius_ratio

    @property
    def central_radius(self) -> Scalar:
        return jnp.ones_like(self.period)

    def relative_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        del parallax

        half_period = 0.5 * self.period
        ref_time = self.time_transit - half_period
        dt = jnp.mod(t - ref_time, self.period) - half_period

        x = self.speed * dt
        y = jnp.full_like(dt, self.impact_param)
        m = jnp.fabs(dt) < 0.5 * self.duration
        z = m * 1.0 - (~m) * 1.0

        return x, y, z
