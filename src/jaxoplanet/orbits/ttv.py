import jax
import jax.numpy as jnp

from jaxoplanet.orbits import TransitOrbit
from jaxoplanet.types import Scalar


@jax.jit
def _compute_linear_ephemeris_single(transit_times: jnp.ndarray, indices: jnp.ndarray):
    """Compute linear ephemeris parameters for a single planet using least squares
    fitting."""
    n = transit_times.shape[0]
    X = jnp.vstack([jnp.ones(n), indices])
    beta = jnp.linalg.solve(X @ X.T, X @ transit_times)
    intercept, slope = beta
    ttvs = transit_times - (intercept + slope * indices)
    return intercept, slope, ttvs


@jax.jit
def _compute_bin_edges_values_single(tt: jnp.ndarray, period: float):
    """Compute bin edges and values for a single planet."""
    midpoints = 0.5 * (tt[1:] + tt[:-1])
    bin_edges = jnp.concatenate(
        [
            jnp.array([tt[0] - 0.5 * period]),
            midpoints,
            jnp.array([tt[-1] + 0.5 * period]),
        ]
    )
    bin_values = jnp.concatenate([jnp.array([tt[0]]), tt, jnp.array([tt[-1]])])
    return bin_edges, bin_values


@jax.jit
def _process_planet_dt_single(
    bin_edges: jnp.ndarray, bin_values: jnp.ndarray, t_mag: jnp.ndarray
):
    """Process dt values for a single planet."""
    inds = jnp.searchsorted(bin_edges, t_mag)
    return bin_values[inds]


class TTVOrbit(TransitOrbit):
    """An extension of TransitOrbit that allows for transit timing variations (TTVs).

    The TTVs can be specified in one of two ways:

    - Provide a tuple (or list) of TTV offset arrays via the argument `ttvs`.
    - Provide a tuple (or list) of observed transit time arrays via `transit_times`
        (optionally with `transit_inds` to label each transit). In this case
        a least-squares linear fit is performed to infer a reference transit time (t₀)
        and period; the residuals define the TTVs.

    Only one of these two options should be provided.

    args added on to TransitOrbit:

    - ttvs: tuple (or list) of Scalar arrays, each giving the “observed minus
        computed” transit time offsets (in days) for one planet.
    - transit_times: tuple (or list) of Scalar arrays giving the observed transit times
         (in days).
    - transit_inds: tuple (or list) of integer arrays (one per planet) labeling the
        transit numbers.
    - delta_log_period: (optional) if using transit_times and the effective transit
        period is to be slightly different from the period that governs the transit
        shape, this parameter gives the offset in natural log.

    """

    transit_times: tuple[jnp.ndarray, ...]
    transit_inds: tuple[
        jnp.ndarray, ...
    ]  # e.g. (jnp.arange(n_transits0), jnp.arange(n_transits1), ...)
    ttvs: tuple[
        jnp.ndarray, ...
    ]  # Residuals (observed minus linear model) for each planet
    t0: jnp.ndarray  # Reference transit times (one per planet)
    ttv_period: jnp.ndarray  # Inferred effective periods (one per planet)

    _bin_edges: tuple[jnp.ndarray, ...]
    _bin_values: tuple[jnp.ndarray, ...]

    def __init__(
        self,
        *,
        period: Scalar | None = None,
        duration: Scalar | None = None,
        speed: Scalar | None = None,
        time_transit: Scalar | None = None,
        impact_param: Scalar | None = None,
        radius_ratio: Scalar | None = None,
        transit_times: tuple[jnp.ndarray, ...] = None,
        transit_inds: tuple[jnp.ndarray, ...] | None = None,
        ttvs: tuple[Scalar, ...] | None = None,
        delta_log_period: float | None = None,
    ):

        if ttvs is not None and transit_times is not None:
            raise ValueError("Supply either ttvs or transit_times, not both.")
        if ttvs is None and transit_times is None:
            raise ValueError("You must supply either transit_times or ttvs.")

        # CASE 1: transit_times are provided
        if transit_times is not None:
            self.transit_times = tuple(jnp.atleast_1d(tt) for tt in transit_times)
            if transit_inds is None:
                self.transit_inds = tuple(
                    jnp.arange(tt.shape[0]) for tt in self.transit_times
                )
            else:
                self.transit_inds = tuple(transit_inds)

            t0_list = []
            period_list = []
            ttvs_list = []
            for tt, inds in zip(self.transit_times, self.transit_inds, strict=False):
                t0_i, period_i, ttv_i = _compute_linear_ephemeris_single(tt, inds)
                t0_list.append(t0_i)
                period_list.append(period_i)
                ttvs_list.append(ttv_i)
            self.t0 = jnp.array(t0_list)
            self.ttv_period = jnp.array(period_list)
            self.ttvs = tuple(ttvs_list)
            if time_transit is None:
                time_transit = self.t0
                # --- Begin delta_log_period adjustment ---
            # If a period was not provided and a delta_log_period is provided,
            # adjust the computed period accordingly.
            if period is None:

                if delta_log_period is not None:
                    # Compute the adjusted period using delta_log_period
                    period = jnp.exp(jnp.log(self.ttv_period) + delta_log_period)
                else:
                    period = self.ttv_period
        else:
            # Here, the user must supply period and time_transit.
            if period is None:
                raise ValueError("When supplying ttvs, period must be provided.")
            period = ensure_Scalar(period, ureg.d)
            if time_transit is None:
                # If time_transit is not provided, assume t0 = 0
                time_transit = 0.0 * ureg.d
            time_transit = ensure_Scalar(time_transit, ureg.d)

            # In the TTVs branch of __init__
            self.ttvs = tuple(
                jnp.atleast_1d(
                    ensure_Scalar(ttv, ureg.d) - jnp.mean(ensure_Scalar(ttv, ureg.d))
                )
                for ttv in ttvs
            )

            # For each planet, define transit_inds based on the shape of ttvs if not
            # provided.
            if transit_inds is None:
                self.transit_inds = tuple(jnp.arange(ttv.shape[0]) for ttv in self.ttvs)
            else:
                self.transit_inds = tuple(transit_inds)
            self.t0 = jnp.atleast_1d(time_transit)
            # For ttvs mode, period is required and used directly:
            self.ttv_period = jnp.atleast_1d(period)
            # Reconstruct transit_times: t0 + period * inds + ttvs
            transit_times_list = []
            for ttv, inds in zip(self.ttvs, self.transit_inds, strict=False):
                # Handle single-planet t0 and period versus multi-planet cases.
                if self.t0.ndim > 0 and self.t0.shape[0] > 1:
                    t0_i = self.t0[len(transit_times_list)]
                    period_i = self.ttv_period[len(transit_times_list)]
                else:
                    t0_i = self.t0.item() if hasattr(self.t0, "item") else self.t0
                    period_i = (
                        self.ttv_period.item()
                        if hasattr(self.ttv_period, "item")
                        else self.ttv_period
                    )
                transit_times_list.append(t0_i + period_i * inds + ttv)
            self.transit_times = tuple(transit_times_list)

        # Initialize the base TransitOrbit.
        super().__init__(
            period=period if period is not None else self.ttv_period,
            duration=duration,
            speed=speed,
            time_transit=time_transit,
            impact_param=impact_param,
            radius_ratio=radius_ratio,
        )

        # Compute bins separately for each planet
        bin_edges_list = []
        bin_values_list = []

        for tt, period in zip(self.transit_times, self.ttv_period, strict=False):
            edges, values = _compute_bin_edges_values_single(tt, period)
            bin_edges_list.append(edges)
            bin_values_list.append(values)

        self._bin_edges = tuple(bin_edges_list)
        self._bin_values = tuple(bin_values_list)

    @jax.jit
    def _get_model_dt(self, t: Scalar) -> jnp.ndarray:
        """Get model dt values for time warping."""
        t_magnitude = jnp.asarray(t)
        dt_list = []
        for edges, values in zip(self._bin_edges, self._bin_values, strict=False):
            edges_mag = edges if hasattr(edges, "to") else edges
            values_mag = values if hasattr(values, "to") else values

            dt = _process_planet_dt_single(edges_mag, values_mag, t_magnitude)
            dt_list.append(dt)

        return jnp.stack(dt_list)

    @jax.jit
    def _warp_times(self, t: Scalar) -> Scalar:
        """Warp times based on transit timing variations."""
        dt = self._get_model_dt(t)
        t0_days = self.t0 if hasattr(self.t0, "to") else self.t0
        return t - (dt - t0_days)

    def relative_position(
        self, t: Scalar, parallax: Scalar | None = None
    ) -> tuple[Scalar, Scalar, Scalar]:
        """Compute relative position of the planet(s)."""
        warped_t = self._warp_times(t)
        x, y, z = super().relative_position(warped_t, parallax=parallax)
        return (
            jnp.squeeze(x),
            jnp.squeeze(y),
            jnp.squeeze(z),
        )

    @property
    def linear_t0(self):
        """Return the linear reference transit time."""
        if hasattr(self.t0, "magnitude"):
            return jnp.atleast_1d(self.t0)
        else:
            return jnp.atleast_1d(self.t0)

    @property
    def linear_period(self):
        """Return the linear period."""
        if hasattr(self.ttv_period, "magnitude"):
            return jnp.atleast_1d(self.ttv_period)
        else:
            return jnp.atleast_1d(self.ttv_period)


def compute_expected_transit_times(min_time, max_time, period, t0):
    """
    Compute expected transit times for each planet and return them as a tuple of 1D
    arrays.

    Args:
        min_time (float): Start time (in days).
        max_time (float): End time (in days).
        period (array-like): Orbital period for each planet (in days). Should be
            convertible to a 1D JAX array.
        t0 (array-like): Reference transit times for each planet (in days). Should be
            convertible to a 1D JAX array.

    Returns:
        transit_times: tuple of JAX arrays, one per planet.
    """
    period = jnp.atleast_1d(period)
    t0 = jnp.atleast_1d(t0)

    transit_times_list = []
    for p, t0_val in zip(period, t0, strict=False):
        i_min = int(jnp.ceil((min_time - t0_val) / p))
        i_max = int(jnp.floor((max_time - t0_val) / p))
        indices = jnp.arange(i_min, i_max + 1)
        times = t0_val + p * indices
        times = times[(times >= min_time) & (times <= max_time)]
        transit_times_list.append(times)

    return tuple(transit_times_list)
