"""
ttv.py

An implementation of a Transit Timing Variation (TTV) orbit in jaxoplanet.
This class extends the TransitOrbit class (from transit.py) by modifying the time‐axis
to account for transit timing variations.
"""
import jax 
import equinox as eqx
import jax.numpy as jnp
from jaxoplanet import units
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg
import jpu.numpy as jnpu
# Import the base TransitOrbit.
# (Adjust the import as needed if your package layout differs.)
from jaxoplanet.orbits import TransitOrbit


@units.quantity_input(
    period=ureg.d,
    duration=ureg.d,
    speed=1 / ureg.d,
    time_transit=ureg.d,
    impact_param=ureg.dimensionless,
    radius_ratio=ureg.dimensionless,
)
class TTVOrbit(TransitOrbit):
    """
    An extension of TransitOrbit that allows for transit timing variations (TTVs).

    The TTVs can be specified in one of two ways:
      - Provide a list of TTV offset arrays via the argument `ttvs`.
      - Provide a list of observed transit time arrays via `transit_times`
        (optionally with `transit_inds` to label each transit). In this case
        a least-squares linear fit is performed to infer a reference transit time (t₀)
        and period; the residuals define the TTVs.

    Only one of these two options should be provided.

    Additional Parameters:
      ttvs: list of Quantity arrays, each giving the “observed minus computed”
            transit time offsets (in days) for one planet.
      transit_times: list of Quantity arrays giving the observed transit times (in days).
      transit_inds: list of integer arrays (one per planet) labeling the transit numbers.
      delta_log_period: (optional) if using transit_times and the effective transit period
                        is to be slightly different from the period that governs the transit
                        shape, this parameter gives the offset in natural log.

    After initialization the transit times (whether provided or computed) are used to
    build a “binning” (i.e. a lookup table) that maps any observation time to the
    transit to which it is closest.
    """

    # These additional attributes are not wrapped as unit‐fields;
    # they are computed from the TTV input.
    ttvs: list[jnp.ndarray]
    transit_times: list[jnp.ndarray]
    transit_inds: list[jnp.ndarray]
    t0: jnp.ndarray  # Reference transit time(s)
    ttv_period: jnp.ndarray  # The effective period(s) as determined from transit_times.
    all_transit_times: list[jnp.ndarray]
    _bin_edges: list[jnp.ndarray]
    _bin_values: list[jnp.ndarray]

    def __init__(
        self,
        *,
        period: Quantity | None = None,
        duration: Quantity | None = None,
        speed: Quantity | None = None,
        time_transit: Quantity | None = None,
        impact_param: Quantity | None = None,
        radius_ratio: Quantity | None = None,
        # TTV-specific arguments:
        ttvs: list[Quantity] | None = None,
        transit_times: list[Quantity] | None = None,
        transit_inds: list[jnp.ndarray] | None = None,
        delta_log_period: float | None = None,
    ):
        # --- Auto-convert inputs if they’re not already proper Quantities ---
        period = ensure_quantity(period, ureg.d) if period is not None else None
        duration = ensure_quantity(duration, ureg.d) if duration is not None else None
        time_transit = ensure_quantity(time_transit, ureg.d) if time_transit is not None else None
        impact_param = ensure_quantity(impact_param, ureg.dimensionless) if impact_param is not None else None
        radius_ratio = ensure_quantity(radius_ratio, ureg.dimensionless) if radius_ratio is not None else None

        if transit_times is not None:
            # Make sure transit_times is a list
            if not isinstance(transit_times, (list, tuple)):
                transit_times = [transit_times]
            # Convert each transit time array to a Quantity (in days)
            transit_times = [ensure_quantity(tt, ureg.d) for tt in transit_times]
            
        # --- Process the TTV inputs ---
        if ttvs is None and transit_times is None:
            raise ValueError("One of 'ttvs' or 'transit_times' must be provided.")

        if transit_times is not None:
            # --- Case 1: transit_times are provided ---
            # Make sure transit_times is a list even if a single array is provided.
            if not isinstance(transit_times, (list, tuple)):
                transit_times = [transit_times]
            # Compute a least-squares linear fit (per planet) to infer the effective period and t₀.
            self.transit_times = []
            self.ttvs = []
            self.transit_inds = []
            t0_list = []
            period_list = []
            for i, times in enumerate(transit_times):
                times = jnp.atleast_1d(jnp.asarray(times.magnitude))
                if transit_inds is None:
                    inds = jnp.arange(times.shape[0])
                else:
                    inds = jnp.asarray(transit_inds[i], dtype=jnp.int64)
                self.transit_inds.append(inds)

                # Standard linear regression:
                N = times.shape[0]
                sumx = jnp.sum(inds)
                sumx2 = jnp.sum(inds ** 2)
                sumy = jnp.sum(times)
                sumxy = jnp.sum(inds * times)
                denom = N * sumx2 - sumx ** 2
                slope = (N * sumxy - sumx * sumy) / denom
                intercept = (sumx2 * sumy - sumx * sumxy) / denom
                expect = intercept + inds * slope

                period_list.append(slope)
                t0_list.append(intercept)
                self.ttvs.append(times - expect)
                self.transit_times.append(times)
            # After computing t0_list and period_list:
            self.t0 = jnp.stack(t0_list)       # originally shape (nplanets,)
            self.ttv_period = jnp.stack(period_list)  # originally shape (nplanets,)

            # For a single‐planet case, extract the scalar:
            if self.t0.shape[0] == 1:
                self.t0 = self.t0[0]
                self.ttv_period = self.ttv_period[0]

            # If period was not provided, use self.ttv_period (also squeezed if needed)
            if period is None:
                if delta_log_period is not None:
                    period = jnp.exp(jnp.log(self.ttv_period) + delta_log_period)
                else:
                    period = self.ttv_period

            # Now, set the reference transit time:
            time_transit = self.t0

        else:
            # --- Case 2: ttvs are provided (transit_times not provided) ---
            self.ttvs = [jnp.asarray(ttv) for ttv in ttvs]
            if transit_inds is None:
                self.transit_inds = [jnp.arange(ttv.shape[0]) for ttv in self.ttvs]
            else:
                self.transit_inds = [jnp.asarray(inds, dtype=jnp.int64) for inds in transit_inds]

            # Use the provided time_transit (or default to 0) as the reference transit time.
            if time_transit is None:
                time_transit = 0.0 * ureg.d
            self.t0 = time_transit

            # For ttvs mode, the period must be provided.
            if period is None:
                raise ValueError("When using ttvs, 'period' must be provided.")
            self.ttv_period = period

        # --- Initialize the base TransitOrbit ---
        # (This will compute speed or duration as in transit.py.)
        super().__init__(
            period=period,
            duration=duration,
            speed=speed,
            time_transit=time_transit,
            impact_param=impact_param,
            radius_ratio=radius_ratio,
        )

        # If only ttvs were provided, compute the (observed) transit times
        if transit_times is None:
            self.transit_times = []
            for i, ttv in enumerate(self.ttvs):
                # Here the transit times are given by: t₀ + period * transit_number + ttv.
                # (Note: if there is more than one planet, ensure that self.t0 and self.period have the correct shapes.)
                if jnp.ndim(self.t0) > 0:
                    t0_i = self.t0[i]
                else:
                    t0_i = self.t0
                if jnp.ndim(self.period) > 0:
                    period_i = self.period[i]
                else:
                    period_i = self.period
                times = t0_i + period_i * self.transit_inds[i] + ttv
                self.transit_times.append(times)

        # --- Build the lookup table for warping times ---
        self.all_transit_times = []
        self._bin_edges = []
        self._bin_values = []
        for i, inds in enumerate(self.transit_inds):
            # For each planet, we first form the full set of transit times by starting with
            # the expected (regular) times and then “inserting” the observed times at the
            # observed indices.
            if jnp.ndim(self.t0) > 0:
                t0_i = self.t0[i] * (1.0 * ureg.d)
            else:
                t0_i = self.t0 * (1.0 * ureg.d)
            if jnp.ndim(self.period) > 0:
                period_i = self.period[i]
            else:
                period_i = self.period

            max_ind = inds.shape[0] - 1  # use the static shape if available
            expected = t0_i + period_i * jnp.arange(max_ind + 1)

            # Replace the expected times at the observed transit indices with the (possibly warped) transit_times.
            full_transits = expected.at[inds].set(self.transit_times[i])
            self.all_transit_times.append(full_transits)

            # Now define bin edges around each transit. 
            half_period = 0.5 * self.ttv_period if jnp.ndim(self.ttv_period) == 0 else 0.5 * self.ttv_period[i]
            if full_transits.shape[0] > 1:
                midpoints = 0.5 * (full_transits[1:] + full_transits[:-1])
                bin_edges = jnp.concatenate(
                    (
                        jnp.array([full_transits[0] - half_period]),
                        midpoints,
                        jnp.array([full_transits[-1] + half_period]),
                    )
                )
            else:
                bin_edges = jnp.array([full_transits[0] - half_period, full_transits[0] + half_period])
            self._bin_edges.append(bin_edges)

            if full_transits.shape[0] > 1:
                bin_values = jnp.concatenate(
                    (
                        jnp.array([full_transits[0]]),
                        full_transits,
                        jnp.array([full_transits[-1]]),
                    )
                )
            else:
                bin_values = jnp.array([full_transits[0], full_transits[0]])
            self._bin_values.append(bin_values)
    @property
    def linear_t0(self):
        """
        Return the reference transit time(s) as a magnitude (unitless)
        in a 1D array (even if there is only one planet).
        """
        if hasattr(self.t0, "magnitude"):
            return jnp.atleast_1d(self.t0.magnitude)
        else:
            return jnp.atleast_1d(self.t0)

    @property
    def linear_period(self):
        """
        Return the effective period(s) (e.g. the inferred ttv_period)
        as a magnitude (unitless) in a 1D array (even if there is only one planet).
        """
        if hasattr(self.ttv_period, "magnitude"):
            return jnp.atleast_1d(self.ttv_period.magnitude)
        else:
            return jnp.atleast_1d(self.ttv_period)


    def _get_model_dt(self, t: Quantity) -> jnp.ndarray:
        dt_list = []
        # Convert the input time to a plain JAX array (in days)
        t_magnitude = jnp.asarray(t.to(ureg.d).magnitude)
        for i in range(len(self._bin_edges)):
            # Convert bin_edges to a plain array (in days)
            bin_edges = jnp.asarray(self._bin_edges[i].to(ureg.d).magnitude if hasattr(self._bin_edges[i], 'to') else self._bin_edges[i])
            inds = jnp.searchsorted(bin_edges, t_magnitude)
            # Convert the corresponding bin_values to a plain array (in days)
            bin_values = jnp.asarray(self._bin_values[i].to(ureg.d).magnitude if hasattr(self._bin_values[i], 'to') else self._bin_values[i])
            dt_i = bin_values[inds]
            dt_list.append(dt_i)
        # Stack the list and reattach the unit
        return jnp.stack(dt_list, axis=-1) * ureg.d

    def _warp_times(self, t: Quantity) -> Quantity:
        dt = self._get_model_dt(t)
        if dt.shape[-1] == 1:
            dt = dt[..., 0]
        # Instead of subtracting dt directly, subtract (dt - self.t0)
        # so that if dt == self.t0 then warped time equals self.t0.
        return t - (dt - self.t0 * ureg.d)


    @units.quantity_input(t=ureg.d, parallax=ureg.arcsec)
    def relative_position(
        self, t: Quantity, parallax: Quantity | None = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        warped_t = self._warp_times(t)
        x, y, z = super().relative_position(warped_t, parallax=parallax)
        # Squeeze out extra axes if present (e.g. if shape is (N,1) make it (N,))
        return jnpu.squeeze(x), jnpu.squeeze(y), jnpu.squeeze(z)


def compute_expected_transit_times(min_time, max_time, period, t0):
    """
    Compute the expected transit times within a dataset    

    Args:
        min_time (float): The start time of the dataset
        max_time (float): The end time of the dataset
        period (array):  The periods of the planets
        t0 (array):  The reference transit times for the planets
    """
    periods = jnp.atleast_1d(period)
    t0s = jnp.atleast_1d(t0)
    transit_times = []
    for period, t0 in zip(periods, t0s):
        min_ind = jnp.floor((min_time - t0) / period)
        max_ind = jnp.ceil((max_time - t0) / period)
        times = t0 + period * jnp.arange(min_ind, max_ind, 1)
        times = times[(min_time <= times) & (times <= max_time)]
        transit_times.append(times)
    return transit_times
def ensure_quantity(x, unit):
    """Convert x to a JAX array with the given unit if it isn’t already a quantity."""
    if hasattr(x, "to"):
        return x
    else:
        return jnp.asarray(x) * unit

