from typing import Any, NamedTuple, Optional

import jax.numpy as jnp

Array = Any


GRAVITATIONAL_CONSTANT = 2942.2062175044193 / (4 * jnp.pi**2)


class KeplerianCentral(NamedTuple):
    mass: Array
    radius: Array
    density: Array

    @classmethod
    def init(cls, *, mass=None, radius=None, density=None):
        if radius is None and mass is None:
            radius = 1.0
            if density is None:
                mass = 1.0

        if sum(arg is None for arg in (mass, radius, density)) != 1:
            raise ValueError(
                "Values must be provided for exactly two of mass, radius, and density"
            )

        if density is None:
            density = 3 * mass / (4 * jnp.pi * radius**3)
        elif radius is None:
            radius = (3 * mass / (4 * jnp.pi * density)) ** (1 / 3)
        elif mass is None:
            mass = 4 * jnp.pi * radius**3 * density / 3.0

        return cls(mass=mass, radius=radius, density=density)

    @classmethod
    def from_orbital_properties(
        cls, *, period, semimajor, radius=None, body_mass=None
    ):
        radius = 1.0 if radius is None else radius

        mass = semimajor**3 / (GRAVITATIONAL_CONSTANT * period**2)
        if body_mass is not None:
            mass -= body_mass

        return cls.init(mass=mass, radius=radius)


class KeplerianBody(NamedTuple):
    ref_time: Array
    period: Array
    inclination: Array = 0.0
    eccentricity: Optional[Array] = None
    omega_peri: Optional[Array] = None
    asc_node: Optional[Array] = None
    mass: Optional[Array] = None
    radius: Optional[Array] = None

    @classmethod
    def init(
        cls,
        *,
        period=None,
        semimajor=None,
        mass=None,
        radius=None,
        central: KeplerianCentral = None,
    ):
        if period is not None and semimajor is not None:
            raise ValueError("")

        central = KeplerianCentral() if central is None else central

        mass_factor = GRAVITATIONAL_CONSTANT * central.mass
        if mass is not None:
            mass_factor += GRAVITATIONAL_CONSTANT * mass

        if semimajor is None:
            semimajor = (mass_factor * period**2) ** (1.0 / 3)
        elif period is None:
            period = semimajor**1.5 / jnp.sqrt(mass_factor)


class KeplerianOrbit(NamedTuple):
    central: KeplerianCentral
    bodies: KeplerianBody

    @classmethod
    def circular_from_duration(
        cls,
        *,
        ref_time,
        period,
        duration,
        impact_parameter,
        radius_ratio,
        central_radius=None,
        body_mass=None,
    ):
        # Get the semimajor axis impled by the duration, period,
        b2 = jnp.square(impact_parameter)
        opk2 = jnp.square(1 + radius_ratio)
        phi = jnp.pi * duration / period
        sinp = jnp.sin(phi)
        cosp = jnp.cos(phi)
        semimajor = jnp.sqrt(opk2 - b2 * cosp**2) / sinp

        central = KeplerianCentral.from_orbital_properties(
            period=period,
            semimajor=semimajor,
            radius=central_radius,
            body_mass=body_mass,
        )

        return cls(
            ref_time=ref_time,
            period=period,
        )


def make_inputs_consistent(*, a, period, rho_star, r_star, m_star, m_planet):
    if a is None and period is None:
        raise ValueError(
            "Values must be provided for at least one of a and period"
        )

    if m_planet is not None:
        m_planet = as_tensor_variable(to_unit(m_planet, u.M_sun))

    if a is not None:
        a = as_tensor_variable(to_unit(a, u.R_sun))
        if m_planet is None:
            m_planet = tt.zeros_like(a)
    if period is not None:
        period = as_tensor_variable(to_unit(period, u.day))
        if m_planet is None:
            m_planet = tt.zeros_like(period)

    # Compute the implied density if a and period are given
    implied_rho_star = False
    if a is not None and period is not None:
        if rho_star is not None or m_star is not None:
            raise ValueError(
                "if both a and period are given, you can't "
                "also define rho_star or m_star"
            )

        # Default to a stellar radius of 1 if not provided
        if r_star is None:
            r_star = as_tensor_variable(1.0)
        else:
            r_star = as_tensor_variable(to_unit(r_star, u.R_sun))

        # Compute the implied mass via Kepler's 3rd law
        m_tot = 4 * np.pi * np.pi * a**3 / (G_grav * period**2)

        # Compute the implied density
        m_star = m_tot - m_planet
        vol_star = 4 * np.pi * r_star**3 / 3.0
        rho_star = m_star / vol_star
        implied_rho_star = True

    # Make sure that the right combination of stellar parameters are given
    if r_star is None and m_star is None:
        r_star = 1.0
        if rho_star is None:
            m_star = 1.0
    if (not implied_rho_star) and sum(
        arg is None for arg in (rho_star, r_star, m_star)
    ) != 1:
        raise ValueError(
            "values must be provided for exactly two of "
            "rho_star, m_star, and r_star"
        )

    if rho_star is not None and not implied_rho_star:
        if has_unit(rho_star):
            rho_star = as_tensor_variable(
                to_unit(rho_star, u.M_sun / u.R_sun**3)
            )
        else:
            rho_star = as_tensor_variable(rho_star) / gcc_per_sun
    if r_star is not None:
        r_star = as_tensor_variable(to_unit(r_star, u.R_sun))
    if m_star is not None:
        m_star = as_tensor_variable(to_unit(m_star, u.M_sun))

    # Work out the stellar parameters
    if rho_star is None:
        rho_star = 3 * m_star / (4 * np.pi * r_star**3)
    elif r_star is None:
        r_star = (3 * m_star / (4 * np.pi * rho_star)) ** (1 / 3)
    elif m_star is None:
        m_star = 4 * np.pi * r_star**3 * rho_star / 3.0

    # Work out the planet parameters
    if a is None:
        a = (
            G_grav * (m_star + m_planet) * period**2 / (4 * np.pi**2)
        ) ** (1.0 / 3)
    elif period is None:
        period = (
            2 * np.pi * a ** (3 / 2) / (tt.sqrt(G_grav * (m_star + m_planet)))
        )

    return a, period, rho_star * gcc_per_sun, r_star, m_star, m_planet
