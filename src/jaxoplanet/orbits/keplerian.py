from typing import Any, Optional

import equinox as eqx
import jax
import jax.numpy as jnp
import jpu.numpy as jnpu

from jaxoplanet import units
from jaxoplanet.core.kepler import kepler
from jaxoplanet.types import Quantity
from jaxoplanet.units import unit_registry as ureg


class Central(eqx.Module):
    mass: Quantity = units.field(units=ureg.M_sun)
    radius: Quantity = units.field(units=ureg.R_sun)
    density: Quantity = units.field(units=ureg.M_sun / ureg.R_sun**3)

    @units.quantity_input(
        mass=ureg.M_sun, radius=ureg.R_sun, density=ureg.M_sun / ureg.R_sun**3
    )
    def __init__(
        self,
        *,
        mass: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
        density: Optional[Quantity] = None,
    ):
        if radius is None and mass is None:
            radius = 1.0 * ureg.R_sun
            if density is None:
                mass = 1.0 * ureg.M_sun

        # Check that all the input values are scalars; we don't support Scalars
        # here
        if any(
            jnp.ndim(arg) != 0 for arg in (mass, radius, density) if arg is not None
        ):
            raise ValueError("All parameters of a KeplerianCentral must be scalars")

        # Compute all three parameters based on the input values
        error_msg = (
            "Values must be provided for exactly two of mass, radius, and density"
        )
        if density is None:
            if mass is None or radius is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = radius
            self.density = 3 * mass / (4 * jnp.pi * radius**3)
        elif radius is None:
            if mass is None or density is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = (3 * mass / (4 * jnp.pi * density)) ** (1 / 3)
            self.density = density
        elif mass is None:
            if radius is None or density is None:
                raise ValueError(error_msg)
            self.mass = 4 * jnp.pi * radius**3 * density / 3.0
            self.radius = radius
            self.density = density

    @classmethod
    def from_orbital_properties(
        cls,
        *,
        period: Quantity,
        semimajor: Quantity,
        radius: Optional[Quantity] = None,
        body_mass: Optional[Quantity] = None,
    ) -> "Central":
        if jnp.ndim(semimajor) != 0:
            raise ValueError(
                "The 'semimajor' argument to "
                "'KeplerianCentral.from_orbital_properties' "
                "must be a scalar; for multi-planet systems, "
                "use 'jax.vmap'"
            )

        radius = 1.0 * ureg.R_sun if radius is None else radius

        mass = semimajor**3 / (ureg.gravitational_constant * period**2)
        if body_mass is not None:
            mass -= body_mass

        return cls(mass=mass, radius=radius)


class Body(eqx.Module):
    time_ref: Quantity = units.field(units=ureg.d)
    time_transit: Quantity = units.field(units=ureg.d)
    period: Quantity = units.field(units=ureg.d)
    semimajor: Quantity = units.field(units=ureg.R_sun)
    sin_inclination: Quantity = units.field(units=ureg.dimensionless)
    cos_inclination: Quantity = units.field(units=ureg.dimensionless)
    impact_param: Quantity = units.field(units=ureg.dimensionless)
    mass: Optional[Quantity] = units.field(units=ureg.M_sun)
    radius: Optional[Quantity] = units.field(units=ureg.R_sun)
    eccentricity: Optional[Quantity] = units.field(units=ureg.dimensionless)
    sin_omega_peri: Optional[Quantity] = units.field(units=ureg.dimensionless)
    cos_omega_peri: Optional[Quantity] = units.field(units=ureg.dimensionless)
    sin_asc_node: Optional[Quantity] = units.field(units=ureg.dimensionless)
    cos_asc_node: Optional[Quantity] = units.field(units=ureg.dimensionless)

    @units.quantity_input(
        time_transit=ureg.d,
        time_peri=ureg.d,
        period=ureg.d,
        semimajor=ureg.R_sun,
        inclination=ureg.radian,
        impact_param=ureg.dimensionless,
        eccentricity=ureg.dimensionless,
        omega_peri=ureg.radian,
        sin_omega_peri=ureg.dimensionless,
        cos_omega_peri=ureg.dimensionless,
        asc_node=ureg.radian,
        sin_asc_node=ureg.dimensionless,
        cos_asc_node=ureg.dimensionless,
        mass=ureg.M_sun,
        radius=ureg.R_sun,
    )
    def __init__(
        self,
        central: Central,
        *,
        time_transit: Optional[Quantity] = None,
        time_peri: Optional[Quantity] = None,
        period: Optional[Quantity] = None,
        semimajor: Optional[Quantity] = None,
        inclination: Optional[Quantity] = None,
        impact_param: Optional[Quantity] = None,
        eccentricity: Optional[Quantity] = None,
        omega_peri: Optional[Quantity] = None,
        sin_omega_peri: Optional[Quantity] = None,
        cos_omega_peri: Optional[Quantity] = None,
        asc_node: Optional[Quantity] = None,
        sin_asc_node: Optional[Quantity] = None,
        cos_asc_node: Optional[Quantity] = None,
        mass: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
    ):
        # Check the input arguments
        provided_input_arguments = [
            arg
            for arg in (
                time_transit,
                time_peri,
                period,
                semimajor,
                inclination,
                impact_param,
                eccentricity,
                omega_peri,
                sin_omega_peri,
                cos_omega_peri,
                asc_node,
                sin_asc_node,
                cos_asc_node,
                mass,
                radius,
            )
            if arg is not None
        ]
        # dtype = jnp.result_type(*provided_input_arguments, jnp.float32)
        if any(jnp.ndim(arg) != 0 for arg in provided_input_arguments):
            raise ValueError(
                "All input arguments to 'KeplerianBody' must be scalars; "
                "for multi-planet systems, use 'jax.vmap'"
            )

        # Save the input mass and radius
        self.radius = radius
        self.mass = mass

        # Work out the period and semimajor axis to be consistent
        total_mass = central.mass if mass is None else mass + central.mass
        mass_factor = ureg.gravitational_constant * total_mass
        if semimajor is None:
            if period is None:
                raise ValueError(
                    "Either `period` or `semimajor` must be specified when constructing "
                    "a KeplerianBody"
                )
            self.semimajor = jnpu.cbrt(mass_factor * period**2 / (4 * jnp.pi**2))
            self.period = period
        elif period is None:
            self.semimajor = semimajor
            self.period = 2 * jnp.pi * semimajor * jnpu.sqrt(semimajor / mass_factor)
        else:
            self.semimajor = semimajor
            self.period = period

        # Handle treatment and normalization of angles
        if omega_peri is not None:
            if sin_omega_peri is not None or cos_omega_peri is not None:
                raise ValueError(
                    "Cannot specify both omega_peri and sin_omega_peri or cos_omega_peri"
                )
            self.sin_omega_peri = jnpu.sin(omega_peri)
            self.cos_omega_peri = jnpu.cos(omega_peri)
        elif (sin_omega_peri is None) != (cos_omega_peri is None):
            raise ValueError("Must specify both sin_omega_peri and cos_omega_peri")
        else:
            self.sin_omega_peri = sin_omega_peri
            self.cos_omega_peri = cos_omega_peri

        if asc_node is not None:
            if sin_asc_node is not None or cos_asc_node is not None:
                raise ValueError(
                    "Cannot specify both asc_node and sin_asc_node or cos_asc_node"
                )
            self.sin_asc_node = jnpu.sin(asc_node)
            self.cos_asc_node = jnpu.cos(asc_node)
        elif (sin_asc_node is None) != (cos_asc_node is None):
            raise ValueError("Must specify both sin_asc_node and cos_asc_node")
        else:
            self.sin_asc_node = sin_asc_node
            self.cos_asc_node = cos_asc_node

        # Handle eccentric and circular orbits
        self.eccentricity = eccentricity
        if eccentricity is None:
            if sin_omega_peri is not None:
                raise ValueError("Cannot specify omega_peri without eccentricity")

            M0 = jnpu.full_like(self.period, 0.5 * jnp.pi)
            incl_factor = 1
        else:
            if self.sin_omega_peri is None:
                raise ValueError("Must specify omega_peri for eccentric orbits")

            opsw = 1 + self.sin_omega_peri
            E0 = 2 * jnpu.arctan2(
                jnpu.sqrt(1 - eccentricity) * self.cos_omega_peri,
                jnpu.sqrt(1 + eccentricity) * opsw,
            )
            M0 = E0 - eccentricity * jnpu.sin(E0)

            ome2 = 1 - eccentricity**2
            incl_factor = (1 + eccentricity * self.sin_omega_peri) / ome2

        # Handle inclined orbits
        dcosidb = incl_factor * central.radius / self.semimajor
        if impact_param is not None:
            if inclination is not None:
                raise ValueError("Cannot specify both inclination and impact_param")
            self.impact_param = impact_param
            self.cos_inclination = dcosidb * impact_param
            self.sin_inclination = jnpu.sqrt(1 - self.cos_inclination**2)
        elif inclination is not None:
            self.cos_inclination = jnpu.cos(inclination)
            self.sin_inclination = jnpu.sin(inclination)
            self.impact_param = self.cos_inclination / dcosidb
        else:
            self.impact_param = jnpu.zeros_like(self.period)
            self.cos_inclination = jnpu.zeros_like(self.period)
            self.sin_inclination = jnpu.ones_like(self.period)

        # Work out all the relevant reference times
        self.time_ref = -M0 * self.period / (2 * jnp.pi)
        if time_transit is not None and time_peri is not None:
            raise ValueError("Cannot specify both time_transit or time_peri")
        elif time_transit is not None:
            self.time_transit = time_transit
        elif time_peri is not None:
            self.time_transit = time_peri - self.time_ref
        else:
            self.time_transit = jnpu.zeros_like(self.period)

    @property
    def time_peri(self) -> Quantity:
        return self.time_transit + self.time_ref  # type: ignore

    @property
    def inclination(self) -> Quantity:
        return jnpu.arctan2(self.sin_inclination, self.cos_inclination)

    @property
    def omega_peri(self) -> Optional[Quantity]:
        if self.eccentricity is None:
            return None
        return jnpu.arctan2(self.sin_omega_peri, self.cos_omega_peri)


class KeplerianSystem(eqx.Module):
    central: Central
    bodies: tuple[Body, ...]

    def __init__(self, central: Central, *, bodies: tuple[Body, ...] = ()):
        self.central = central
        self.bodies = bodies

    def add_body(self, body: Optional[Body] = None, **kwargs: Any) -> "KeplerianSystem":
        if body is None:
            body = Body(self.central, **kwargs)
        return KeplerianSystem(central=self.central, bodies=self.bodies + (body,))


class KeplerianCentral(eqx.Module):
    mass: Quantity = units.field(units=ureg.M_sun)
    radius: Quantity = units.field(units=ureg.R_sun)
    density: Quantity = units.field(units=ureg.M_sun / ureg.R_sun**3)

    @units.quantity_input(
        mass=ureg.M_sun, radius=ureg.R_sun, density=ureg.M_sun / ureg.R_sun**3
    )
    def __init__(
        self,
        *,
        mass: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
        density: Optional[Quantity] = None,
    ):
        if radius is None and mass is None:
            radius = 1.0 * ureg.R_sun
            if density is None:
                mass = 1.0 * ureg.M_sun

        # Check that all the input values are scalars; we don't support Scalars
        # here
        if any(
            jnp.ndim(arg) != 0 for arg in (mass, radius, density) if arg is not None
        ):
            raise ValueError("All parameters of a KeplerianCentral must be scalars")

        # Compute all three parameters based on the input values
        error_msg = (
            "Values must be provided for exactly two of mass, radius, and density"
        )
        if density is None:
            if mass is None or radius is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = radius
            self.density = 3 * mass / (4 * jnp.pi * radius**3)
        elif radius is None:
            if mass is None or density is None:
                raise ValueError(error_msg)
            self.mass = mass
            self.radius = (3 * mass / (4 * jnp.pi * density)) ** (1 / 3)
            self.density = density
        elif mass is None:
            if radius is None or density is None:
                raise ValueError(error_msg)
            self.mass = 4 * jnp.pi * radius**3 * density / 3.0
            self.radius = radius
            self.density = density

    @classmethod
    def from_orbital_properties(
        cls,
        *,
        period: Quantity,
        semimajor: Quantity,
        radius: Optional[Quantity] = None,
        body_mass: Optional[Quantity] = None,
    ) -> "KeplerianCentral":
        if jnp.ndim(semimajor) != 0:
            raise ValueError(
                "The 'semimajor' argument to "
                "'KeplerianCentral.from_orbital_properties' "
                "must be a scalar; for multi-planet systems, "
                "use 'jax.vmap'"
            )

        radius = 1.0 * ureg.R_sun if radius is None else radius

        mass = semimajor**3 / (ureg.gravitational_constant * period**2)
        if body_mass is not None:
            mass -= body_mass

        return cls(mass=mass, radius=radius)

    @property
    def shape(self) -> tuple[int, ...]:
        leaves, _ = jax.tree_util.tree_flatten(self)
        shape = jnp.shape(leaves[0])
        if any(jnp.shape(leaf) != shape for leaf in leaves[1:]):
            raise ValueError("Inconsistent shapes for parameters of 'KeplerianCentral'")
        return shape


class KeplerianBody(eqx.Module):
    central: KeplerianCentral
    time_ref: Quantity = units.field(units=ureg.d)
    time_transit: Quantity = units.field(units=ureg.d)
    period: Quantity = units.field(units=ureg.d)
    semimajor: Quantity = units.field(units=ureg.R_sun)
    sin_inclination: Quantity = units.field(units=ureg.dimensionless)
    cos_inclination: Quantity = units.field(units=ureg.dimensionless)
    impact_param: Quantity = units.field(units=ureg.dimensionless)
    mass: Optional[Quantity] = units.field(units=ureg.M_sun)
    radius: Optional[Quantity] = units.field(units=ureg.R_sun)
    eccentricity: Optional[Quantity] = units.field(units=ureg.dimensionless)
    sin_omega_peri: Optional[Quantity] = units.field(units=ureg.dimensionless)
    cos_omega_peri: Optional[Quantity] = units.field(units=ureg.dimensionless)
    sin_asc_node: Optional[Quantity] = units.field(units=ureg.dimensionless)
    cos_asc_node: Optional[Quantity] = units.field(units=ureg.dimensionless)

    @units.quantity_input(
        time_transit=ureg.d,
        time_peri=ureg.d,
        period=ureg.d,
        semimajor=ureg.R_sun,
        inclination=ureg.radian,
        impact_param=ureg.dimensionless,
        eccentricity=ureg.dimensionless,
        omega_peri=ureg.radian,
        sin_omega_peri=ureg.dimensionless,
        cos_omega_peri=ureg.dimensionless,
        asc_node=ureg.radian,
        sin_asc_node=ureg.dimensionless,
        cos_asc_node=ureg.dimensionless,
        mass=ureg.M_sun,
        radius=ureg.R_sun,
        central_radius=ureg.R_sun,
    )
    def __init__(
        self,
        *,
        time_transit: Optional[Quantity] = None,
        time_peri: Optional[Quantity] = None,
        period: Optional[Quantity] = None,
        semimajor: Optional[Quantity] = None,
        inclination: Optional[Quantity] = None,
        impact_param: Optional[Quantity] = None,
        eccentricity: Optional[Quantity] = None,
        omega_peri: Optional[Quantity] = None,
        sin_omega_peri: Optional[Quantity] = None,
        cos_omega_peri: Optional[Quantity] = None,
        asc_node: Optional[Quantity] = None,
        sin_asc_node: Optional[Quantity] = None,
        cos_asc_node: Optional[Quantity] = None,
        mass: Optional[Quantity] = None,
        radius: Optional[Quantity] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[Quantity] = None,
    ):
        self.central = self._central_from_orbital_properties(
            period=period,
            semimajor=semimajor,
            mass=mass,
            central=central,
            central_radius=central_radius,
        )
        if self.central.shape != ():
            raise ValueError(
                "The central body for a 'KeplerianBody' must have scalar "
                "parameters; for multi-planet systems, use 'jax.vmap'"
            )

        # Check the input arguments
        provided_input_arguments = [
            arg
            for arg in (
                time_transit,
                time_peri,
                period,
                semimajor,
                inclination,
                impact_param,
                eccentricity,
                omega_peri,
                sin_omega_peri,
                cos_omega_peri,
                asc_node,
                sin_asc_node,
                cos_asc_node,
                mass,
                radius,
                central_radius,
            )
            if arg is not None
        ]
        # dtype = jnp.result_type(*provided_input_arguments, jnp.float32)
        if any(jnp.ndim(arg) != 0 for arg in provided_input_arguments):
            raise ValueError(
                "All input arguments to 'KeplerianBody' must be scalars; "
                "for multi-planet systems, use 'jax.vmap'"
            )

        # Save the input mass and radius
        self.radius = radius
        self.mass = mass

        # Work out the period and semimajor axis to be consistent
        mass_factor = ureg.gravitational_constant * self.total_mass
        if semimajor is None:
            if period is None:
                raise ValueError(
                    "Either `period` or `semimajor` must be specified when constructing "
                    "a KeplerianBody"
                )
            self.semimajor = jnpu.cbrt(mass_factor * period**2 / (4 * jnp.pi**2))
            self.period = period
        elif period is None:
            self.semimajor = semimajor
            self.period = 2 * jnp.pi * semimajor * jnpu.sqrt(semimajor / mass_factor)
        else:
            self.semimajor = semimajor
            self.period = period

        # Handle treatment and normalization of angles
        if omega_peri is not None:
            if sin_omega_peri is not None or cos_omega_peri is not None:
                raise ValueError(
                    "Cannot specify both omega_peri and sin_omega_peri or cos_omega_peri"
                )
            self.sin_omega_peri = jnpu.sin(omega_peri)
            self.cos_omega_peri = jnpu.cos(omega_peri)
        elif (sin_omega_peri is None) != (cos_omega_peri is None):
            raise ValueError("Must specify both sin_omega_peri and cos_omega_peri")
        else:
            self.sin_omega_peri = sin_omega_peri
            self.cos_omega_peri = cos_omega_peri

        if asc_node is not None:
            if sin_asc_node is not None or cos_asc_node is not None:
                raise ValueError(
                    "Cannot specify both asc_node and sin_asc_node or cos_asc_node"
                )
            self.sin_asc_node = jnpu.sin(asc_node)
            self.cos_asc_node = jnpu.cos(asc_node)
        elif (sin_asc_node is None) != (cos_asc_node is None):
            raise ValueError("Must specify both sin_asc_node and cos_asc_node")
        else:
            self.sin_asc_node = sin_asc_node
            self.cos_asc_node = cos_asc_node

        # Handle eccentric and circular orbits
        self.eccentricity = eccentricity
        if eccentricity is None:
            if sin_omega_peri is not None:
                raise ValueError("Cannot specify omega_peri without eccentricity")

            M0 = jnpu.full_like(self.period, 0.5 * jnp.pi)
            incl_factor = 1
        else:
            if self.sin_omega_peri is None:
                raise ValueError("Must specify omega_peri for eccentric orbits")

            opsw = 1 + self.sin_omega_peri
            E0 = 2 * jnpu.arctan2(
                jnpu.sqrt(1 - eccentricity) * self.cos_omega_peri,
                jnpu.sqrt(1 + eccentricity) * opsw,
            )
            M0 = E0 - eccentricity * jnpu.sin(E0)

            ome2 = 1 - eccentricity**2
            incl_factor = (1 + eccentricity * self.sin_omega_peri) / ome2

        # Handle inclined orbits
        dcosidb = incl_factor * self.central.radius / self.semimajor
        if impact_param is not None:
            if inclination is not None:
                raise ValueError("Cannot specify both inclination and impact_param")
            self.impact_param = impact_param
            self.cos_inclination = dcosidb * impact_param
            self.sin_inclination = jnpu.sqrt(1 - self.cos_inclination**2)
        elif inclination is not None:
            self.cos_inclination = jnpu.cos(inclination)
            self.sin_inclination = jnpu.sin(inclination)
            self.impact_param = self.cos_inclination / dcosidb
        else:
            self.impact_param = jnpu.zeros_like(self.period)
            self.cos_inclination = jnpu.zeros_like(self.period)
            self.sin_inclination = jnpu.ones_like(self.period)

        # Work out all the relevant reference times
        self.time_ref = -M0 * self.period / (2 * jnp.pi)
        if time_transit is not None and time_peri is not None:
            raise ValueError("Cannot specify both time_transit or time_peri")
        elif time_transit is not None:
            self.time_transit = time_transit
        elif time_peri is not None:
            self.time_transit = time_peri - self.time_ref
        else:
            self.time_transit = jnpu.zeros_like(self.period)

    @classmethod
    @units.quantity_input(
        period=ureg.d,
        duration=ureg.d,
        impact_param=ureg.dimensionless,
        radius=ureg.R_sun,
        mass=ureg.M_sun,
        central_radius=ureg.R_sun,
    )
    def circular_from_duration(
        cls,
        *,
        period: Quantity,
        duration: Quantity,
        impact_param: Quantity,
        radius: Quantity,
        mass: Optional[Quantity] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[Quantity] = None,
        **kwargs: Any,
    ) -> "KeplerianBody":
        central = cls._central_from_orbital_properties(
            period=period,
            mass=mass,
            central=central,
            central_radius=central_radius,
        )

        # Get the semimajor axis implied by the duration, period, and impact
        # parameter
        b2 = jnpu.square(impact_param)
        opk2 = jnpu.square(1 + radius / central.radius)
        phi = jnp.pi * duration / period
        sinp = jnpu.sin(phi)
        cosp = jnpu.cos(phi)
        semimajor = jnpu.sqrt(opk2 - b2 * cosp**2) / sinp

        central = KeplerianCentral.from_orbital_properties(
            period=period,
            semimajor=semimajor,
            radius=central_radius,
            body_mass=mass,
        )

        return cls(
            period=period,
            impact_param=impact_param,
            radius=radius,
            mass=mass,
            central=central,
            **kwargs,
        )

    @classmethod
    def _central_from_orbital_properties(
        cls,
        *,
        period: Optional[Quantity] = None,
        semimajor: Optional[Quantity] = None,
        mass: Optional[Quantity] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[Quantity] = None,
    ) -> KeplerianCentral:
        if period is None and semimajor is None:
            raise ValueError("Either a period or a semimajor axis must be provided")

        # When both period and semimajor axis are provided, we set up the
        # central with the density implied by these parameters
        if period is not None and semimajor is not None:
            if central is not None:
                raise ValueError(
                    "Cannot specify both period and semimajor axis when also "
                    "providing a central body"
                )
            central = KeplerianCentral.from_orbital_properties(
                period=period,
                semimajor=semimajor,
                radius=central_radius,
                body_mass=mass,
            )

        return KeplerianCentral(radius=central_radius) if central is None else central

    @property
    def shape(self) -> tuple[int, ...]:
        leaves, _ = jax.tree_util.tree_flatten(self)
        shape = jnp.shape(leaves[0])
        if any(jnp.shape(leaf) != shape for leaf in leaves[1:]):
            raise ValueError("Inconsistent shapes for parameters of 'KeplerianBody'")
        return shape

    @property
    def time_peri(self) -> Quantity:
        return self.time_transit + self.time_ref  # type: ignore

    @property
    def inclination(self) -> Quantity:
        return jnpu.arctan2(self.sin_inclination, self.cos_inclination)

    @property
    def omega_peri(self) -> Optional[Quantity]:
        if self.eccentricity is None:
            return None
        return jnpu.arctan2(self.sin_omega_peri, self.cos_omega_peri)

    @property
    def total_mass(self) -> Quantity:
        if self.mass is None:
            return self.central.mass
        else:
            return self.central.mass + self.mass  # type: ignore

    @property
    def central_radius(self) -> Quantity:
        return self.central.radius

    @property
    def _baseline_rv_semiamplitude(self) -> Quantity:
        k0 = 2 * jnp.pi * self.semimajor / (self.total_mass * self.period)
        if self.eccentricity is None:
            return k0
        return k0 / jnpu.sqrt(1 - self.eccentricity**2)

    def position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's position in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        semimajor = -self.semimajor * self.central.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def central_position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """The central's position in the barycentric frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        semimajor = self.semimajor * self.mass / self.total_mass
        return self._get_position_and_velocity(
            t, semimajor=semimajor, parallax=parallax
        )[0]

    def relative_position(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's position relative to the central in the X,Y,Z frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        return self._get_position_and_velocity(
            t, semimajor=-self.semimajor, parallax=parallax  # type: ignore
        )[0]

    def relative_angles(
        self, t: Quantity, parallax: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity]:
        """This body's relative position to the central in the sky plane, in
        separation, position angle coordinates

        Args:
            t: The times where the angles should be evaluated.

        Returns:
            The separation (arcseconds) and position angle (radians, measured
            east of north) of the planet relative to the star.
        """
        X, Y, _ = self.relative_position(t, parallax=parallax)[0]
        rho = jnpu.sqrt(X**2 + Y**2)
        theta = jnpu.arctan2(Y, X)
        return rho, theta

    def velocity(
        self, t: Quantity, semiamplitude: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's velocity in the barycentric frame

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.
        """
        if semiamplitude is None:
            mass: Quantity = -self.central.mass  # type: ignore
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.central.mass / self.mass
        return self._get_position_and_velocity(t, semiamplitude=k)[1]

    def central_velocity(
        self, t: Quantity, semiamplitude: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """The central's velocity in the barycentric frame

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.
        """
        if semiamplitude is None:
            return self._get_position_and_velocity(t, mass=self.mass)[1]
        _, v = self._get_position_and_velocity(t, semiamplitude=semiamplitude)
        return v

    def relative_velocity(
        self, t: Quantity, semiamplitude: Optional[Quantity] = None
    ) -> tuple[Quantity, Quantity, Quantity]:
        """This body's velocity relative to the central

        Args:
            t: The times where the velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. By convention, this
                is half of the peak-to-peak amplitude of the central's velocity.
                If provided, the ``mass`` and ``inclination`` parameters will be
                ignored and this amplitude will be used instead.

        Returns:
            The components of the velocity vector at ``t`` in units of
            ``M_sun/day``.
        """
        if semiamplitude is None:
            mass: Quantity = -self.total_mass  # type: ignore
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.total_mass / self.mass
        _, v = self._get_position_and_velocity(t, semiamplitude=k)
        return v

    def radial_velocity(
        self, t: Quantity, semiamplitude: Optional[Quantity] = None
    ) -> Quantity:
        """Get the radial velocity of the central

        .. note:: The convention is that positive `z` points *towards* the
            observer. However, for consistency with radial velocity literature
            this method returns values where positive radial velocity
            corresponds to a redshift as expected.

        Args:
            t: The times where the radial velocity should be evaluated.
            semiamplitude: The semi-amplitude of the orbit. If provided, the
                ``mass`` and ``inclination`` parameters will be ignored and
                this amplitude will be used instead.

        Returns:
            The reflex radial velocity evaluated at ``t``.
        """
        return -self.central_velocity(t, semiamplitude=semiamplitude)[2]  # type: ignore

    def _warp_times(self, t: Quantity) -> Quantity:
        return t - self.time_transit  # type: ignore

    @units.quantity_input(t=ureg.d)
    def _get_true_anomaly(self, t: Quantity) -> tuple[Quantity, Quantity]:
        M = 2 * jnp.pi * (self._warp_times(t) - self.time_ref) / self.period
        if self.eccentricity is None:
            return jnpu.sin(M), jnpu.cos(M)
        return kepler(M.magnitude, self.eccentricity.magnitude)

    def _rotate_vector(
        self, x: Quantity, y: Quantity, *, include_inclination: bool = True
    ) -> tuple[Quantity, Quantity, Quantity]:
        """Apply the rotation matrices to go from orbit to observer frame

        In order,
        1. rotate about the z axis by an amount omega -> x1, y1, z1
        2. rotate about the x1 axis by an amount -incl -> x2, y2, z2
        3. rotate about the z2 axis by an amount Omega -> x3, y3, z3

        Args:
            x: A tensor representing the x coodinate in the plane of the
                orbit.
            y: A tensor representing the y coodinate in the plane of the
                orbit.

        Returns:
            Three tensors representing ``(X, Y, Z)`` in the observer frame.
        """

        # 1) rotate about z0 axis by omega
        if self.eccentricity is None:
            x1 = x
            y1 = y
        else:
            x1 = self.cos_omega_peri * x - self.sin_omega_peri * y
            y1 = self.sin_omega_peri * x + self.cos_omega_peri * y

        # 2) rotate about x1 axis by -incl
        # z3 = z2, subsequent rotation by Omega doesn't affect it
        if include_inclination:
            x2 = x1
            y2 = self.cos_inclination * y1
            Z = -self.sin_inclination * y1
        else:
            x2 = x1
            y2 = y1
            Z = -y1

        # 3) rotate about z2 axis by Omega
        if self.cos_asc_node is None:
            return x2, y2, Z  # type: ignore

        X = self.cos_asc_node * x2 - self.sin_asc_node * y2
        Y = self.sin_asc_node * x2 + self.cos_asc_node * y2
        return X, Y, Z  # type: ignore

    @units.quantity_input(
        t=ureg.d,
        semimajor=ureg.R_sun,
        mass=ureg.M_sun,
        parallax=ureg.arcsec,
    )
    def _get_position_and_velocity(
        self,
        t: Quantity,
        semimajor: Optional[Quantity] = None,
        mass: Optional[Quantity] = None,
        semiamplitude: Optional[Quantity] = None,
        parallax: Optional[Quantity] = None,
    ) -> tuple[
        tuple[Quantity, Quantity, Quantity], tuple[Quantity, Quantity, Quantity]
    ]:
        sinf, cosf = self._get_true_anomaly(t)

        if semiamplitude is None:
            factor = 1 if mass is None else mass
            k0 = factor * self._baseline_rv_semiamplitude
            if parallax is not None:
                k0 = k0.to(ureg.au / ureg.d).magnitude * parallax / ureg.day
        else:
            k0 = semiamplitude

        r0 = 1
        if semimajor is not None:
            if parallax is None:
                r0 = semimajor
            else:
                r0 = semimajor.to(ureg.au).magnitude * parallax

        if self.eccentricity is None:
            v1, v2 = -k0 * sinf, k0 * cosf
        else:
            v1, v2 = -k0 * sinf, k0 * (cosf + self.eccentricity)
            r0 *= (1 - self.eccentricity**2) / (1 + self.eccentricity * cosf)

        x, y, z = self._rotate_vector(r0 * cosf, r0 * sinf)
        vx, vy, vz = self._rotate_vector(
            v1, v2, include_inclination=semiamplitude is None
        )

        return (x, y, z), (vx, vy, vz)


# class KeplerianOrbit(NamedTuple):
#     bodies: KeplerianBody

#     def __len__(self) -> int:
#         return len(self.bodies.period)

#     @property
#     def shape(self) -> tuple[int, ...]:
#         return self.bodies.shape

#     @property
#     def radius(self) -> Scalar:
#         return self.bodies.radius

#     @property
#     def central_radius(self) -> Scalar:
#         return self.bodies.central.radius

#     @classmethod
#     def init(
#         cls, central: Optional[KeplerianCentral] = None, **body_args: Any
#     ) -> "KeplerianOrbit":
#         names, values = zip(*body_args.items())
#         # values = jnpu.broadcast_arrays(*(jnpu.atleast_1d(x) for x in values))'
#         bodies = jax.vmap(partial(KeplerianBody, central=central))(**body_args)
#         return cls(bodies=bodies)

#     def position(
#         self, t: Scalar, parallax: Optional[Scalar] = None
#     ) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(partial(KeplerianBody.position, t=t, parallax=parallax))(
#             self.bodies
#         )

#     def central_position(
#         self, t: Scalar, parallax: Optional[Scalar] = None
#     ) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(
#             partial(KeplerianBody.central_position, t=t, parallax=parallax)
#         )(self.bodies)

#     def relative_position(
#         self, t: Scalar, parallax: Optional[Scalar] = None
#     ) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(
#             partial(KeplerianBody.relative_position, t=t, parallax=parallax)
#         )(self.bodies)

#     def velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(partial(KeplerianBody.velocity, t=t))(self.bodies)

#     def central_velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(partial(KeplerianBody.central_velocity, t=t))(self.bodies)

#     def relative_velocity(self, t: Scalar) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(partial(KeplerianBody.relative_velocity, t=t))(self.bodies)

#     def radial_velocity(
#         self, t: Scalar, semiamplitude: Optional[Scalar] = None
#     ) -> tuple[Scalar, Scalar, Scalar]:
#         return jax.vmap(partial(KeplerianBody.radial_velocity, t=t))(
#             self.bodies, semiamplitude=semiamplitude
#         )
