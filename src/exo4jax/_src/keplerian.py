from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from exo4jax._src.core.kepler import kepler
from exo4jax._src.types import Array, Scalar

# FIXME: Switch to constants from astropy
GRAVITATIONAL_CONSTANT = 2942.2062175044193 / (4 * jnp.pi**2)
AU_PER_R_SUN = 0.00465046726096215


class KeplerianCentral(NamedTuple):
    mass: Scalar
    radius: Scalar
    density: Scalar

    @property
    def shape(self) -> Tuple[int, ...]:
        leaves, _ = jax.tree_util.tree_flatten(self)
        shape = leaves[0].shape
        if any(leaf.shape != shape for leaf in leaves[1:]):
            raise ValueError(
                "Inconsistent shapes for parameters of 'KeplerianCentral'"
            )
        return shape

    @classmethod
    def init(
        cls,
        *,
        mass: Optional[Scalar] = None,
        radius: Optional[Scalar] = None,
        density: Optional[Scalar] = None,
    ) -> "KeplerianCentral":
        if radius is None and mass is None:
            radius = 1.0
            if density is None:
                mass = 1.0

        if sum(arg is None for arg in (mass, radius, density)) != 1:
            raise ValueError(
                "Values must be provided for exactly two of mass, radius, and density"
            )

        # Check that all the input values are scalars; we don't support Scalars
        # here
        if any(
            jnp.ndim(arg) != 0
            for arg in (mass, radius, density)
            if arg is not None
        ):
            raise ValueError(
                "All parameters of a KeplerianCentral must be scalars"
            )

        # Compute all three parameters based on the input values
        if density is None:
            if TYPE_CHECKING:
                assert mass is not None
                assert radius is not None
            density = 3 * mass / (4 * jnp.pi * radius**3)
        elif radius is None:
            if TYPE_CHECKING:
                assert mass is not None
                assert density is not None
            radius = (3 * mass / (4 * jnp.pi * density)) ** (1 / 3)
        elif mass is None:
            if TYPE_CHECKING:
                assert density is not None
                assert radius is not None
            mass = 4 * jnp.pi * radius**3 * density / 3.0

        # Convert dtypes to be at least float32
        dtype = jnp.result_type(mass, radius, density, jnp.float32)
        mass = jnp.asarray(mass, dtype=dtype)
        radius = jnp.asarray(radius, dtype=dtype)
        density = jnp.asarray(density, dtype=dtype)

        return cls(mass=mass, radius=radius, density=density)

    @classmethod
    def from_orbital_properties(
        cls,
        *,
        period: Scalar,
        semimajor: Scalar,
        radius: Optional[Scalar] = None,
        body_mass: Optional[Scalar] = None,
    ) -> "KeplerianCentral":
        if jnp.ndim(semimajor) != 0:
            raise ValueError(
                "The 'semimajor' argument to "
                "'KeplerianCentral.from_orbital_properties' "
                "must be a scalar; for multi-planet systems, "
                "use 'jax.vmap'"
            )

        radius = 1.0 if radius is None else radius

        mass = semimajor**3 / (GRAVITATIONAL_CONSTANT * period**2)
        if body_mass is not None:
            mass -= body_mass

        return cls.init(mass=mass, radius=radius)


class KeplerianBody(NamedTuple):
    central: KeplerianCentral
    time_ref: Scalar
    time_transit: Scalar
    period: Scalar
    semimajor: Scalar
    sin_inclination: Scalar
    cos_inclination: Scalar
    impact_param: Scalar
    mass: Scalar
    radius: Scalar
    eccentricity: Optional[Scalar]
    sin_omega_peri: Optional[Scalar]
    cos_omega_peri: Optional[Scalar]
    sin_asc_node: Optional[Scalar]
    cos_asc_node: Optional[Scalar]

    @property
    def shape(self) -> Tuple[int, ...]:
        leaves, _ = jax.tree_util.tree_flatten(self)
        shape = leaves[0].shape
        if any(leaf.shape != shape for leaf in leaves[1:]):
            raise ValueError(
                "Inconsistent shapes for parameters of 'KeplerianBody'"
            )
        return shape

    @property
    def time_peri(self) -> Scalar:
        return self.time_transit + self.time_ref

    @property
    def total_mass(self) -> Scalar:
        return (
            self.central.mass
            if self.mass is None
            else self.central.mass + self.mass
        )

    @property
    def _baseline_rv_semiamplitude(self) -> Scalar:
        k0 = 2 * jnp.pi * self.semimajor / (self.total_mass * self.period)
        if self.eccentricity is None:
            return k0
        return k0 / jnp.sqrt(1 - self.eccentricity**2)

    @classmethod
    def _central_from_orbital_properties(
        cls,
        *,
        period: Optional[Scalar] = None,
        semimajor: Optional[Scalar] = None,
        mass: Optional[Scalar] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[Scalar] = None,
    ) -> KeplerianCentral:
        if period is None and semimajor is None:
            raise ValueError(
                "Either a period or a semimajor axis must be provided"
            )

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

        return (
            KeplerianCentral.init(radius=central_radius)
            if central is None
            else central
        )

    @classmethod
    def init(
        cls,
        *,
        time_transit: Optional[Scalar] = None,
        time_peri: Optional[Scalar] = None,
        period: Optional[Scalar] = None,
        semimajor: Optional[Scalar] = None,
        inclination: Optional[Scalar] = None,
        impact_param: Optional[Scalar] = None,
        eccentricity: Optional[Scalar] = None,
        omega_peri: Optional[Scalar] = None,
        sin_omega_peri: Optional[Scalar] = None,
        cos_omega_peri: Optional[Scalar] = None,
        asc_node: Optional[Scalar] = None,
        sin_asc_node: Optional[Scalar] = None,
        cos_asc_node: Optional[Scalar] = None,
        mass: Optional[Scalar] = None,
        radius: Optional[Scalar] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[Scalar] = None,
    ) -> "KeplerianBody":
        central = cls._central_from_orbital_properties(
            period=period,
            semimajor=semimajor,
            mass=mass,
            central=central,
            central_radius=central_radius,
        )
        if central.shape != ():
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
        dtype = jnp.result_type(*provided_input_arguments, jnp.float32)
        if any(jnp.ndim(arg) != 0 for arg in provided_input_arguments):
            raise ValueError(
                "All input arguments to 'KeplerianBody.init' must be scalars; "
                "for multi-planet systems, use 'jax.vmap'"
            )

        # Work out the period and semimajor axis to be consistent
        total_mass = central.mass + mass if mass is not None else central.mass
        mass_factor = GRAVITATIONAL_CONSTANT * total_mass
        if semimajor is None:
            if TYPE_CHECKING:
                assert period is not None
            semimajor = (mass_factor * period**2) ** (1.0 / 3)
        elif period is None:
            period = semimajor**1.5 / jnp.sqrt(mass_factor)

        # Handle treatment and normalization of angles
        if omega_peri is not None:
            if sin_omega_peri is not None or cos_omega_peri is not None:
                raise ValueError(
                    "Cannot specify both omega_peri and sin_omega_peri or cos_omega_peri"
                )
            sin_omega_peri = jnp.sin(omega_peri)
            cos_omega_peri = jnp.cos(omega_peri)
        elif (sin_omega_peri is None) != (cos_omega_peri is None):
            raise ValueError(
                "Must specify both sin_omega_peri and cos_omega_peri"
            )

        if asc_node is not None:
            if sin_asc_node is not None or cos_asc_node is not None:
                raise ValueError(
                    "Cannot specify both asc_node and sin_asc_node or cos_asc_node"
                )
            sin_asc_node = jnp.sin(asc_node)
            cos_asc_node = jnp.cos(asc_node)
        elif (sin_asc_node is None) != (cos_asc_node is None):
            raise ValueError("Must specify both sin_asc_node and cos_asc_node")

        # Handle eccentric and circular orbits
        if eccentricity is None:
            if sin_omega_peri is not None:
                raise ValueError(
                    "Cannot specify omega_peri without eccentricity"
                )

            M0 = jnp.full_like(period, 0.5 * jnp.pi)
            incl_factor = 1
        else:
            if sin_omega_peri is None:
                raise ValueError(
                    "Must specify omega_peri for eccentric orbits"
                )

            opsw = 1 + sin_omega_peri
            E0 = 2 * jnp.arctan2(
                jnp.sqrt(1 - eccentricity) * cos_omega_peri,
                jnp.sqrt(1 + eccentricity) * opsw,
            )
            M0 = E0 - eccentricity * jnp.sin(E0)

            ome2 = 1 - eccentricity**2
            incl_factor = (1 + eccentricity * sin_omega_peri) / ome2

        # Handle inclined orbits
        dcosidb = incl_factor * central.radius / semimajor
        if impact_param is not None:
            if inclination is not None:
                raise ValueError(
                    "Cannot specify both inclination and impact_param"
                )
            cos_inclination = dcosidb * impact_param
            sin_inclination = jnp.sqrt(1 - cos_inclination**2)
        elif inclination is not None:
            cos_inclination = jnp.cos(inclination)
            sin_inclination = jnp.sin(inclination)
            impact_param = cos_inclination / dcosidb
        else:
            impact_param = jnp.zeros_like(period)
            cos_inclination = jnp.zeros_like(period)
            sin_inclination = jnp.ones_like(period)

        # Work out all the relevant reference times
        time_ref = -M0 * period / (2 * jnp.pi)
        if time_transit is not None and time_peri is not None:
            raise ValueError("Cannot specify both time_transit or time_peri")
        elif time_transit is not None:
            time_peri = time_transit + time_ref
        elif time_peri is not None:
            time_transit = time_peri - time_ref
        else:
            time_transit = jnp.zeros_like(period)
            time_peri = time_ref

        return cls(
            central=central,
            time_ref=jnp.asarray(time_ref, dtype=dtype),
            time_transit=jnp.asarray(time_transit, dtype=dtype),
            period=jnp.asarray(period, dtype=dtype),
            semimajor=jnp.asarray(semimajor, dtype=dtype),
            sin_inclination=jnp.asarray(sin_inclination, dtype=dtype),
            cos_inclination=jnp.asarray(cos_inclination, dtype=dtype),
            impact_param=jnp.asarray(impact_param, dtype=dtype),
            mass=jnp.asarray(0, dtype=dtype)
            if mass is None
            else jnp.asarray(mass, dtype=dtype),
            radius=jnp.asarray(0, dtype=dtype)
            if radius is None
            else jnp.asarray(radius, dtype=dtype),
            eccentricity=None
            if eccentricity is None
            else jnp.asarray(eccentricity, dtype=dtype),
            sin_omega_peri=None
            if sin_omega_peri is None
            else jnp.asarray(sin_omega_peri, dtype=dtype),
            cos_omega_peri=None
            if cos_omega_peri is None
            else jnp.asarray(cos_omega_peri, dtype=dtype),
            sin_asc_node=None
            if sin_asc_node is None
            else jnp.asarray(sin_asc_node, dtype=dtype),
            cos_asc_node=None
            if cos_asc_node is None
            else jnp.asarray(cos_asc_node, dtype=dtype),
        )

    @classmethod
    def circular_from_duration(
        cls,
        *,
        period: Scalar,
        duration: Scalar,
        impact_param: Scalar,
        radius: Scalar,
        mass: Optional[Scalar] = None,
        central: Optional[KeplerianCentral] = None,
        central_radius: Optional[KeplerianCentral] = None,
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
        b2 = jnp.square(impact_param)
        opk2 = jnp.square(1 + radius / central.radius)
        phi = jnp.pi * duration / period
        sinp = jnp.sin(phi)
        cosp = jnp.cos(phi)
        semimajor = jnp.sqrt(opk2 - b2 * cosp**2) / sinp

        central = KeplerianCentral.from_orbital_properties(
            period=period,
            semimajor=semimajor,
            radius=central_radius,
            body_mass=mass,
        )

        return cls.init(
            period=period,
            impact_param=impact_param,
            radius=radius,
            mass=mass,
            central=central,
            **kwargs,
        )

    @property
    def central_radius(self) -> Scalar:
        return self.central.radius

    def position(
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
        """This body's position relative to the central in the X,Y,Z frame

        Args:
            t: The times where the position should be evaluated.

        Returns:
            The components of the position vector at ``t`` in units of
            ``R_sun``.
        """
        return self._get_position_and_velocity(
            t, semimajor=-self.semimajor, parallax=parallax
        )[0]

    def relative_angles(
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar]:
        """This body's relative position to the central in the sky plane, in
        separation, position angle coordinates

        Args:
            t: The times where the angles should be evaluated.

        Returns:
            The separation (arcseconds) and position angle (radians, measured
            east of north) of the planet relative to the star.
        """
        X, Y, _ = self.relative_position(t, parallax=parallax)[0]
        rho = jnp.sqrt(X**2 + Y**2)
        theta = jnp.arctan2(Y, X)
        return (rho, theta)

    def velocity(
        self, t: Scalar, semiamplitude: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
            mass = -self.central.mass
            return self._get_position_and_velocity(t, mass=mass)[1]
        k = -semiamplitude * self.central.mass / self.mass
        return self._get_position_and_velocity(t, semiamplitude=k)[1]

    def central_velocity(
        self, t: Scalar, semiamplitude: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
        self, t: Scalar, semiamplitude: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
            return self._get_position_and_velocity(t, mass=-self.total_mass)[1]
        k = -semiamplitude * self.total_mass / self.mass
        _, v = self._get_position_and_velocity(t, semiamplitude=k)
        return v

    def radial_velocity(
        self, t: Scalar, semiamplitude: Optional[Scalar] = None
    ) -> Scalar:
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
        return -self.central_velocity(t, semiamplitude=semiamplitude)[2]

    def _warp_times(self, t: Scalar) -> Scalar:
        return t - self.time_transit

    def _get_true_anomaly(self, t: Scalar) -> Tuple[Scalar, Scalar]:
        M = 2 * jnp.pi * (self._warp_times(t) - self.time_ref) / self.period
        if self.eccentricity is None:
            return jnp.sin(M), jnp.cos(M)
        return kepler(M, self.eccentricity)

    def _rotate_vector(
        self, x: Scalar, y: Scalar, *, include_inclination: bool = True
    ) -> Tuple[Scalar, Scalar, Scalar]:
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
            return (x2, y2, Z)

        X = self.cos_asc_node * x2 - self.sin_asc_node * y2
        Y = self.sin_asc_node * x2 + self.cos_asc_node * y2
        return X, Y, Z

    def _get_position_and_velocity(
        self,
        t: Scalar,
        semimajor: Optional[Scalar] = None,
        mass: Optional[Scalar] = None,
        semiamplitude: Optional[Scalar] = None,
        parallax: Optional[Scalar] = None,
    ) -> Tuple[Tuple[Scalar, Scalar, Scalar], Tuple[Scalar, Scalar, Scalar]]:
        sinf, cosf = self._get_true_anomaly(t)

        if semiamplitude is None:
            factor = 1 if mass is None else mass
            factor *= 1 if parallax is None else parallax * AU_PER_R_SUN
            k0 = factor * self._baseline_rv_semiamplitude
        else:
            k0 = semiamplitude

        r0 = 1
        if semimajor is not None:
            r0 *= (
                semimajor
                if parallax is None
                else semimajor * parallax * AU_PER_R_SUN
            )

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


class KeplerianOrbit(NamedTuple):
    bodies: KeplerianBody

    def __len__(self) -> int:
        return len(self.bodies.period)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.bodies.shape

    @property
    def radius(self) -> Scalar:
        return self.bodies.radius

    @property
    def central_radius(self) -> Scalar:
        return self.bodies.central.radius

    @classmethod
    def init(
        cls, central: Optional[KeplerianCentral] = None, **body_args: Scalar
    ) -> "KeplerianOrbit":
        names, values = zip(*body_args.items())
        values = jnp.broadcast_arrays(*(jnp.atleast_1d(x) for x in values))
        bodies = jax.vmap(partial(KeplerianBody.init, central=central))(
            **dict(zip(names, values))
        )
        return cls(bodies=bodies)

    def position(
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(
            partial(KeplerianBody.position, t=t, parallax=parallax)
        )(self.bodies)

    def central_position(
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(
            partial(KeplerianBody.central_position, t=t, parallax=parallax)
        )(self.bodies)

    def relative_position(
        self, t: Scalar, parallax: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(
            partial(KeplerianBody.relative_position, t=t, parallax=parallax)
        )(self.bodies)

    def velocity(self, t: Scalar) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(partial(KeplerianBody.velocity, t=t))(self.bodies)

    def central_velocity(self, t: Scalar) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(partial(KeplerianBody.central_velocity, t=t))(
            self.bodies
        )

    def relative_velocity(self, t: Scalar) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(partial(KeplerianBody.relative_velocity, t=t))(
            self.bodies
        )

    def radial_velocity(
        self, t: Scalar, semiamplitude: Optional[Scalar] = None
    ) -> Tuple[Scalar, Scalar, Scalar]:
        return jax.vmap(partial(KeplerianBody.radial_velocity, t=t))(
            self.bodies, semiamplitude=semiamplitude
        )
