__all__ = ["calc_poly_coeffs"]


import jax.numpy as jnp
from jaxtyping import Array


def calc_poly_coeffs(
    mu: Array,
    intensity_profile: Array,
    poly_degree: int = 10,
) -> Array:
    """Calculate the polynomial coefficients for a limb-darkening intensity profile.

    As `jaxoplanet` uses an arbitrary polynomial limb-darkening law
    (see `Agol et al. (2020) <https://arxiv.org/pdf/1908.03222>`), this function may
    be useful for users wanting to use non-polynomial limb-darkening laws
    within the `jaxoplanet` framework.
    Given a limb-darkening profile, this function will fit the profile with
    a polynomial of a user-specified order/degree and return the least-squares
    polynomial coefficients.
    These coefficients can then be passed as the `u` parameter when using the
    `light_curve` objects.

    The intensity profile should be relatively densely evaluated as otherwise
    the polynomial approximation may be non-monotonic.

    Args:
        mu (Array): An array of values between 0 and 1 indicating the locations where
            the limb-darkening intensity profile was evaluated. `mu` is the cosine
            of the angle between the line of sight and the emergent intensity and
            is `1` at the center of the star and `0` at the limb.
        intensity_profile (Array): A stellar limb-darkening intensity profile.
            We assume the profile is normalized so that it's value is `1` at the center
            of the star.
        poly_degree (int): The degree/order of the polynomial. Setting this to `1` would
            fit with a straight line, `2` would fit with a quadratic, and so on.
            We set the default to a relatively high value of 10.

    Returns:
        An array containing the polynomial coefficients that can be passed to
        the `light_curve` objects. The size of this array will be equal to the
        `poly_degree` value, and the ordering of the coefficients corresponds to the
        lowest order to the highest order of the polynomial.
    """

    assert len(mu) == len(intensity_profile)

    sort = jnp.argsort(mu)
    mu = mu[sort]
    intensity_profile = intensity_profile[sort]

    # The first column of ones is removed
    X = jnp.vander(
        1 - mu,
        N=poly_degree + 1,
        increasing=True,
    )[:, 1:]

    coeffs, _, _, _ = jnp.linalg.lstsq(
        X,
        intensity_profile - 1.0,
    )

    # We multiply by -1 to make the coeffs in the expected form
    return -1 * coeffs
