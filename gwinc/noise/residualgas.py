'''Functions to calculate residual gas noise

'''
from __future__ import division
from numpy import sqrt, log, pi

from .. import const
from .. import Struct


def residual_gas_scattering_arm(f, ifo, cavity, species):
    """Residual gas noise strain spectrum due to scattering from one arm

    Noise caused by the passage of residual gas molecules through the
    laser beams in one arm cavity due to scattering.

    :f: frequency array in Hz
    :ifo: gwinc IFO structure
    :cavity: arm cavity structure
    :species: molecular species structure

    :returns: arm strain noise power spectrum at :f:

    The method used here is presented by Rainer Weiss, Micheal
    E. Zucker, and Stanley E. Whitcomb in their paper Optical
    Pathlength Noise in Sensitive Interferometers Due to Residual Gas.

    Added to Bench by Zhigang Pan, Summer 2006
    Cleaned up by PF, Apr 07
    Eliminated numerical integration and substituted first order
    expansion of exp, to speed it up.

    """
    L = ifo.Infrastructure.Length
    kT = ifo.Infrastructure.Temp * const.kB
    P = species.BeamtubePressure
    M = species.mass
    alpha = species.polarizability

    rho = P / (kT)                   # number density of Gas
    v0 = sqrt(2*kT / M)              # mean speed of Gas

    waist = cavity.w0                # Gaussian beam waist size
    zr = cavity.zr                   # Rayleigh range
    z1 = -cavity.zBeam_ITM           # location of ITM relative to the waist
    z2 = cavity.zBeam_ETM            # location of ETM relative to the waist

    # The exponential of Eq. 1 of P940008 is expanded to first order; this
    # can be integrated analytically
    zint = log(z2 + sqrt(z2**2 + zr**2)) - log(z1 + sqrt(z1**2 + zr**2))
    zint = zint * zr/waist
    zint = zint - 2*pi*L*f/v0
    # optical path length for one arm
    zint = zint*((4*rho*(2*pi*alpha)**2)/v0)
    # eliminate any negative values due to first order approx.
    zint[zint < 0] = 0

    return zint


def residual_gas_damping_test_mass(f, ifo, species, sustf, squeezed_film):
    """Noise due to residual gas damping for one test mass

    :f: frequency array in Hz
    :ifo: gwinc IFO structure
    :species: molecular species structure
    :sustf: suspension transfer function structure
    :squeezed_film: squeezed film damping structure

    :returns: displacement noise
    """
    sus = ifo.Suspension
    if 'Temp' in sus.Stage[0]:
        kT = sus.Stage[0].Temp * const.kB
    else:
        kT = sus.Temp * const.kB

    mass = species.mass
    radius = ifo.Materials.MassRadius
    thickness = ifo.Materials.MassThickness
    thermal_vel = sqrt(kT/mass)  # thermal velocity

    # pressure in the test mass chambers; possibly different from the pressure
    # in the arms due to outgassing near the test mass
    pressure = species.ChamberPressure

    # infinite volume viscous damping coefficient for a cylinder
    # table 1 of https://doi.org/10.1016/j.physleta.2010.06.041
    beta_inf = pi * radius**2 * pressure/thermal_vel
    beta_inf *= sqrt(8/pi) * (1 + thickness/(2*radius) + pi/4)

    force_noise = 4 * kT * beta_inf

    # add squeezed film damping if necessary as parametrized by
    # Eq (5) of http://dx.doi.org/10.1103/PhysRevD.84.063007
    if squeezed_film.keys():
        # the excess force noise and diffusion time are specified directly
        if 'ExcessDamping' in squeezed_film:
            # ExcessDamping is the ratio of the total gas damping noise at DC
            # to damping in the infinite volume limit (in amplitude)
            DeltaS0 = (squeezed_film.ExcessDamping**2 - 1) * force_noise
            if DeltaS0 < 0:
                raise ValueError('ExcessDamping must be > 1')

            try:
                diffusion_time = squeezed_film.DiffusionTime
            except AttributeError:
                msg = 'If excess residual gas damping is given a diffusion ' \
                    + 'time must be specified as well'
                raise ValueError(msg)

        # if a gap between the test mass and another object is specified
        # use the approximate model of section IIIA and B
        elif 'gap' in squeezed_film:
            gap = squeezed_film.gap

            # Eq (14)
            diffusion_time = sqrt(pi/2) * radius**2 / (gap * thermal_vel)
            diffusion_time /= log(1 + (radius/gap)**2)

            # Eq (11) factoring out the low pass cutoff as in (5)
            DeltaS0 = 4 * kT * pi*radius**2 * pressure * diffusion_time / gap

        else:
            raise ValueError('Must specify either excess damping or a gap')

        # Eq (5)
        force_noise += DeltaS0 / (1 + (2*pi*f*diffusion_time)**2)

    # convert force to displacement noise using the suspension susceptibility
    noise = force_noise * abs(sustf.tst_suscept)**2

    return noise
