'''Functions to calculate residual gas noise

'''
from __future__ import division
import numpy as np
from numpy import sqrt, log, pi
from scipy.integrate import trapezoid

from .. import const
from .. import Struct
from .. import nb
from ..ifo.noises import dhdl, arm_cavity
from ..suspension import precomp_suspension


RESGAS_STYLES = dict(
    H2 = dict(
        label='H$_2$',
        color='xkcd:red orange',
    ),

    N2 = dict(
        label='N$_2$',
        color='xkcd:emerald',
    ),

    H2O = dict(
        label='H$_2$O',
        color='xkcd:water blue',
    ),

    O2 = dict(
        label='O$_2$',
        color='xkcd:grey',
    ),
)


# FIXME HACK: it's unclear if a phase noise in the arms like
# the excess gas noise should get the same dhdL strain
# calibration as the other displacement noises.  However, we
# would like to use the one Strain calibration for all noises,
# so we need to divide by the sinc_sqr here to undo the
# application of the dhdl in the Strain calibration.  But this
# is ultimately a superfluous extra calculation with the only
# to provide some simplication in the Budget definition, so
# should be re-evaluated at some point.


def ResidualGasScattering_constructor(species_name):
    """Residual gas scattering for a single species

    """

    class GasScatteringSpecies(nb.Noise):
        name = 'Scattering' + species_name
        style = dict(
            label=RESGAS_STYLES[species_name]['label'] + ' scattering',
            color=RESGAS_STYLES[species_name]['color'],
            linestyle='-',
        )

        def calc(self):
            cavity = arm_cavity(self.ifo)
            Larm_m = self.ifo.Infrastructure.Length
            species = self.ifo.Infrastructure.ResidualGas[species_name]
            # position along the beamtube starting from vertex corner
            tube_pos = np.linspace(0, Larm_m, 100)
            # pressure profile is constant by default
            pressure_Pa = species.BeamtubePressure * np.ones_like(tube_pos)
            n = residual_gas_scattering_arm(
                self.freq, self.ifo, cavity, species, pressure_Pa, tube_pos)
            dhdl_sqr, sinc_sqr = dhdl(self.freq, Larm_m)
            return n * 2 / sinc_sqr

    return GasScatteringSpecies


def ResidualGasDamping_constructor(species_name):
    """Reisidual gas damping for a single species

    """

    class GasDampingSpecies(nb.Noise):
        name = 'Damping' + species_name
        style = dict(
            label=RESGAS_STYLES[species_name]['label'] + ' damping',
            color=RESGAS_STYLES[species_name]['color'],
            linestyle='--',
        )

        @nb.precomp(sustf=precomp_suspension)
        def calc(self, sustf):
            rg = self.ifo.Infrastructure.ResidualGas
            species = rg[species_name]
            squeezed_film = rg.get('SqueezedFilm', Struct())

            if squeezed_film is None:
                raise ValueError('Must specify either excess damping or a gap')

            # Calculate squeezed film for ETM and ITM seperately if either is given
            # explicitly. If only one is given, it is not computed for the other one.
            if ('ETM' in squeezed_film) or ('ITM' in squeezed_film):
                squeezed_film_ETM = squeezed_film.get('ETM', Struct())
                squeezed_film_ITM = squeezed_film.get('ITM', Struct())
                n_ETM = residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film_ETM)
                n_ITM = residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film_ITM)
                n = 2 * (n_ETM + n_ITM)

            # Otherwise the same calculation is used for both.
            else:
                n = 4 * residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film)

            return n

    return GasDampingSpecies


class ResidualGas(nb.Budget):
    """Residual Gas

    """
    style = dict(
        label='Residual Gas',
        color='#add00d',
        linestyle='-',
    )

    noises = [
        ResidualGasScattering_constructor('H2'),
        ResidualGasScattering_constructor('N2'),
        ResidualGasScattering_constructor('H2O'),
        ResidualGasScattering_constructor('O2'),
        ResidualGasDamping_constructor('H2'),
        ResidualGasDamping_constructor('N2'),
        ResidualGasDamping_constructor('H2O'),
        ResidualGasDamping_constructor('O2'),
    ]


def residual_gas_scattering_arm(
        f, ifo, cavity, species, pressure_Pa, tube_pos):
    """Residual gas scattering from one arm using measured beamtube pressures

    :f: frequency array in Hz
    :ifo: gwinc IFO structure
    :cavity: arm cavity structure
    :species: molecular species structure
    :pressure_Pa: beamtube pressure profile in Pa
    :tubepos_m: vector of positions where pressure is given in m

    :returns: arm strain noise power spectrum at :f:

    The method used here is presented by Rainer Weiss, Micheal
    E. Zucker, and Stanley E. Whitcomb in their paper Optical
    Pathlength Noise in Sensitive Interferometers Due to Residual Gas.

    """
    # beam profile
    ww_m = cavity.w0 * np.sqrt(1 + ((tube_pos - cavity.zBeam_ITM)/cavity.zr)**2)
    kT = ifo.Infrastructure.Temp * const.kB
    M = species.mass
    alpha = species.polarizability

    v0 = np.sqrt(2*kT / M)  # most probable speed of Gas

    alpha = species.polarizability

    # put the integrand into a (numfreq, numpressure) array for faster
    # integration with trapezoid
    integrand = np.exp(np.einsum('i,j->ij', -2*np.pi*f, ww_m/v0))
    integrand *= np.einsum('i,j->ij', np.ones_like(f), pressure_Pa / ww_m)
    zint = trapezoid(integrand, tube_pos, axis=1)

    noise = 4 * (2*np.pi*alpha)**2 / (v0 * kT) * zint

    return noise


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
