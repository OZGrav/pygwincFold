import copy
import numpy as np
from numpy import pi, sin, exp, sqrt

from .. import logger
from .. import const
from ..struct import Struct
from .. import nb
from .. import noise
from .. import suspension


############################################################
# helper functions
############################################################

def mirror_struct(ifo, tm):
    """Create a "mirror" Struct for a LIGO core optic

    This is a copy of the ifo.Materials Struct, containing Substrate
    and Coating sub-Structs, as well as some basic geometrical
    properties of the optic.

    """
    # NOTE: we deepcopy this struct since we'll be modifying it (and
    # it would otherwise be stored by reference)
    mirror = copy.deepcopy(ifo.Materials)
    optic = ifo.Optics.get(tm)
    if 'CoatLayerOpticalThickness' in optic:
        mirror.Coating.dOpt = optic['CoatLayerOpticalThickness']
    else:
        T = optic.Transmittance
        dL = optic.CoatingThicknessLown
        dCap = optic.CoatingThicknessCap
        mirror.Coating.dOpt = noise.coatingthermal.getCoatDopt(mirror, T, dL, dCap=dCap)
    mirror.update(optic)
    mirror.MassVolume = pi * mirror.MassRadius**2 * mirror.MassThickness
    mirror.MirrorMass = mirror.MassVolume * mirror.Substrate.MassDensity
    return mirror


def arm_cavity(ifo):
    L = ifo.Infrastructure.Length

    g1 = 1 - L / ifo.Optics.Curvature.ITM
    g2 = 1 - L / ifo.Optics.Curvature.ETM
    gcav = sqrt(g1 * g2 * (1 - g1 * g2))
    gden = g1 - 2 * g1 * g2 + g2

    if (g1 * g2 * (1 - g1 * g2)) <= 0:
        raise Exception('Unstable arm cavity g-factors.  Change ifo.Optics.Curvature')
    elif gcav < 1e-3:
        logger.warning('Nearly unstable arm cavity g-factors.  Reconsider ifo.Optics.Curvature')

    ws = sqrt(L * ifo.Laser.Wavelength / pi)
    w1 = ws * sqrt(abs(g2) / gcav)
    w2 = ws * sqrt(abs(g1) / gcav)

    # waist size
    w0 = ws * sqrt(gcav / abs(gden))
    zr = pi * w0**2 / ifo.Laser.Wavelength
    z1 = L * g2 * (1 - g1) / gden
    z2 = L * g1 * (1 - g2) / gden

    # waist, input, output
    cavity = Struct()
    cavity.w0 = w0
    cavity.wBeam_ITM = w1
    cavity.wBeam_ETM = w2
    cavity.zr = zr
    cavity.zBeam_ITM = z1
    cavity.zBeam_ETM = z2
    return cavity


def ifo_power(ifo, PRfixed=True):
    """Compute power on beamsplitter, finesse, and power recycling factor.

    """
    pin = ifo.Laser.Power
    t1 = sqrt(ifo.Optics.ITM.Transmittance)
    r1 = sqrt(1 - ifo.Optics.ITM.Transmittance)
    r2 = sqrt(1 - ifo.Optics.ETM.Transmittance)
    t5 = sqrt(ifo.Optics.PRM.Transmittance)
    r5 = sqrt(1 - ifo.Optics.PRM.Transmittance)
    loss = ifo.Optics.Loss  # single TM loss
    bsloss = ifo.Optics.BSLoss
    acoat = ifo.Optics.ITM.CoatingAbsorption
    pcrit = ifo.Optics.pcrit

    # Finesse, effective number of bounces in cavity, power recycling factor
    finesse = 2*pi / (t1**2 + 2*loss)  # arm cavity finesse
    neff = 2 * finesse / pi

    # Arm cavity reflectivity with finite loss
    garm = t1 / (1 - r1*r2*sqrt(1-2*loss))  # amplitude gain wrt input field
    rarm = r1 - t1 * r2 * sqrt(1-2*loss) * garm

    if PRfixed:
        Tpr = ifo.Optics.PRM.Transmittance  # use given value
    else:
        Tpr = 1-(rarm*sqrt(1-bsloss))**2  # optimal recycling mirror transmission
        t5 = sqrt(Tpr)
        r5 = sqrt(1 - Tpr)
    prfactor = t5**2 / (1 + r5 * rarm * sqrt(1-bsloss))**2

    pbs = pin * prfactor  # BS power from input power
    parm = pbs * garm**2 / 2  # arm power from BS power

    thickness = ifo.Optics.ITM.get('Thickness', ifo.Materials.MassThickness)
    asub = 1.3 * 2 * thickness * ifo.Optics.SubstrateAbsorption
    pbsl = 2 * pcrit / (asub+acoat*neff)  # bs power limited by thermal lensing

    if pbs > pbsl:
        logger.warning('P_BS exceeds BS Thermal limit!')

    power = Struct()
    power.pbs = pbs
    power.parm = parm
    power.finesse = finesse
    power.gPhase = finesse * 2/np.pi
    power.prfactor = prfactor
    power.Tpr = Tpr
    return power


############################################################
# calibration
############################################################

def dhdl(f, armlen):
    """Strain to length conversion for noise power spetra

    This takes into account the GW wavelength and is only important
    when this is comparable to the detector arm length.

    From R. Schilling, CQG 14 (1997) 1513-1519, equation 5,
    with n = 1, nu = 0.05, ignoring overall phase and cos(nu)^2.
    A small value of nu is used instead of zero to avoid infinities.

    Returns the square of the dh/dL function, and the same divided by
    the arm length squared.

    """
    c = const.c
    nu_small = 15*pi/180
    omega_arm = pi * f * armlen / c
    omega_arm_f = (1 - sin(nu_small)) * pi * f * armlen / c
    omega_arm_b = (1 + sin(nu_small)) * pi * f * armlen / c
    sinc_sqr = 4 / abs(sin(omega_arm_f) * exp(-1j * omega_arm) / omega_arm_f
                       + sin(omega_arm_b) * exp(1j * omega_arm) / omega_arm_b)**2
    dhdl_sqr = sinc_sqr / armlen**2
    return dhdl_sqr, sinc_sqr


class Strain(nb.Calibration):
    """Calibrate displacement to strain
    """
    def calc(self):
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return dhdl_sqr


class Force(nb.Calibration):
    """Calibrate displacement to force
    """
    def calc(self):
        mass = mirror_struct(self.ifo, 'ETM').MirrorMass
        return (mass * (2*pi*self.freq)**2)**2


############################################################
# noise sources
############################################################

#########################
# quantum
#########################

class QuantumVacuum(nb.Noise):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        total = np.zeros_like(quantum.ASvac)
        for nn in quantum.values():
            total += nn
        return total


class QuantumVacuumAS(nb.Noise):
    """Quantum vacuum from the AS port

    """
    style = dict(
        label='AS Port Vacuum',
        color='xkcd:emerald green'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.ASvac


class QuantumVacuumArm(nb.Noise):
    """Quantum vacuum due to arm cavity loss

    """
    style = dict(
        label='Arm Loss',
        color='xkcd:orange brown'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.Arm


class QuantumVacuumSEC(nb.Noise):
    """Quantum vacuum due to SEC loss

    """
    style = dict(
        label='SEC Loss',
        color='xkcd:cerulean'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.SEC


class QuantumVacuumFilterCavity(nb.Noise):
    """Quantum vacuum due to filter cavity loss

    """
    style = dict(
        label='Filter Cavity Loss',
        color='xkcd:goldenrod'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.FC


class QuantumVacuumInjection(nb.Noise):
    """Quantum vacuum due to injection loss

    """
    style = dict(
        label='Injection Loss',
        color='xkcd:fuchsia'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.Injection


class QuantumVacuumReadout(nb.Noise):
    """Quantum vacuum due to readout loss

    """
    style = dict(
        label='Readout Loss',
        color='xkcd:mahogany'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.PD


class QuantumVacuumQuadraturePhase(nb.Noise):
    """Quantum vacuum noise due to quadrature phase noise
    """
    style = dict(
        label='Quadrature Phase',
        color='xkcd:slate'
    )

    @nb.precomp(quantum=noise.quantum.precomp_quantum)
    def calc(self, quantum):
        return quantum.Phase


class StandardQuantumLimit(nb.Noise):
    """Standard Quantum Limit

    """
    style = dict(
        label="Standard Quantum Limit",
        color="#000000",
        linestyle=":",
    )

    def calc(self):
        ETM = mirror_struct(self.ifo, 'ETM')
        return 8 * const.hbar / (ETM.MirrorMass * (2 * np.pi * self.freq) ** 2)


#########################
# seismic
#########################

def Seismic_constructor(direction):
    """Seismic noise for a single direction

    """
    if direction == 'horiz':
        label = 'Horizontal'
        color = 'xkcd:muted blue'
    elif direction == 'vert':
        label = 'Vertical'
        color = 'xkcd:brick red'

    class SeismicDirection(nb.Noise):
        name = label
        style = dict(
            label=label,
            color=color,
        )

        @nb.precomp(sustf=suspension.precomp_suspension)
        def calc(self, sustf):
            nt, nr = noise.seismic.platform_motion(self.freq, self.ifo)
            n = noise.seismic.seismic_suspension_filtered(sustf, nt, direction)
            return n * 4

    return SeismicDirection


class Seismic(nb.Budget):
    """Seismic

    """
    style = dict(
        label='Seismic',
        color='#855700',
    )

    noises = [
        Seismic_constructor('vert'),
        Seismic_constructor('horiz'),
    ]


#########################
# Newtonian
#########################

class Newtonian(nb.Noise):
    """Newtonian Gravity

    """
    style = dict(
        label='Newtonian Gravity',
        color='#15b01a',
    )

    def calc(self):
        n = noise.newtonian.gravg(self.freq, self.ifo.Seismic)
        return n * 4


class NewtonianRayleigh(nb.Noise):
    """Newtonian Gravity, Rayleigh waves

    """
    style = dict(
        label='Newtonian (Rayleigh waves)',
        color='#1b2431',
    )

    def calc(self):
        n = noise.newtonian.gravg_rayleigh(self.freq, self.ifo.Seismic)
        return n * 2


class NewtonianBody(nb.Noise):
    """Newtonian Gravity, body waves

    """
    style = dict(
        label='Newtonian (body waves)',
        color='#85a3b2',
    )

    def calc(self):
        np = noise.newtonian.gravg_pwave(self.freq, self.ifo.Seismic)
        ns = noise.newtonian.gravg_swave(self.freq, self.ifo.Seismic)
        return (np + ns) * 4


class NewtonianInfrasound(nb.Noise):
    """Newtonian Gravity, infrasound

    """
    style = dict(
        label='Newtonian (infrasound)',
        color='#ffa62b',
    )

    def calc(self):
        n = noise.newtonian.atmois(self.freq, self.ifo.Atmospheric, self.ifo.Seismic)
        return n * 2


#########################
# suspension thermal
#########################

def SuspensionThermal_constructor(stage_num, direction):
    """Suspension thermal for a single stage in one direction

    """
    if stage_num == 0:
        stage_name = 'Top'
        color = 'xkcd:orangeish'
    elif stage_num == 1:
        stage_name = 'UIM'
        color = 'xkcd:mustard'
    elif stage_num == 2:
        stage_name = 'PUM'
        color = 'xkcd:turquoise'
    elif stage_num == 3:
        stage_name = 'Test mass'
        color = 'xkcd:bright purple'

    if direction == 'horiz':
        name0 = 'Horiz' + stage_name
        label = 'Horiz. ' + stage_name
        linestyle = '-'
    elif direction == 'vert':
        name0 = 'Vert' + stage_name
        label = 'Vert. ' + stage_name
        linestyle = '--'

    class SuspensionThermalStage(nb.Noise):
        name = name0
        style = dict(
            label=label,
            color=color,
            linestyle=linestyle,
            alpha=0.7,
        )

        @nb.precomp(sustf=suspension.precomp_suspension)
        def calc(self, sustf):
            n = noise.suspensionthermal.susptherm_stage(
                self.freq, self.ifo.Suspension, sustf, stage_num, direction)
            return abs(n) * 4

    return SuspensionThermalStage


class SuspensionThermal(nb.Budget):
    """Suspension Thermal

    """

    name = 'SuspensionThermal'

    style = dict(
        label='Suspension Thermal',
        color='#0d75f8',
    )

    noises = [
        SuspensionThermal_constructor(0, 'horiz'),
        SuspensionThermal_constructor(1, 'horiz'),
        SuspensionThermal_constructor(2, 'horiz'),
        SuspensionThermal_constructor(3, 'horiz'),
        SuspensionThermal_constructor(0, 'vert'),
        SuspensionThermal_constructor(1, 'vert'),
        SuspensionThermal_constructor(2, 'vert'),
    ]


#########################
# coating thermal
#########################

class CoatingBrownian(nb.Noise):
    """Coating Brownian

    """
    style = dict(
        label='Coating Brownian',
        color='#fe0002',
    )

    def calc(self):
        ITM = mirror_struct(self.ifo, 'ITM')
        ETM = mirror_struct(self.ifo, 'ETM')
        cavity = arm_cavity(self.ifo)
        wavelength = self.ifo.Laser.Wavelength
        nITM = noise.coatingthermal.coating_brownian(
            self.freq, ITM, wavelength, cavity.wBeam_ITM
        )
        nETM = noise.coatingthermal.coating_brownian(
            self.freq, ETM, wavelength, cavity.wBeam_ETM
        )
        return (nITM + nETM) * 2


class CoatingThermoOptic(nb.Noise):
    """Coating Thermo-Optic

    """
    style = dict(
        label='Coating Thermo-Optic',
        color='#02ccfe',
        linestyle='--',
    )

    def calc(self):
        wavelength = self.ifo.Laser.Wavelength
        materials = self.ifo.Materials
        ITM = mirror_struct(self.ifo, 'ITM')
        ETM = mirror_struct(self.ifo, 'ETM')
        cavity = arm_cavity(self.ifo)
        nITM, junk1, junk2, junk3 = noise.coatingthermal.coating_thermooptic(
            self.freq, ITM, wavelength, cavity.wBeam_ITM,
        )
        nETM, junk1, junk2, junk3 = noise.coatingthermal.coating_thermooptic(
            self.freq, ETM, wavelength, cavity.wBeam_ETM,
        )
        return (nITM + nETM) * 2


#########################
# substrate thermal
#########################

class ITMThermoRefractive(nb.Noise):
    """ITM Thermo-Refractive

    """
    style = dict(
        label='ITM Thermo-Refractive',
        color='#448ee4',
        linestyle='--',
    )

    def calc(self):
        power = ifo_power(self.ifo)
        gPhase = power.finesse * 2/np.pi
        cavity = arm_cavity(self.ifo)
        n = noise.substratethermal.substrate_thermorefractive(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        return n * 2 / gPhase**2


class SubstrateBrownian(nb.Noise):
    """Substrate Brownian

    """
    style = dict(
        label='Substrate Brownian',
        color='#fb7d07',
        linestyle='--',
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        nITM = noise.substratethermal.substrate_brownian(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        nETM = noise.substratethermal.substrate_brownian(
            self.freq, self.ifo.Materials, cavity.wBeam_ETM)
        return (nITM + nETM) * 2


class SubstrateThermoElastic(nb.Noise):
    """Substrate Thermo-Elastic

    """
    style = dict(
        label='Substrate Thermo-Elastic',
        color='#f5bf03',
        linestyle='--',
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        nITM = noise.substratethermal.substrate_thermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        nETM = noise.substratethermal.substrate_thermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ETM)
        return (nITM + nETM) * 2


#########################
# residual gas
#########################

# FIXME HACK: it's unclear if a phase noise in the arms like
# the excess gas noise should get the same dhdL strain
# calibration as the other displacement noises.  However, we
# would like to use the one Strain calibration for all noises,
# so we need to divide by the sinc_sqr here to undo the
# application of the dhdl in the Strain calibration.  But this
# is ultimately a superfluous extra calculation with the only
# to provide some simplication in the Budget definition, so
# should be re-evaluated at some point.

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
            species = self.ifo.Infrastructure.ResidualGas[species_name]
            n = noise.residualgas.residual_gas_scattering_arm(
                self.freq, self.ifo, cavity, species)
            dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
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

        @nb.precomp(sustf=suspension.precomp_suspension)
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
                n_ETM = noise.residualgas.residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film_ETM)
                n_ITM = noise.residualgas.residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film_ITM)
                n = 2 * (n_ETM + n_ITM)

            # Otherwise the same calculation is used for both.
            else:
                n = 4 * noise.residualgas.residual_gas_damping_test_mass(
                    self.freq, self.ifo, species, sustf, squeezed_film)

            return n

    return GasDampingSpecies


class ExcessGas(nb.Budget):
    """Excess Gas

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
