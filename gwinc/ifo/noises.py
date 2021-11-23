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

##################################################

def precomp_suspension(f, ifo):
    pc = Struct()
    pc.VHCoupling = Struct()
    if 'VHCoupling' in ifo.Suspension:
        pc.VHCoupling.theta = ifo.Suspension.VHCoupling.theta
    else:
        pc.VHCoupling.theta = ifo.Infrastructure.Length / const.R_earth
    hForce, vForce, hTable, vTable, tst_suscept = suspension.suspQuad(
        f, ifo.Suspension)
    pc.hForce = hForce
    pc.vForce = vForce
    pc.hTable = hTable
    pc.vTable = vTable
    pc.tst_suscept = tst_suscept
    return pc


@nb.precomp(sustf=precomp_suspension)
def precomp_quantum(f, ifo, sustf):
    pc = Struct()
    power = ifo_power(ifo)
    noise_dict = noise.quantum.shotrad(f, ifo, sustf, power)
    pc.ASvac = noise_dict['ASvac']
    pc.SEC = noise_dict['SEC']
    pc.Arm = noise_dict['arm']
    pc.Injection = noise_dict['injection']
    pc.PD = noise_dict['pd']

    # FC0 are the noises from the filter cavity losses and FC0_unsqzd_back
    # are noises from the unsqueezed vacuum injected at the back mirror
    # Right now there are at most one filter cavity in all the models;
    # if there were two, there would also be FC1 and FC1_unsqzd_back, etc.
    # keys = list(noise_dict.keys())
    fc_keys = [key for key in noise_dict.keys() if 'FC' in key]
    pc.FC = np.zeros_like(pc.ASvac)
    if fc_keys:
        for key in fc_keys:
            pc.FC += noise_dict[key]

    if 'phase' in noise_dict.keys():
        pc.Phase = noise_dict['phase']

    if 'ofc' in noise_dict.keys():
        pc.OFC = noise_dict['OFC']

    return pc


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

    @nb.precomp(quantum=precomp_quantum)
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

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.ASvac


class QuantumVacuumArm(nb.Noise):
    """Quantum vacuum due to arm cavity loss

    """
    style = dict(
        label='Arm Loss',
        color='xkcd:orange brown'
    )

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.Arm


class QuantumVacuumSEC(nb.Noise):
    """Quantum vacuum due to SEC loss

    """
    style = dict(
        label='SEC Loss',
        color='xkcd:cerulean'
    )

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.SEC


class QuantumVacuumFilterCavity(nb.Noise):
    """Quantum vacuum due to filter cavity loss

    """
    style = dict(
        label='Filter Cavity Loss',
        color='xkcd:goldenrod'
    )

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.FC


class QuantumVacuumInjection(nb.Noise):
    """Quantum vacuum due to injection loss

    """
    style = dict(
        label='Injection Loss',
        color='xkcd:fuchsia'
    )

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.Injection


class QuantumVacuumReadout(nb.Noise):
    """Quantum vacuum due to readout loss

    """
    style = dict(
        label='Readout Loss',
        color='xkcd:mahogany'
    )

    @nb.precomp(quantum=precomp_quantum)
    def calc(self, quantum):
        return quantum.PD


class QuantumVacuumQuadraturePhase(nb.Noise):
    """Quantum vacuum noise due to quadrature phase noise
    """
    style = dict(
        label='Quadrature Phase',
        color='xkcd:slate'
    )

    @nb.precomp(quantum=precomp_quantum)
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

class SeismicHorizontal(nb.Noise):
    """Horizontal seismic noise

    """
    style = dict(
        label='Horizontal',
        color='xkcd:muted blue',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        nt, nr = noise.seismic.platform_motion(self.freq, self.ifo)
        n = noise.seismic.seismic_suspension_filtered(sustf, nt, 'horiz')
        return n * 4


class SeismicVertical(nb.Noise):
    """Vertical seismic noise

    """
    style = dict(
        label='Vertical',
        color='xkcd:brick red',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        nt, nr = noise.seismic.platform_motion(self.freq, self.ifo)
        n = noise.seismic.seismic_suspension_filtered(sustf, nt, 'vert')
        return n * 4


class Seismic(nb.Budget):
    """Seismic

    """
    style = dict(
        label='Seismic',
        color='#855700',
    )

    noises = [
        SeismicHorizontal,
        SeismicVertical,
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

class SuspensionThermalHorizTop(nb.Noise):
    """Horizontal suspension thermal around the top mass

    """
    style = dict(
        label='Horiz. Top',
        color='xkcd:orangeish',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 0, 'horiz')
        return abs(n) * 4


class SuspensionThermalHorizAPM(nb.Noise):
    """Horizontal suspension thermal around the upper intermediate mass

    """
    style = dict(
        label='Horiz. APM',
        color='xkcd:mustard',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 1, 'horiz')
        return abs(n) * 4


class SuspensionThermalHorizPUM(nb.Noise):
    """Horizontal suspension thermal around the penultimate mass

    """
    style = dict(
        label='Horiz. PUM',
        color='xkcd:turquoise',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 2, 'horiz')
        return abs(n) * 4


class SuspensionThermalHorizTM(nb.Noise):
    """Horizontal suspension thermal around the test

    """
    style = dict(
        label='Horiz. Test mass',
        color='xkcd:bright purple',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 3, 'horiz')
        return abs(n) * 4


class SuspensionThermalVertTop(nb.Noise):
    """Vertical suspension thermal around the top mass

    """
    style = dict(
        label='Vert. Top',
        color='xkcd:orangeish',
        linestyle='--',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 0, 'vert')
        return abs(n) * 4


class SuspensionThermalVertAPM(nb.Noise):
    """Vertical suspension thermal around the upper intermediate mass

    """
    style = dict(
        label='Vert. APM',
        color='xkcd:mustard',
        linestyle='--',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 1, 'vert')
        return abs(n) * 4


class SuspensionThermalVertPUM(nb.Noise):
    """Vertical suspension thermal around the penultimate mass

    """
    style = dict(
        label='Vert. PUM',
        color='xkcd:turquoise',
        linestyle='--',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 2, 'vert')
        return abs(n) * 4


class SuspensionThermalVertTM(nb.Noise):
    """Vertical suspension thermal around the test

    """
    style = dict(
        label='Vert. Test mass',
        color='xkcd:bright purple',
        linestyle='--',
        alpha=0.7,
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.susptherm_stage(
            self.freq, self.ifo.Suspension, sustf, 3, 'vert')
        return abs(n) * 4


class SuspensionThermal(nb.Budget):
    """Suspension Thermal

    """

    name = 'SuspensionThermal'

    style = dict(
        label='Suspension Thermal',
        color='#0d75f8',
    )

    noises = [
        SuspensionThermalHorizTop,
        SuspensionThermalHorizAPM,
        SuspensionThermalHorizPUM,
        SuspensionThermalHorizTM,
        SuspensionThermalVertTop,
        SuspensionThermalVertAPM,
        SuspensionThermalVertPUM,
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


def calc_total_residual_gas_damping(f, ifo, species, sustf):
    """Total residual gas damping from all four test masses

    :f: frequency array in Hz
    :ifo: gwinc IFO structure
    :species: molecular species structure
    :sustf: suspension transfer function structure

    :returns: displacement noise
    """
    squeezed_film = ifo.Infrastructure.ResidualGas.get('SqueezedFilm', Struct())

    if squeezed_film is None:
        raise ValueError('Must specify either excess damping or a gap')

    # Calculate squeezed film for ETM and ITM seperately if either is given
    # explicitly. If only one is given, it is not computed for the other one.
    if ('ETM' in squeezed_film) or ('ITM' in squeezed_film):
        squeezed_film_ETM = squeezed_film.get('ETM', Struct())
        squeezed_film_ITM = squeezed_film.get('ITM', Struct())
        n_ETM = noise.residualgas.residual_gas_damping_test_mass(
            f, ifo, species, sustf, squeezed_film_ETM)
        n_ITM = noise.residualgas.residual_gas_damping_test_mass(
            f, ifo, species, sustf, squeezed_film_ITM)
        n = 2 * (n_ETM + n_ITM)

    # Otherwise the same calculation is used for both.
    else:
        n = 4 * noise.residualgas.residual_gas_damping_test_mass(
            f, ifo, species, sustf, squeezed_film)

    return n


class ExcessGasScatteringH2(nb.Noise):
    """Excess gas scattering for H2

    """
    style = dict(
        label='H$_2$ scattering',
        color='xkcd:red orange'
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        species = self.ifo.Infrastructure.ResidualGas.H2
        n = noise.residualgas.residual_gas_scattering_arm(
            self.freq, self.ifo, cavity, species)
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return n * 2 / sinc_sqr


class ExcessGasScatteringN2(nb.Noise):
    """Excess gas scattering for N2

    """
    style = dict(
        label='N$_2$ scattering',
        color='xkcd:emerald'
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        species = self.ifo.Infrastructure.ResidualGas.N2
        n = noise.residualgas.residual_gas_scattering_arm(
            self.freq, self.ifo, cavity, species)
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return n * 2 / sinc_sqr


class ExcessGasScatteringH2O(nb.Noise):
    """Excess gas scattering for H2O

    """
    style = dict(
        label='H$_2$O scattering',
        color='xkcd:water blue'
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        species = self.ifo.Infrastructure.ResidualGas.H2O
        n = noise.residualgas.residual_gas_scattering_arm(
            self.freq, self.ifo, cavity, species)
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return n * 2 / sinc_sqr


class ExcessGasScatteringO2(nb.Noise):
    """Excess gas scattering for O2

    """
    style = dict(
        label='O$_2$ scattering',
        color='xkcd:grey'
    )

    def calc(self):
        cavity = arm_cavity(self.ifo)
        species = self.ifo.Infrastructure.ResidualGas.O2
        n = noise.residualgas.residual_gas_scattering_arm(
            self.freq, self.ifo, cavity, species)
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return n * 2 / sinc_sqr


class ExcessGasDampingH2(nb.Noise):
    """Residual gas damping for H2

    """
    style = dict(
        label='H$_2$ damping',
        color='xkcd:red orange',
        linestyle='--',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        species = self.ifo.Infrastructure.ResidualGas.H2
        return calc_total_residual_gas_damping(self.freq, self.ifo, species, sustf)


class ExcessGasDampingN2(nb.Noise):
    """Excess gas damping for N2

    """
    style = dict(
        label='N$_2$ damping',
        color='xkcd:emerald',
        linestyle='--',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        species = self.ifo.Infrastructure.ResidualGas.N2
        return calc_total_residual_gas_damping(self.freq, self.ifo, species, sustf)


class ExcessGasDampingH2O(nb.Noise):
    """Excess gas damping for H2O

    """
    style = dict(
        label='H$_2$O damping',
        color='xkcd:water blue',
        linestyle='--',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        species = self.ifo.Infrastructure.ResidualGas.H2O
        return calc_total_residual_gas_damping(self.freq, self.ifo, species, sustf)


class ExcessGasDampingO2(nb.Noise):
    """Excess gas damping for O2

    """
    style = dict(
        label='O$_2$ damping',
        color='xkcd:grey',
        linestyle='--',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        species = self.ifo.Infrastructure.ResidualGas.O2
        return calc_total_residual_gas_damping(self.freq, self.ifo, species, sustf)


class ExcessGas(nb.Budget):
    """Excess Gas

    """
    style = dict(
        label='Residual Gas',
        color='#add00d',
        linestyle='-',
    )

    noises = [
        ExcessGasScatteringH2,
        ExcessGasScatteringN2,
        ExcessGasScatteringH2O,
        ExcessGasScatteringO2,
        ExcessGasDampingH2,
        ExcessGasDampingN2,
        ExcessGasDampingH2O,
        ExcessGasDampingO2,
    ]
