import copy
import numpy as np
from numpy import pi, sin, exp, sqrt

from .. import logger
from .. import const
from ..struct import Struct
from .. import nb
from .. import noise
from .. import suspension

##################################################


def coating_thickness(materials, optic):
    if 'CoatLayerOpticalThickness' in optic:
        return optic['CoatLayerOpticalThickness']
    T = optic.Transmittance
    dL = optic.CoatingThicknessLown
    dCap = optic.CoatingThicknessCap
    return noise.coatingthermal.getCoatDopt(materials, T, dL, dCap=dCap)


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
    mirror.Coating.dOpt = coating_thickness(ifo.Materials, optic)
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
    return w0, w1, w2


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

    return pbs, parm, finesse, prfactor, Tpr


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

##################################################


def precomp_suspension(f, ifo):
    pc = Struct()
    pc.VHCoupling = Struct()
    if 'VHCoupling' in ifo.Suspension:
        pc.VHCoupling.theta = ifo.Suspension.VHCoupling.theta
    else:
        pc.VHCoupling.theta = ifo.Infrastructure.Length / const.R_earth
    hForce, vForce, hTable, vTable = suspension.suspQuad(f, ifo.Suspension)
    pc.hForce = hForce
    pc.vForce = vForce
    pc.hTable = hTable
    pc.vTable = vTable
    return pc


def precomp_mirror(f, ifo):
    pc = Struct()
    pc.ITM = mirror_struct(ifo, 'ITM')
    pc.ETM = mirror_struct(ifo, 'ETM')
    return pc


def precomp_power(f, ifo):
    pc = Struct()
    pbs, parm, finesse, prfactor, Tpr = ifo_power(ifo)
    pc.pbs = pbs
    pc.parm = parm
    pc.finesse = finesse
    pc.gPhase = finesse * 2/np.pi
    pc.prfactor = prfactor
    return pc


def precomp_cavity(f, ifo):
    pc = Struct()
    w0, wBeam_ITM, wBeam_ETM = arm_cavity(ifo)
    pc.w0 = w0
    pc.wBeam_ITM = wBeam_ITM
    pc.wBeam_ETM = wBeam_ETM
    return pc

##################################################

class Strain(nb.Calibration):
    def calc(self):
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return dhdl_sqr

##################################################

class QuantumVacuum(nb.Noise):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    @nb.precomp(mirror=precomp_mirror)
    @nb.precomp(power=precomp_power)
    def calc(self, mirror, power):
        return noise.quantum.shotrad(self.freq, self.ifo, mirror.ETM.MirrorMass, power)


class StandardQuantumLimit(nb.Noise):
    """Standard Quantum Limit

    """
    style = dict(
        label="Standard Quantum Limit",
        color="#000000",
        linestyle=":",
    )

    @nb.precomp(mirror=precomp_mirror)
    def calc(self, mirror):
        return 8 * const.hbar / (mirror.ETM.MirrorMass * (2 * np.pi * self.freq) ** 2)


class Seismic(nb.Noise):
    """Seismic

    """
    style = dict(
        label='Seismic',
        color='#855700',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        if 'PlatformMotion' in self.ifo.Seismic:
            if self.ifo.Seismic.PlatformMotion == 'BSC':
                nt, nr = noise.seismic.seismic_BSC_ISI(self.freq)
            elif self.ifo.Seismic.PlatformMotion == '6D':
                nt, nr = noise.seismic.seismic_BSC_ISI_6D(self.freq)
            else:
                nt, nr = noise.seismic.seismic_BSC_ISI(self.freq)
        else:
            nt, nr = noise.seismic.seismic_BSC_ISI(self.freq)
        n, nh, nv = noise.seismic.seismic_suspension_fitered(
            self.ifo.Suspension, sustf, nt)
        return n * 4


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


class SuspensionThermal(nb.Noise):
    """Suspension Thermal

    """
    style = dict(
        label='Suspension Thermal',
        color='#0d75f8',
    )

    @nb.precomp(sustf=precomp_suspension)
    def calc(self, sustf):
        n = noise.suspensionthermal.suspension_thermal(
            self.freq, self.ifo.Suspension, sustf)
        return n * 4


class CoatingBrownian(nb.Noise):
    """Coating Brownian

    """
    style = dict(
        label='Coating Brownian',
        color='#fe0002',
    )

    @nb.precomp(mirror=precomp_mirror)
    @nb.precomp(cavity=precomp_cavity)
    @nb.precomp(power=precomp_power)
    def calc(self, mirror, cavity, power):
        wavelength = self.ifo.Laser.Wavelength
        nITM = noise.coatingthermal.coating_brownian(
            self.freq, mirror.ITM, wavelength, cavity.wBeam_ITM, power.parm,
        )
        nETM = noise.coatingthermal.coating_brownian(
            self.freq, mirror.ETM, wavelength, cavity.wBeam_ETM, power.parm,
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

    @nb.precomp(mirror=precomp_mirror)
    @nb.precomp(cavity=precomp_cavity)
    def calc(self, mirror, cavity):
        wavelength = self.ifo.Laser.Wavelength
        materials = self.ifo.Materials
        nITM, junk1, junk2, junk3 = noise.coatingthermal.coating_thermooptic(
            self.freq, mirror.ITM, wavelength, cavity.wBeam_ITM,
        )
        nETM, junk1, junk2, junk3 = noise.coatingthermal.coating_thermooptic(
            self.freq, mirror.ETM, wavelength, cavity.wBeam_ETM,
        )
        return (nITM + nETM) * 2


class ITMThermoRefractive(nb.Noise):
    """ITM Thermo-Refractive

    """
    style = dict(
        label='ITM Thermo-Refractive',
        color='#448ee4',
        linestyle='--',
    )

    @nb.precomp(cavity=precomp_cavity)
    @nb.precomp(power=precomp_power)
    def calc(self, cavity, power):
        n = noise.substratethermal.substrate_thermorefractive(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        return n * 2 / power.gPhase**2


class SubstrateBrownian(nb.Noise):
    """Substrate Brownian

    """
    style = dict(
        label='Substrate Brownian',
        color='#fb7d07',
        linestyle='--',
    )

    @nb.precomp(cavity=precomp_cavity)
    def calc(self, cavity):
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

    @nb.precomp(cavity=precomp_cavity)
    def calc(self, cavity):
        nITM = noise.substratethermal.substrate_thermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ITM)
        nETM = noise.substratethermal.substrate_thermoelastic(
            self.freq, self.ifo.Materials, cavity.wBeam_ETM)
        return (nITM + nETM) * 2


class ExcessGas(nb.Noise):
    """Excess Gas

    """
    style = dict(
        label='Excess Gas',
        color='#add00d',
        linestyle='--',
    )

    def calc(self):
        n = noise.residualgas.residual_gas_cavity(self.freq, self.ifo)
        # FIXME HACK: it's unclear if a phase noise in the arms like
        # the excess gas noise should get the same dhdL strain
        # calibration as the other displacement noises.  However, we
        # would like to use the one Strain calibration for all noises,
        # so we need to divide by the sinc_sqr here to undo the
        # application of the dhdl in the Strain calibration.  But this
        # is ultimately a superfluous extra calculation with the only
        # to provide some simplication in the Budget definition, so
        # should be re-evaluated at some point.
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        return n * 2 / sinc_sqr
