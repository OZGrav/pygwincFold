import numpy as np
from numpy import pi, sin, exp, sqrt

from .. import logger
from .. import const
from ..struct import Struct
from .. import nb
from .. import suspension


############################################################
# helper functions
############################################################


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

    #allow either the input power or the arm power to be the principle power used
    #input power is good for new budgets, while arm power is good for site noise
    #budgets
    pin = ifo.Laser.get('Power', None)
    parm = ifo.Laser.get('ArmPower', None)
    if pin is not None:
        if parm is not None:
            #TODO, make a ConfigError or IfoError?
            raise RuntimeError("Cannot specify both Laser.Power and Laser.ArmPower")
        pbs = pin * prfactor  # BS power from input power
        parm = pbs * garm**2 / 2  # arm power from BS power
    else:
        if parm is None:
            #TODO, make a ConfigError or IfoError?
            raise RuntimeError("Need to specify either Laser.Power or Laser.ArmPower")
        pbs = parm / (garm**2 / 2)  # arm power from BS power
        pin = pbs / prfactor  # BS power from input power

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
        from ..noise.coatingthermal import mirror_struct

        mass = mirror_struct(self.ifo, 'ETM').MirrorMass
        return (mass * (2*pi*self.freq)**2)**2


class Acceleration(nb.Calibration):
    """Calibrate displacement to acceleration
    """
    def calc(self):
        return (2*pi*self.freq)**4


class Velocity(nb.Calibration):
    """Calibrate displacement to velocity
    """
    def calc(self):
        return (2*pi*self.freq)**2
