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

def qITM(RcITM, RcETM, RcFM, length):
    q1 = (-4*length**4 + 2*length*RcETM*RcFM**2 + 4*length**3*(RcETM+2*RcFM)- 2*length**2*RcFM*(3*RcETM+2*RcFM)+np.sqrt(2)*RcETM*RcFM**2*RcITM*np.sqrt(1/(RcETM**2*RcFM**4*RcITM**2)*length*(length-RcFM)*(2*length**2+RcETM*RcFM-2*length*(RcETM+RcFM))*(2*length**2 +RcFM*RcITM - 2*length*(RcFM+RcITM))*(2*length**2+RcETM*RcFM+2*RcETM*RcITM+RcFM*RcITM-2*length*(RcETM+RcFM+RcITM)),dtype=np.complex128()))/(8*length**3 - 4*length**2 *(2*RcETM+3*RcFM+RcITM)+4*length*(RcFM*(RcFM+RcITM)+RcETM*(2*RcFM+RcITM))-RcFM*(RcFM*RcITM+RcETM*(RcFM+2*RcITM)))
    q2 = (-4*length**4 + 2*length*RcETM*RcFM**2 + 4*length**3*(RcETM+2*RcFM)- 2*length**2*RcFM*(3*RcETM+2*RcFM)-np.sqrt(2)*RcETM*RcFM**2*RcITM*np.sqrt(1/(RcETM**2*RcFM**4*RcITM**2)*length*(length-RcFM)*(2*length**2+RcETM*RcFM-2*length*(RcETM+RcFM))*(2*length**2 +RcFM*RcITM - 2*length*(RcFM+RcITM))*(2*length**2+RcETM*RcFM+2*RcETM*RcITM+RcFM*RcITM-2*length*(RcETM+RcFM+RcITM)),dtype=np.complex128()))/(8*length**3 - 4*length**2 *(2*RcETM+3*RcFM+RcITM)+4*length*(RcFM*(RcFM+RcITM)+RcETM*(2*RcFM+RcITM))-RcFM*(RcFM*RcITM+RcETM*(RcFM+2*RcITM)))
    q1_1 = np.where(np.imag(q1) > 0, q1, 0)
    q2_1 = np.where(np.imag(q2) > 0, q2, 0)
    q_out = np.where((q1_1+q2_1) != 0, (q1_1+q2_1), np.nan)
    return q_out

def qETM(RcITM, RcETM, RcFM, length):

    q1 = (-4*length**4 + 2*length*RcITM*RcFM**2 + 4*length**3*(RcITM+2*RcFM)- 2*length**2*RcFM*(3*RcITM+2*RcFM)+np.sqrt(2)*RcETM*RcFM**2*RcITM*np.sqrt(1/(RcETM**2*RcFM**4*RcITM**2)*length*(length-RcFM)*(2*length**2+RcETM*RcFM-2*length*(RcETM+RcFM))*(2*length**2 +RcFM*RcITM - 2*length*(RcFM+RcITM))*(2*length**2+RcETM*RcFM+2*RcETM*RcITM+RcFM*RcITM-2*length*(RcETM+RcFM+RcITM)),dtype=np.complex128()))/(8*length**3 - 4*length**2 *(RcETM+3*RcFM+2*RcITM)+4*length*(RcETM*(RcFM+RcITM)+RcFM*(RcFM+2*RcITM))-RcFM*(RcFM*RcITM+RcETM*(RcFM+2*RcITM)))
    q2 = (-4*length**4 + 2*length*RcITM*RcFM**2 + 4*length**3*(RcITM+2*RcFM)- 2*length**2*RcFM*(3*RcITM+2*RcFM)-np.sqrt(2)*RcETM*RcFM**2*RcITM*np.sqrt(1/(RcETM**2*RcFM**4*RcITM**2)*length*(length-RcFM)*(2*length**2+RcETM*RcFM-2*length*(RcETM+RcFM))*(2*length**2 +RcFM*RcITM - 2*length*(RcFM+RcITM))*(2*length**2+RcETM*RcFM+2*RcETM*RcITM+RcFM*RcITM-2*length*(RcETM+RcFM+RcITM)),dtype=np.complex128()))/(8*length**3 - 4*length**2 *(RcETM+3*RcFM+2*RcITM)+4*length*(RcETM*(RcFM+RcITM)+RcFM*(RcFM+2*RcITM))-RcFM*(RcFM*RcITM+RcETM*(RcFM+2*RcITM)))
    q1_1 = np.where(np.imag(q1) > 0, q1, 0)
    q2_1 = np.where(np.imag(q2) > 0, q2, 0)
    q_out = np.where((q1_1+q2_1) != 0, (q1_1+q2_1), np.nan)
    return q_out

def wITM(RcITM, RcETM, RcFM, length, wavelength):
    q = qITM(RcITM, RcETM, RcFM, length)
    return np.sqrt(wavelength/np.pi * np.abs(q)**2/np.imag(q))


def wFM(RcITM, RcETM, RcFM, length, wavelength):
    q = qITM(RcITM, RcETM, RcFM, length) + length
    return np.sqrt(wavelength/np.pi * np.abs(q)**2/np.imag(q))


def wETM(RcITM, RcETM, RcFM, length, wavelength):
    q = qETM(RcITM, RcETM, RcFM, length)
    return np.sqrt(wavelength/np.pi * np.abs(q)**2/np.imag(q))

def gFactor(RcITM, RcETM, RcFM, length):
    g = ((2*length**2+RcETM*RcFM-2*length*(RcETM+RcFM))*(2*length**2+RcFM*RcITM-2*length*(RcFM+RcITM)))/(RcETM*RcFM**2*RcITM)
    return g

def arm_cavity(ifo):
    L = ifo.Infrastructure.FoldedLength
    wavelength = ifo.Laser.Wavelength
    Rc1 = ifo.Optics.Curvature.ITM
    Rc2 = ifo.Optics.Curvature.ETM
    Rc3 = ifo.Optics.Curvature.FM
    
    w1 = wITM(Rc1, Rc2, Rc3, L, wavelength)
    w2 = wETM(Rc1, Rc2, Rc3, L, wavelength)
    w3 = wFM(Rc1, Rc2, Rc3, L, wavelength)

    gcav = gFactor(Rc1, Rc2, Rc3, L)	

    if np.logical_and(gcav<=0, gcav>=1):
        raise Exception('Unstable arm cavity g-factors.  Change ifo.Optics.Curvature')
    elif np.logical_and(gcav<1e-3, gcav>(1-1e-3)):
        logger.warning('Nearly unstable arm cavity g-factors.  Reconsider ifo.Optics.Curvature')

    # waist, input, output
    cavity = Struct()
    cavity.w0 = np.nan
    cavity.wBeam_ITM = w1
    cavity.wBeam_ETM = w2
    cavity.wBeam_FM = w3
    cavity.zr = np.nan
    cavity.zBeam_ITM = np.nan
    cavity.zBeam_ETM = np.nan
    return cavity


def ifo_power(ifo, PRfixed=True):
    """Compute power on beamsplitter, finesse, and power recycling factor.

    """
    t1 = sqrt(ifo.Optics.ITM.Transmittance)
    r1 = sqrt(1 - ifo.Optics.ITM.Transmittance)
    r2 = sqrt(1 - ifo.Optics.ETM.Transmittance)
    r3 = sqrt(1 - ifo.Optics.FM.Transmittance)
    t5 = sqrt(ifo.Optics.PRM.Transmittance)
    r5 = sqrt(1 - ifo.Optics.PRM.Transmittance)
    T1 = ifo.Optics.ITM.Transmittance
    T2 = ifo.Optics.ETM.Transmittance
    T3 = ifo.Optics.FM.Transmittance
    loss = ifo.Optics.Loss  # single TM loss
    bsloss = ifo.Optics.BSLoss
    acoat = ifo.Optics.ITM.CoatingAbsorption
    pcrit = ifo.Optics.pcrit

    # Finesse, effective number of bounces in cavity, power recycling factor
    rho = (1-T1)*(1-T2)*(1-T3)*(1-loss)**3
    finesse = np.pi/(2*np.arcsin((1-np.sqrt(rho))/(2*rho**0.25)))  # arm cavity finesse
    neff = 2 * finesse / pi

    # Arm cavity reflectivity with finite loss
    garm = t1 / (1 - r1*r2*sqrt(1-(3*loss+T3)))  # amplitude gain wrt input field
    rarm = r1 - t1 * r2 * sqrt(1-(3*loss+T3)) * garm

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

class NoCalibration(nb.Calibration):
    """Calibrate displacement to strain
    """
    def calc(self):
        dhdl_sqr = dhdl2(self.freq, self.ifo.Infrastructure.Length)
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
