'''Functions to calculate quantum noise

'''
from __future__ import division
import numpy as np
from numpy import pi, sqrt, arctan, sin, cos, exp, log10, conj
from copy import deepcopy
from collections.abc import Sequence

from .. import logger
from .. import const
from ..struct import Struct


def sqzOptimalSqueezeAngle(Mifo, eta):
    vHD = np.array([[sin(eta), cos(eta)]])
    H = getProdTF(vHD, Mifo)[0]
    alpha = arctan(abs(H[1]), abs(H[0]))
    return alpha


def getSqzParams(ifo):
    """Determine squeezer type, if any, and extract common parameters

    Returns a struct with the following attributes:
    SQZ_DB: squeezing in dB
    ANTISQZ_DB: anti-squeezing in dB
    alpha: freq Indep Squeeze angle [rad]
    lambda_in: loss to squeezing before injection [Power]
    etaRMS: quadrature noise [rad]
    eta: homodyne angle [rad]
    """
    params = Struct()

    if 'Squeezer' not in ifo:
        sqzType = None
    else:
        sqzType = ifo.Squeezer.get('Type', 'Freq Independent')

    params.sqzType = sqzType

    # extract squeezer parameters
    if sqzType is None:
        params.SQZ_DB = 0
        params.ANTISQZ_DB = 0
        params.alpha = 0
        params.lambda_in = 0
        params.etaRMS = 0

    else:
        params.SQZ_DB = ifo.Squeezer.AmplitudedB
        params.ANTISQZ_DB = ifo.Squeezer.get('AntiAmplitudedB', params.SQZ_DB)
        params.alpha = ifo.Squeezer.SQZAngle
        params.lambda_in = ifo.Squeezer.InjectionLoss
        params.etaRMS = ifo.Squeezer.get('LOAngleRMS', 0)

    # Homodyne Readout phase
    eta_orig = ifo.Optics.get('Quadrature', Struct()).get('dc', None)

    ifoRead = ifo.get('Squeezer', Struct()).get('Readout', None)
    if ifoRead is None:
        eta = eta_orig
        if eta_orig is None:
            raise Exception("must add Quadrature.dc or Readout...")

    elif ifoRead.Type == 'DC':
        eta = np.sign(ifoRead.fringe_side) * \
            np.arccos((ifoRead.defect_PWR_W / ifoRead.readout_PWR_W)**.5)

    elif ifoRead.Type == 'Homodyne':
        eta = ifoRead.Angle

    else:
        raise Exception("Unknown Readout Type")

    if eta_orig is not None:
        # logger.warn((
        #     'Quadrature.dc is redundant with '
        #     'Squeezer.Readout and is deprecated.'
        # ))
        if eta_orig != eta:
            raise Exception("Quadrature.dc inconsistent with Readout eta")

    params.eta = eta

    return params


######################################################################
# Main quantum noise function
######################################################################


def shotrad(f, ifo, sustf, power):
    """Quantum noise strain spectrum

    :f: frequency array in Hz
    :ifo: gwinc IFO Struct
    :sustf: suspension transfer function struct
    :power: gwinc power Struct

    :returns: displacement noise power spectrum at :f:

    corresponding author: mevans

    """
    ######################################################################
    # Vacuum sources will be individually tracked with the Mnoise dict   #
    # The keys are the following                                         #
    #   * arm: arm cavity losses                                         #
    #   * SEC: SEC losses                                                #
    #   * ASvac: AS port vacuum                                          #
    #   * pd: readout losses                                             #
    #   * injection: injection losses                                    #
    #   * phase: quadrature phase noise                                  #
    #                                                                    #
    # If there is a filter cavity there are an additional set of keys    #
    # starting with 'FC' for the filter cavity losses                    #
    #                                                                    #
    # If there is an output filter cavity the key 'OFC' gives its losses #
    ######################################################################

    # call the main IFO Quantum Model
    if 'Type' not in ifo.Optics or ifo.Optics.Type == 'SignalRecycled':
        coeff, Mifo, Msig, Mn = shotradSignalRecycled(f, ifo, sustf, power)
    elif ifo.Optics.Type == 'SignalRecycledBnC':
        coeff, Mifo, Msig, Mn = shotradSignalRecycledBnC(f, ifo, power)
    else:
        raise ValueError('Unrecognized IFO type ' + ifo.Optics.Type)

    # separate arm and SEC loss
    Mnoise = dict(arm=Mn[:, :2, :], SEC=Mn[:, 2:, :])

    sqz_params = getSqzParams(ifo)

    if sqz_params.sqzType == 'Optimal':
        # compute optimal squeezing angle
        sqz_params.alpha = sqzOptimalSqueezeAngle(Mifo, sqz_params.eta)

    vHD = np.array([[sin(sqz_params.eta), cos(sqz_params.eta)]])

    def homodyne(signal):
        """Readout the eta quadrature of the signal signal
        """
        return np.squeeze(np.sum(abs(getProdTF(vHD, signal))**2, axis=1))

    # optomechanical plant
    lambda_PD = 1 - ifo.Optics.PhotoDetectorEfficiency  # PD losses
    Msig = Msig * sqrt(1 - lambda_PD)
    plant = homodyne(Msig)

    # get all the losses in the squeezed quadrature
    Mnoise = propagate_noise_fc_ifo(
        f, ifo, Mifo, Mnoise, sqz_params, sqz_params.alpha)

    # add quadrature phase fluctuations if any
    if sqz_params.etaRMS:
        # get all losses in the anti-squeezed quadrature
        Mnoise_antisqz = propagate_noise_fc_ifo(
            f, ifo, Mifo, Mnoise, sqz_params, sqz_params.alpha + pi/2)

        # sum over these since they're not individually tracked
        var_antisqz = np.zeros_like(f)
        for nn in Mnoise_antisqz.values():
            var_antisqz += homodyne(nn)

        # The total variance is
        # V(alpha) * cos(etaRMS)^2 + V(alpha + pi/2) * sin(etaRMS)^2
        Mnoise = {key: cos(sqz_params.etaRMS)**2 * homodyne(nn)
                  for key, nn in Mnoise.items()}
        Mnoise['phase'] = sin(sqz_params.etaRMS)**2 * var_antisqz

    else:
        Mnoise = {key: homodyne(nn) for key, nn in Mnoise.items()}

    # calibrate into displacement
    # coeff can be removed if shotradSignalRecycledBnC isn't used
    Mnoise = {key: coeff*nn/plant for key, nn in Mnoise.items()}
    psd = {key: nn * ifo.Infrastructure.Length**2 for key, nn in Mnoise.items()}

    return psd


######################################################################
# The following two compile functions generate analytic expressions
# that are hard coded in shotradSignalRecycled and are not called
# when shotrad is called
######################################################################


def compile_ARM_RES_TF():
    import sympy as sp
    ID = sp.eye(2)
    rITM, tArm, exp_2jOmegaL_c, K = sp.symbols('rITM tArm exp_2jOmegaL_c K')
    ARM = tArm * exp_2jOmegaL_c * sp.Matrix([[1, 0], [-K, 1]])
    ARM_RES = (ID - rITM*ARM)**-1

    subexprs, ARM_RES_expr = sp.cse(ARM_RES)
    for expr in subexprs:
        print(str(expr[0]), '=', str(expr[1]))
    print('RES', '=', str(ARM_RES_expr[0]).replace('Matrix', 'np.array').replace(', 0]', ', np.zeros(nf)]'))


def compile_SEC_RES_TF():
    import sympy as sp
    ID = sp.eye(2)
    phi, exp_1jOmegal_c, tArm, exp_2jOmegaL_c, K, r00, r10, r11, R, T, rITM, tSR, rho = sp.symbols('phi exp_1jOmegal_c tArm exp_2jOmegaL_c K r00 r10 r11 R T rITM tSR rho')
    SEr = sp.Matrix([[sp.cos(phi), sp.sin(phi)], [-sp.sin(phi), sp.cos(phi)]])
    SE = SEr * exp_1jOmegal_c
    ARM = tArm * exp_2jOmegaL_c * sp.Matrix([[1, 0], [-K, 1]])
    ARM_RES = sp.Matrix([[r00, 0], [r10, r11]])
    rho_ARM = ARM_RES * ((R + T) * ARM - rITM * ID)
    SEC = tSR * SE * rho_ARM * SE
    SEC_RES = (ID + rho*SEC)**-1

    subexprs, SEC_RES_expr = sp.cse(SEC_RES)
    for expr in subexprs:
        print(str(expr[0]), '=', str(expr[1]))
    print('RES', '=', str(SEC_RES_expr[0]).replace('Matrix', 'np.array'))


######################################################################
# Main IFO quantum models
######################################################################

def shotradSignalRecycled(f, ifo, sustf, power):
    """Quantum noise model for signal recycled IFO (see shotrad for more info)

    New version July 2016 by JH based on transfer function formalism

    coeff = frequency dependent overall noise coefficient (Nx1)
            (not required anymore, but kept for compatibility with shotrad.m)
    Mifo = IFO input-output relation for the AS port
    Msig = signal transfer to the AS port
    Mnoise = noise fields produced by losses in the IFO at the AS port

    """
    lambda_ = ifo.Laser.Wavelength               # Laser Wavelength [m]
    hbar = const.hbar                            # Plancks Constant [Js]
    c = const.c                                  # SOL [m/s]
    omega_0 = 2*pi*c/lambda_                     # Laser angular frequency [rads/s]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    L = ifo.Infrastructure.Length                # Length of arm cavities [m]
    l = ifo.Optics.SRM.CavityLength              # SRC Length [m]
    T = ifo.Optics.ITM.Transmittance             # ITM Transmittance [Power]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    bsloss = ifo.Optics.BSLoss                   # BS Loss [Power]
    mismatch = 1 - ifo.Optics.coupling           # Mismatch
    mismatch = mismatch + ifo.TCS.SRCloss        # Mismatch

    # BSloss + mismatch has been incorporated into a SRC Loss
    lambda_SR = 1 - (1 - mismatch) * (1 - bsloss)  # SR cavity loss [Power]

    tau = sqrt(ifo.Optics.SRM.Transmittance)     # SRM Transmittance [amplitude]
    rho = sqrt(1 - tau**2)                       # SRM Reflectivity [amplitude]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ds = ifo.Optics.SRM.Tunephase                # SRC Detunning
    phi = ds/2

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    lambda_arm = 1 - (1 - ifo.Optics.Loss)**2 * (1 - ifo.Optics.ETM.Transmittance)

    R = 1 - T - ifo.Optics.Loss                  # ITM Reflectivity [Power]

    P = power.parm                               # use precomputed value

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    nf = len(f)

    ID = np.array([[np.ones(nf), np.zeros(nf)], [np.zeros(nf), np.ones(nf)]])

    # transfer matrices for dark port input and signal field
    Mifo = np.zeros((2, 2, nf), dtype=complex)
    Msig = np.zeros((2, 1, nf), dtype=complex)

    # transfer matrices for SEC and arm loss fields
    Mp = np.zeros((2, 2, nf), dtype=complex)
    Mn = np.zeros((2, 2, nf), dtype=complex)

    # SRC rotation matrix
    SEr = np.array([[np.tile(cos(phi), nf), np.tile(sin(phi), nf)],
                    [np.tile(-sin(phi), nf), np.tile(cos(phi), nf)]])

    # some precomputed parameters
    tITM = sqrt(T)
    rITM = sqrt(R)

    tArm = sqrt(1 - lambda_arm)
    tSR = sqrt(1 - lambda_SR)
    tSig = sqrt((1 - lambda_arm / 2) * (1 - lambda_SR / 2))
    RT_SRM = rho**2 + tau**2

    lossArm = sqrt(lambda_arm)
    lossSR = sqrt(lambda_SR)

    Omega = 2*pi*f                                          # Signal angular frequency [rad/s]
    h_SQL = sqrt(8 * hbar * np.abs(sustf.tst_suscept)) / L  # SQL Strain
    K = - 16 * P * omega_0 / c**2 * sustf.tst_suscept

    # arm cavity
    exp_2jOmegaL_c = exp(2j*Omega*L/c)
    ARM = tArm * exp_2jOmegaL_c * np.array([[np.ones(nf), np.zeros(nf)], [-K, np.ones(nf)]])

    # the following code is generated by compile_ARM_RES_TF()
    # and is equivalent to the following:
    # RES = np.zeros((2,2,nf), dtype=complex)
    # RES = np.linalg.pinv(ID.transpose((2,0,1)) - rITM * ARM.transpose((2,0,1))).transpose((1,2,0))
    x0 = exp_2jOmegaL_c*rITM*tArm
    x1 = -x0 + 1
    x2 = 1/x1
    RES = np.array([[x2, np.zeros(nf)], [-K*x0/x1**2, x2]])
    # end of generated code

    rho_ARM = getProdTF(RES, (R + T) * ARM - rITM * ID)
    tau_ARM = tITM * RES

    # signal-extraction cavity
    SE = SEr * exp(1j * Omega * l / c)
    SEC = getProdTF(tSR * SE, getProdTF(rho_ARM, SE))

    exp_1jOmegal_c = exp(1j*Omega*l/c)
    r00 = RES[0,0,:]
    r10 = RES[1,0,:]
    r11 = RES[1,1,:]

    # the following code is generated by compile_SEC_RES_TF()
    # and is equivalent to the following:
    # RES = np.zeros((2,2,nf), dtype=complex)
    # RES = np.linalg.pinv(ID.transpose((2,0,1)) + rho * SEC.transpose((2,0,1))).transpose((1,2,0))
    x0 = cos(phi)
    x1 = exp_2jOmegaL_c*tArm*(R + T)
    x2 = -rITM + x1
    x3 = exp_1jOmegal_c**2*r11*tSR*x2
    x4 = sin(phi)
    x5 = exp_1jOmegal_c*r00*tSR*x2
    x6 = exp_1jOmegal_c*tSR*(-K*r11*x1 + r10*x2)
    x7 = exp_1jOmegal_c*(x0*x6 - x4*x5)
    x8 = rho*(x0**2*x3 + x4*x7) + 1
    x9 = x0*x3*x4
    x10 = exp_1jOmegal_c*(x0*x5 + x4*x6)
    x11 = x10*x4 + x9
    x12 = x0*x7 - x9
    x13 = rho*(x0*x10 - x3*x4**2) + 1
    x14 = 1/(-rho**2*x11*x12 + x13*x8)
    x15 = rho*x14
    RES = np.array([[x14*x8, -x11*x15], [-x12*x15, x13*x14]])
    # end of generated code

    rho_SEC = getProdTF(RES, RT_SRM * SEC + rho * ID)
    tau_SEC = tau * getProdTF(RES, SE)
    tau_SEC_ARM = getProdTF(tau_SEC, tau_ARM)

    # signal field
    Msig = tSig * exp(1j * Omega * L / c) * getProdTF(tau_SEC_ARM, np.array([[np.zeros(nf)], [sqrt(2 * K) / h_SQL]]))

    # dark-port input field
    Mifo = rho_SEC

    # loss field from arm cavity
    Mn = lossArm * tau_SEC_ARM

    # loss field from signal-extraction cavity
    Mp = lossSR * tau_SEC

    # adapt to GWINC phase convention
    Msig = Msig[[1, 0], :, :]
    Msig[1, 0, :] = -Msig[1, 0, :]

    def adapt_to_gwinc(Mx):
        My = np.zeros(Mx.shape, dtype=complex)
        My[0, 0, :] = Mx[1, 1, :]
        My[1, 1, :] = Mx[0, 0, :]
        My[0, 1, :] = -Mx[1, 0, :]
        My[1, 0, :] = -Mx[0, 1, :]
        return My

    Mifo = adapt_to_gwinc(Mifo)
    Mn = adapt_to_gwinc(Mn)
    Mp = adapt_to_gwinc(Mp)

    # overall coefficient
    coeff = 1

    # put all loss fields together
    Mnoise = np.hstack([Mn, Mp])

    return coeff, Mifo, Msig, Mnoise


def shotradSignalRecycledBnC(f, ifo, power):
    """Quantum noise model for signal recycled IFO

    See shotrad for more info.

    All references to Buonanno & Chen PRD 64 042006 (2001) (hereafter BnC)
    Updated to include losses DEC 2006 Kirk McKenzie using BnC notation
    Updated to include squeezing April 2009 KM
    Updated April 2010 KM, LB

    moved out of shotrad May 2010, mevans
    output is used in shotrad to compute final noise as
      n = coeff * (vHD * Msqz * Msqz' * vHD') / (vHD * Md * Md' * vHD')
    where
      Msqz = [Mc MsqueezeInput, Mn]

    coeff = frequency dependent overall noise coefficient (Nx1)
    Mifo = IFO input-output relation for the AS port
    Msig = signal transfer to the AS port
    Mnoise = noise fields produced by losses in the IFO at the AS port

    """
    lambda_ = ifo.Laser.Wavelength               # Laser Wavelength [m]
    hbar = const.hbar                            # Plancks Constant [Js]
    c = const.c                                  # SOL [m/s]
    Omega = 2*pi*f                               # [BnC, table 1] Signal angular frequency [rads/s]
    omega_0 = 2*pi*c/lambda_                     # [BnC, table 1] Laser angular frequency [rads/s]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    L = ifo.Infrastructure.Length                # Length of arm cavities [m]
    l = ifo.Optics.SRM.CavityLength              # SRC Length [m]
    T = ifo.Optics.ITM.Transmittance             # ITM Transmittance [Power]
    m = ifo.Suspension.Stage[0].Mass             # Mirror mass [kg]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    bsloss = ifo.Optics.BSLoss                   # BS Loss [Power]
    mismatch = 1 - ifo.Optics.coupling           # Mismatch
    mismatch = mismatch + ifo.TCS.SRCloss        # Mismatch

    # BSloss + mismatch has been incorporated into a SRC Loss
    lambda_SR = mismatch + bsloss                # SR cavity loss [Power]

    tau = sqrt(ifo.Optics.SRM.Transmittance)     # SRM Transmittance [amplitude]
    rho = sqrt(1 - tau**2 - lambda_SR)           # SRM Reflectivity [amplitude]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    ds = ifo.Optics.SRM.Tunephase                # SRC Detunning
    phi = (pi-ds)/2                              # [BnC, between 2.14 & 2.15] SR Detuning

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    lambda_arm = ifo.Optics.Loss*2               # [BnC, after 5.2] Round Trip loss in arm [Power]
    gamma_ac = T*c/(4*L)                         # [KLMTV-PRD2001] Arm cavity half bandwidth [1/s]
    epsilon = lambda_arm/(2*gamma_ac*L/c)        # [BnC, after 5.2] Loss coefficent for arm cavity

    I_0 = power.pbs                              # [BnC, Table 1] Power at BS (Power*prfactor) [W]
    I_SQL = (m*L**2*gamma_ac**4)/(4*omega_0)     # [BnC, 2.14] Power to reach free mass SQL
    Kappa = 2*((I_0/I_SQL)*gamma_ac**4)/ \
              (Omega**2*(gamma_ac**2+Omega**2))  # [BnC 2.13] Effective Radiation Pressure Coupling
    beta = arctan(Omega/gamma_ac)                # [BnC, after 2.11] Phase shift of GW SB in arm
    h_SQL = sqrt(8*hbar/(m*(Omega*L)**2))        # [BnC, 2.12] SQL Strain


    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Coefficients [BnC, Equations 5.8 to 5.12]
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    exp_1jbeta = exp(1j*beta)
    cos_beta = exp_1jbeta.real
    invexp_1jbeta = 1/exp_1jbeta
    exp_2jbeta = exp_1jbeta**2
    cos_2beta = exp_2jbeta.real
    invexp_2jbeta = 1/exp_2jbeta
    exp_4jbeta = exp_2jbeta**2
    C11_L = ( (1+rho**2) * ( cos(2*phi) + Kappa/2 * sin(2*phi) ) -
              2*rho*cos_2beta - 1/4*epsilon * ( -2 * (1+exp_2jbeta)**2 * rho + 4 * (1+rho**2) *
                                                cos_beta**2*cos(2*phi) + ( 3+exp_2jbeta ) *
                                                Kappa * (1+rho**2) * sin(2*phi) ) +
              lambda_SR * ( exp_2jbeta*rho-1/2 * (1+rho**2) * ( cos(2*phi)+Kappa/2 * sin(2*phi) ) ) )

    C22_L = C11_L

    C12_L = tau**2 * ( - ( sin(2*phi) + Kappa*sin(phi)**2 )+
                       1/2*epsilon*sin(phi) * ( (3+exp_2jbeta) * Kappa * sin(phi) + 4*cos_beta**2 * cos(phi)) +
                       1/2*lambda_SR * ( sin(2*phi)+Kappa*sin(phi)**2) )

    C21_L = tau**2 * ( (sin(2*phi)-Kappa*cos(phi)**2 ) +
                       1/2*epsilon*cos(phi) * ( (3+exp_2jbeta )*Kappa*cos(phi) - 4*cos_beta**2*sin(phi) ) +
                       1/2*lambda_SR * ( -sin(2*phi) + Kappa*cos(phi)**2) )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    D1_L = ( - (1+rho*exp_2jbeta ) * sin(phi) +
             1/4*epsilon * ( 3+rho+2*rho*exp_4jbeta + exp_2jbeta*(1+5*rho) ) * sin(phi)+
             1/2*lambda_SR * exp_2jbeta * rho * sin(phi) )

    D2_L = ( - (-1+rho*exp_2jbeta ) * cos(phi) +
             1/4*epsilon * ( -3+rho+2*rho*exp_4jbeta + exp_2jbeta * (-1+5*rho) ) * cos(phi)+
             1/2*lambda_SR * exp_2jbeta * rho * cos(phi) )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    P11 = 0.5 * sqrt(lambda_SR) * tau * \
          ( -2*rho*exp_2jbeta+2*cos(2*phi)+Kappa*sin(2*phi) )
    P22 = P11
    P12 = -sqrt(lambda_SR)*tau*sin(phi)*(2*cos(phi)+Kappa*sin(phi) )
    P21 =  sqrt(lambda_SR)*tau*cos(phi)*(2*sin(phi)-Kappa*cos(phi) )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # this was the PD noise source, but that belongs outside of this function
    #   I have used the equation for Q11 to properly normalize the other noises
    #   as well as the input-output relation Mc and the signal matrix Md

    Q11 = 1 / \
        ( invexp_2jbeta+rho**2*exp_2jbeta-rho*(2*cos(2*phi)+Kappa*sin(2*phi)) +
          1/2*epsilon*rho * (invexp_2jbeta*cos(2*phi)+exp_2jbeta*
                             ( -2*rho-2*rho*cos_2beta+cos(2*phi)+Kappa*sin(2*phi) ) +
                             2*cos(2*phi)+3*Kappa*sin(2*phi))-1/2*lambda_SR*rho *
          ( 2*rho*exp_2jbeta-2*cos(2*phi)-Kappa*sin(2*phi) ) )
    Q22 = Q11
    Q12 = 0
    Q21 = 0

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    N11 = sqrt(epsilon/2)*tau *(Kappa*(1+rho*exp_2jbeta)*sin(phi)+
                                2*cos_beta*(invexp_1jbeta*cos(phi)-rho*exp_1jbeta*(cos(phi)+Kappa*sin(phi))))
    N22 = -sqrt(2*epsilon)*tau*(-invexp_1jbeta+rho*exp_1jbeta)*cos_beta*cos(phi)
    N12 = -sqrt(2*epsilon)*tau*(invexp_1jbeta+rho*exp_1jbeta)*cos_beta*sin(phi);
    N21 = sqrt(epsilon/2)*tau*(-Kappa*(1+rho)*cos(phi)+
                               2*cos_beta*(invexp_1jbeta+rho*exp_1jbeta)*cos_beta*sin(phi))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # overall coefficient
    coeff = h_SQL**2/(2*Kappa*tau**2)

    # make normalization matrix
    Mq = make2x2TF(Q11, Q12, Q21, Q22)

    # 3D transfer matrices from vectors for each element
    Mifo = getProdTF(Mq, make2x2TF(C11_L, C12_L, C21_L, C22_L))
    Msig = getProdTF(Mq, np.array([D1_L, D2_L]).reshape(2, 1, f.size))

    # put all output noises together
    Mp = make2x2TF(P11, P12, P21, P22)
    Mn = make2x2TF(N11, N12, N21, N22)
    Mnoise = getProdTF(Mq, np.hstack((Mn, Mp)))

    return coeff, Mifo, Msig, Mnoise


def propagate_noise_fc_ifo(f, ifo, Mifo, Mnoise, sqz_params, alpha):
    """Propagate quantum noise through the filter cavities and IFO

    f: frequency vector [Hz]
    ifo: ifo Struct
    Mifo: IFO input-output relation for the AS port
    Mnoise: dictionary of existing noise sources
    sqz_params: Struct of squeeze parameters
    alpha: squeeze quadrature

    Returns a new Mnoise dict with the noises due to AS port vacuum, injection
    loss, and any input or output filter cavities added.
    """
    Mnoise = deepcopy(Mnoise)

    #>>>>>>>>    QUANTUM NOISE POWER SPECTRAL DENSITY WITH SQZ [BnC PRD 2004, 62]
    #<<<<<<<<<<<<<<<<< Modified to include losses (KM)
    #<<<<<<<<<<<<<<<<< Modified to include frequency dependent squeezing angle (LB)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ------------------------------------------- equation 63 BnC PRD 2004
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Define input matrix of Squeezing
    R = sqz_params.SQZ_DB / (20 * log10(exp(1)))           # squeeze factor
    R_anti = sqz_params.ANTISQZ_DB / (20 * log10(exp(1)))  # anti-squeeze factor
    Msqz = np.array([[exp(-R), 0], [0, exp(R_anti)]])

    # expand to Nfreq
    Msqz = np.transpose(np.tile(Msqz, (len(f), 1, 1)), axes=(1, 2, 0))

    # add input rotation
    MsqzRot = make2x2TF(cos(alpha), -sin(alpha), sin(alpha), cos(alpha))
    Msqz = getProdTF(MsqzRot, Msqz)
    Msqz = dict(ASvac=Msqz)

    # Include losses (lambda_in=ifo.Squeezer.InjectionLoss)
    Msqz = sqzInjectionLoss(Msqz, sqz_params.lambda_in, 'injection')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Inject squeezed field into the IFO via some filter cavities
    if sqz_params.sqzType == 'Freq Dependent' and 'FilterCavity' in ifo.Squeezer:
        if not isinstance(ifo.Squeezer.FilterCavity, Sequence):
            fc_list = [ifo.Squeezer.FilterCavity]
        else:
            fc_list = ifo.Squeezer.FilterCavity
        logger.debug('  Applying %d input filter cavities' % len(fc_list))
        Mr, Msqz = sqzFilterCavityChain(
            f, fc_list, Msqz)

    # apply the IFO dependent squeezing matrix to get the total noise
    # due to quantum fluctuations coming in from the AS port
    Msqz = {key: getProdTF(Mifo, nn) for key, nn in Msqz.items()}

    # add this to the other noises
    Mnoise.update(Msqz)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # pass IFO output through some filter cavities
    if 'OutputFilter' in ifo:
        if ifo.OutputFilter.Type == 'None':
            # do nothing, say nothing
            pass

        elif ifo.OutputFilter.Type == 'Chain':
            logger.debug('  Applying %d output filter cavities' % np.atleast_1d(ifo.OutputFilter.FilterCavity).size)

            Mr, Mnoise = sqzFilterCavityChain(
                f, np.atleast_1d(ifo.OutputFilter.FilterCavity), Mnoise, key='OFC')
            Msig = getProdTF(Mr, Msig)
            #  Mnoise = getProdTF(Mn, Mnoise);

        elif ifo.OutputFilter.Type == 'Optimal':
            raise NotImplementedError("Cannot do optimal phase yet")
            logger.debug('  Optimal output filtering!')

            # compute optimal angle, including upcoming PD losses
            MnPD = sqzInjectionLoss(Mnoise, lambda_PD)
            zeta = sqzOptimalReadoutPhase(Msig, MnPD)

            # rotate by that angle, less the homodyne angle
            #zeta_opt = eta;
            cs = cos(zeta - eta)
            sn = sin(zeta - eta)
            Mrot = make2x2TF(cs, -sn, sn, cs)
            Mnoise = getProdTF(Mrot, Mnoise)
            Msig = getProdTF(Mrot, Msig)

        else:
            raise Exception('ifo.OutputFilter.Type must be None, Chain or Optimal, not "%s"' % ifo.OutputFilter.Type)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # add PD efficiency
    lambda_PD = 1 - ifo.Optics.PhotoDetectorEfficiency  # PD losses
    Mnoise = sqzInjectionLoss(Mnoise, lambda_PD, 'pd')

    return Mnoise


######################################################################
# Auxiliary functions
######################################################################

def make2x2TF(A11, A12, A21, A22):
    """Create transfer matrix with 2x2xnF.

    """
    A11, A12, A21, A22 = np.broadcast_arrays(A11, A12, A21, A22)
    M3 = np.array([[A11, A12], [A21, A22]])
    return M3.reshape(2, 2, -1)


def getProdTF(lhs, rhs):
    """Compute the product of M Nout x Nin x Naf frequency dependent transfer matrices

    See also getTF.

    NOTE: To perform more complicated operations on transfer
          matrices, see LTI object FRD ("help frd").  This
          function is the same as: freqresp(frd(lhs) * frd(rhs), f)

    """
    # check matrix size
    if lhs.shape[1] != rhs.shape[0]:
        raise Exception('Matrix size mismatch size(lhs, 2) = %d != %d = size(rhs, 1)' % (lhs.shape[1], rhs.shape[0]))
    if len(lhs.shape) == 3:
        lhs = np.transpose(lhs, axes=(2, 0, 1))
    if len(rhs.shape) == 3:
        rhs = np.transpose(rhs, axes=(2, 0, 1))

    # compute product
    if len(lhs.shape) < 3 or lhs.shape[0] == 1:
        rslt = np.matmul(lhs, rhs)
    elif len(rhs.shape) < 3 or rhs.shape[0] == 1:
        rslt = np.matmul(lhs, rhs)
    elif lhs.shape[0] == rhs.shape[0]:
        rslt = np.matmul(lhs, rhs)
    else:
        raise Exception('Matrix size mismatch lhs.shape[2] = %d != %d = rhs.shape[2]' % (lhs.shape[2], rhs.shape[2]))

    if len(rslt.shape) == 3:
        rslt = np.transpose(rslt, axes=(1, 2, 0))
    return rslt


def sqzInjectionLoss(Min, L, key):
    """Injection losses for squeezed field

    Calculates noise where unsqueezed vacuum is injected at a source of loss
    and adds this to a dictionary of existing noise sources

    Min = dictionary of existing sources
    L = the loss
    key = key for the new noise source
    Mout = updated dictionary with the original noises reduced by sqrt(1 - L)
           and a new item for the new noise source
    """
    # number of existing noise fields
    num_noise_fld = sum([nn.shape[1] for nn in Min.values()])
    # number of frequency points
    npts = Min[list(Min.keys())[0]].shape[2]
    # each of the existing noise sources should be reduced by sqrt(1 - L)
    Mout = {nkey: nn * sqrt(1 - L) for nkey, nn in Min.items()}

    # add new noise fields
    eye2 = np.eye(2, num_noise_fld)
    Meye = np.transpose(np.tile(eye2, (npts, 1, 1)), axes=(1, 2, 0))
    Mout[key] = Meye * sqrt(L)

    return Mout


def sqzFilterCavityChain(f, params, Mn, key='FC'):
    """Transfer relation for a chain of filter cavities

    Noise added by cavity losses are also output.

    f = frequency vector [Hz]
    param.fdetune = detuning [Hz]
    param.L = cavity length
    param.Ti = input mirror trasmission [Power]
    param.Li = input mirror loss
    param.Te = end mirror trasmission
    param.Le = end mirror loss
    param.Rot = phase rotation after cavity

    Mn0 = input noise
    Mc = input to output transfer
    Mn = filtered input noise, plus noise due to cavity losses
    key = key to use for new entries to the noise dictionary

    Note:
        [Mc, Mn] = sqzFilterCavityChain(f, params, Mn0)
      is the same as
        [Mc, Mn] = sqzFilterCavityChain(f, params);
        Mn = [getProdTF(Mc, Mn0), Mn];

    corresponding author: mevans

    """
    # make an identity TF
    Mc = make2x2TF(np.ones(f.shape), 0, 0, 1)

    # loop through the filter cavites
    for k, fc in enumerate(params):
        # extract parameters for this filter cavity
        Lf = fc.L
        fdetune = fc.fdetune
        Ti = fc.Ti
        Te = fc.Te
        Lrt = fc.Lrt
        theta = fc.Rot

        # compute new Mn
        fc_key = key + str(k)
        Mr, Mt, Mn = sqzFilterCavity(f, Lf, Ti, Te, Lrt, fdetune, Mn, key=fc_key)

        # apply phase rotation after filter cavity
        Mrot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        Mn = {key: getProdTF(Mrot, nn) for key, nn in Mn.items()}

        # update Mc
        Mc = getProdTF(Mrot, getProdTF(Mr, Mc))

    return Mc, Mn


def sqzFilterCavity(f, Lcav, Ti, Te, Lrt, fdetune, MinR, MinT=1, key='FC'):
    """Reflection/transmission matrix for filter cavity

    Function which gives the reflection matrix for vacuum fluctuations
    entering the input mirror and the transmission matrix for vacuum
    fluctuations entering the end mirror of one filter cavity.  The
    input parameters are the cavity parameters and the 2X2 matrix of
    the incoming fields in the two-photon formalism.

    (R_alpha x S_r) for a freq independent squeezed field.
    f = vector frequency in Hz
    Lf = length of the filter cavity
    Ti = transmission and losses of the input mirror
    Te = transmission and losses of the end mirror
    Lrt = round-trip losses in the cavity (mirror transmissoins not included)
    fdetune: detuning frequency of the filter cavity [Hz]
    MinR: squeezed field injected from the input mirror of the filter cavity (a1,a2 basis)
         if this argument is empty, it is assumed that the user will use Mr,
         so no noise field is added to Mnoise.  If no argument is given, or
         the scalar 1 is given, an Mr unsqueezed input is assumed and Mr is
         concatenated into Mnoise.
    MinT: squeezed field injected from the back of the filter cavity
         with MinR, this argument can be omitted or set to 1 to indicate
         and unsqueezed input. [] can be used to avoid adding a noise
         term to Mnoise.
    key: key to use for new entries to the noise dictionary
         the losses from the filter cavity are key and the additional losses
         injected from the back mirror are key_unsqzd_back

    corresponding authors: LisaB, mevans

    """

    # reflectivities
    Ri = 1 - Ti
    Re = 1 - Te

    ri = sqrt(Ri)
    re = sqrt(Re)
    rr = ri * re * sqrt(1 - Lrt)  # include round-trip losses

    # Phases for positive and negative audio sidebands
    c = const.c
    omega = 2 * pi * f
    wf = 2 * pi * fdetune
    Phi_p = 2 * (omega-wf) * Lcav / c
    Phi_m = 2 * (-omega-wf) * Lcav / c

    ephi_p = exp(1j * Phi_p)
    ephi_m = exp(1j * Phi_m)

    # cavity gains
    g_p = 1 / (1 - rr * ephi_p)
    g_m = 1 / (1 - rr * ephi_m)

    # Reflectivity for vacuum flactuation entering the cavity from
    # the input mirror (check sign)
    r_p = ri - re * Ti * ephi_p * g_p
    r_m = ri - re * Ti * ephi_m * g_m

    # Transmissivity for vacuum flactuation entering the cavity from
    # the back mirror (check sign)
    t_p = sqrt(Ti * Te * ephi_p) * g_p
    t_m = sqrt(Ti * Te * ephi_m) * g_m

    # Transmissivity for vacuum flactuation entering the cavity from
    # the losses in the cavity
    l_p = sqrt(Ti * Lrt * ephi_p) * g_p
    l_m = sqrt(Ti * Lrt * ephi_m) * g_m

    # Relfection matrix for vacuum fluctuations entering from the input
    # mirror in the A+, (a-)* basis
    Mr_temp = make2x2TF(r_p, 0, 0, conj(r_m))

    # Transmission matrix for vacuum fluctuations entering from the end mirror
    Mt_temp = make2x2TF(t_p, 0, 0, conj(t_m))

    # Transmission matrix for vacuum fluctuations entering from the cavity losses
    Ml_temp = make2x2TF(l_p, 0, 0, conj(l_m))

    # Apply matrix which changes from two-photon basis to a+ and (a-)*
    Mbasis = np.array([[1, 1j], [1, -1j]])

    Mr = getProdTF(np.linalg.inv(Mbasis), getProdTF(Mr_temp, Mbasis))
    Mt = getProdTF(np.linalg.inv(Mbasis), getProdTF(Mt_temp, Mbasis))
    Ml = getProdTF(np.linalg.inv(Mbasis), getProdTF(Ml_temp, Mbasis))

    ###### output

    # reflected fields
    Mnoise = {}
    for nkey, nn in MinR.items():
        if np.isscalar(nn):
            Mnoise[nkey] = Mr * nn
        else:
            Mnoise[nkey] = getProdTF(Mr, nn)

    # transmitted fields
    if MinT != {} and Te > 0:
        if np.isscalar(MinT) and MinT == 1:
            # inject unsqueezed vacuum at the back
            Mnoise[key + '_unsqzd_back'] = Mt
        else:
            # inject a given noise at the back
            for nkey, nn in MinT.items():
                if np.isscalar(nn):
                    Mnoise[nkey] = Mt * nn
                else:
                    Mnoise[nkey] = getProdTF(Mt, nn)
    elif MinT != {} and Te == 0:
        Mnoise[key + '_unsqzd_back'] = np.zeros_like(Mt)

    # loss fields
    if Lrt > 0:
        Mnoise[key] = Ml

    return Mr, Mt, Mnoise
