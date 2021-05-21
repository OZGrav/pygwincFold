'''Functions to calculate Newtonian noise

'''
from __future__ import division
import numpy as np
from numpy import pi, sqrt, exp, log10
import scipy.integrate as scint
import scipy.special as sp

from .seismic import seismic_ground_NLNM
from .. import const


def gravg(f, seismic):
    """Gravity gradient noise for single test mass

    :f: frequency array in Hz
    :seismic: gwinc Seismic struct

    :returns: displacement noise power spectrum at :f:, in meters

    References:

     Saulson 1984,           http://dx.doi.org/10.1103/PhysRevD.30.732
     Hughes and Thorne 1998, http://dx.doi.org/10.1103/PhysRevD.58.122002

     Driggers and Harms 2011, ``Results of Phase 1 Newtonian Noise
     Measurements at the LIGO Sites,'' February-March 2011.  T1100237.
     https://dcc.ligo.org/LIGO-T1100237

    Written by Enrico Camagna (?)

    added to Bench by Gregg Harry 8/27/03
    seismic spectrum modified by Jan Harms 05/11/2010
    Calculates gravity gradient noise for four mirrors

    """

    fk = seismic.KneeFrequency
    a = seismic.LowFrequencyLevel
    gamma = seismic.Gamma
    rho = seismic.Rho
    # factor to account for correlation between masses
    # and the height of the mirror above the ground
    beta = seismic.Beta
    h = seismic.TestMassHeight
    c_rayleigh = seismic.RayleighWaveSpeed

    if 'Omicron' in seismic:
        omicron = seismic.Omicron
    else:
        omicron = 1

    # a sort of theta function (Fermi distr.)
    coeff = 3**(-gamma*f)/(3**(-gamma*f) + 3**(-gamma*fk))

    # modelization of seismic noise (vertical)
    ground = a*coeff + a*(1-coeff)*(fk/f)**2
    if 'Site' in seismic and seismic.Site == 'LLO':
        ground = a*coeff*(fk/f) + a*(1-coeff)*(fk/f)**2

    # effective GG spring frequency, with G gravitational
    fgg = sqrt(const.G * rho) / (2*pi)

    # fixed numerical factors, 5/9/06, PF
    n = (beta*2*pi*(fgg**2/f**2)*ground)**2

    # The following two terms are corrections due to Jan Harms
    # https://git.ligo.org/rana-adhikari/CryogenicLIGO/issues/45
    # (1) projection of NN force onto the direction of the arm
    n = n * 1/2
    # (2) exponential cutoff at frequency (seismic speed)/(test mass height)
    n = n * exp(-4*pi*f*h/c_rayleigh)

    # Feedforward cancellation
    n /= (omicron**2)

    return n


def gravg_rayleigh(f, seismic):
    """Gravity gradient noise for single arm cavity from seismic Rayleigh waves

    :f: frequency array in Hz
    :seismic: gwinc Seismic structure

    :returns: displacement noise power spectrum at :f:, in meters

    Following Harms LRR: https://doi.org/10.1007/lrr-2015-3
    and Amann et al.: https://doi.org/10.1063/5.0018414

    """
    fk = seismic.KneeFrequency
    a = seismic.LowFrequencyLevel
    gamma = seismic.Gamma
    rho = seismic.Rho
    h = seismic.TestMassHeight
    c_rayleigh = seismic.RayleighWaveSpeed

    if 'Omicron' in seismic:
        omicron = seismic.Omicron
    else:
        omicron = 1

    # a sort of theta function (Fermi distr.)
    coeff = 3**(-gamma*f)/(3**(-gamma*f) + 3**(-gamma*fk))

    # modelization of seismic noise (vertical)
    ground = a*coeff + a*(1-coeff)*(fk/f)**2
    if 'Site' in seismic and seismic.Site == 'LLO':
        ground = a*coeff*(fk/f) + a*(1-coeff)*(fk/f)**2

    # Harms LRR eqs. 35, 96, and 98
    w = 2 * pi * f
    k = w / c_rayleigh
    kP = w / seismic.pWaveSpeed
    kS = w / seismic.sWaveSpeed
    qzP = sqrt(k**2 - kP**2)
    qzS = sqrt(k**2 - kS**2)
    zeta = sqrt(qzP / qzS)

    gnu = k * (1 - zeta) / (qzP - k * zeta)
    
    if h >= 0:
        # Harms LRR
        n = (2 * pi * const.G * rho * exp(-h * k) * gnu)**2 * ground**2 / w**4

    else:
        # Amann et al., eqs. 2-6. Note h is there defined as depth;
        # We define it as height.
        r0 = k * (1 - zeta)
        sh = -k * (1 + zeta) * exp(h * k)
        bh = (2 / 3) * (2 * k * exp(h * qzP) + zeta * qzS * exp(h * qzS))
        Rcal = np.abs((sh + bh) / r0)**2
        n = 4 * (sqrt(2) * pi * const.G * rho * gnu)**2 * Rcal * ground**2 / w**4

    n /= omicron**2

    return n


def gravg_pwave(f, seismic, exact=False):
    """Gravity gradient noise for single test mass from seismic p-waves

    :f: frequency array in Hz
    :seismic: gwinc Seismic structure
    :exact: whether to use a slower numerical integral good to higher frequencies
            or faster special functions. If exact=False, nan is returned at
            high frequencies where the special functions have numerical errors.

    :returns: displacement noise power spectrum at :f:, in meters

    Following Harms LRR: https://doi.org/10.1007/lrr-2015-3

    """
    cP = seismic.pWaveSpeed
    levelP = seismic.pWaveLevel
    tmheight = seismic.TestMassHeight
    rho_ground = seismic.Rho

    kP = (2 * pi * f) / cP

    psd_ground_pwave = (levelP * seismic_ground_NLNM(f))**2

    xP = np.abs(kP * tmheight)

    if tmheight >= 0:
        # Surface facility
        # The P-S conversion at the surface is not implemented
        if exact:
            height_supp_power = (3 / 2) * np.array(
                [scint.quad(lambda th, x: np.sin(th)**3
                            * np.exp(-2 * x * np.sin(th)), 0, pi / 2, args=(x,))[0]
                 for x in xP])

        else:
            xP[xP > 10] = np.nan
            height_supp_power = 3 / (24*xP) * (
                8*xP - 9*pi*sp.iv(2, 2*xP) - 6*pi*xP*sp.iv(3, 2*xP)
                + 6*pi*xP*sp.modstruve(1, 2*xP) - 3*pi*sp.modstruve(2, 2*xP))

    else:
        # Underground facility
        # The cavity effect is not included
        if exact:
            height_supp_power = (3 / 4) * np.array(
                [scint.quad(lambda th, x: np.sin(th)**3
                            * (2 - np.exp(-x * np.sin(th)))**2, 0, pi, args=(x,))[0]
                 for x in xP])

        else:
            xP[xP > 2] = np.nan
            height_supp_power = 1 + 3*pi/(8*xP) * (
                24*sp.iv(2, xP) - 3*sp.iv(2, 2*xP) + 8*xP*sp.iv(3, xP)
                - 2*xP*sp.iv(3, 2*xP) - 8*xP*sp.modstruve(1, xP)
                + 2*xP*sp.modstruve(1, 2*xP) - 8*sp.modstruve(2, xP)
                + sp.modstruve(2, 2*xP))

    psd_gravg_pwave = ((2 * pi * const.G * rho_ground)**2
            * psd_ground_pwave * height_supp_power)
    psd_gravg_pwave /= (2 * pi * f)**4
    return psd_gravg_pwave


def gravg_swave(f, seismic, exact=False):
    """Gravity gradient noise for single test mass from seismic s-waves

    :f: frequency array in Hz
    :seismic: gwinc Seismic structure
    :exact: whether to use a slower numerical integral good to higher frequencies
            or faster special functions. If exact=False, nan is returned at
            high frequencies where the special functions have numerical errors.

    :returns: displacement noise power spectrum at :f:, in meters

    Following Harms LRR: https://doi.org/10.1007/lrr-2015-3

    """
    cS = seismic.sWaveSpeed
    levelS = seismic.sWaveLevel
    tmheight = seismic.TestMassHeight
    rho_ground = seismic.Rho

    kS = (2 * pi * f) / cS

    psd_ground_swave = (levelS * seismic_ground_NLNM(f))**2

    xS = np.abs(kS * tmheight)

    # For both surface and underground facilities
    if exact:
        height_supp_power = (3 / 2) * np.array(
            [scint.quad(lambda th, x: np.sin(th)**3
                        * np.exp(-2 * x * np.sin(th)), 0, pi / 2, args=(x,))[0]
             for x in xS])

    else:
        xS[xS > 10] = np.nan
        height_supp_power = 3 / (24*xS) * (
            8*xS - 9*pi*sp.iv(2, 2*xS) - 6*pi*xS*sp.iv(3, 2*xS)
            + 6*pi*xS*sp.modstruve(1, 2*xS) - 3*pi*sp.modstruve(2, 2*xS))

    psd_gravg_swave = ((2 * pi * const.G * rho_ground)**2
            * psd_ground_swave * height_supp_power)
    psd_gravg_swave /= (2 * pi * f)**4

    return psd_gravg_swave


def atmois(f, atmos, seismic):
    """Atmospheric infrasound newtonian noise for single arm cavity

    :f: frequency array in Hz
    :atmos: gwinc Atmospheric structure
    :seismic: gwinc Seismic structure

    :returns: displacement noise power spectrum at :f:, in meters

    """
    p_air = atmos.AirPressure
    rho_air = atmos.AirDensity
    ai_air = atmos.AdiabaticIndex
    c_sound = atmos.SoundSpeed
    h = seismic.TestMassHeight

    w = 2 * pi * f
    k = w / c_sound

    # Pressure spectrum
    try:
        a_if = atmos.InfrasoundLevel1Hz
        e_if = atmos.InfrasoundExponent
        psd_if = (a_if * f**e_if)**2
    except AttributeError:
        psd_if = atmoBowman(f)**2

    # Harms LRR (2015), eq. 172
    # https://doi.org/10.1007/lrr-2015-3
    # And with the Bessel terms ignored... for 4 km this amounts to a 10%
    # correction at 10 Hz and a 30% correction at 1 Hz
    coupling_if = 2/3 * (4 * pi / (k * w**2) * const.G * rho_air / (ai_air * p_air))**2

    n_if = coupling_if * psd_if

    return n_if


def atmoBowman(f):
    """The Bowman infrasound model
   
    """ 
    freq = np.array([
        0.01, 0.0155, 0.0239, 0.0367, 0.0567,
        0.0874, 0.1345, 0.2075, 0.32, 0.5,
        0.76, 1.17, 1.8, 2.79, 4.3,
        6.64, 10, 100,
    ])
    pressure_asd = sqrt([
        22.8, 4, 0.7, 0.14, 0.027, 0.004,
        0.0029, 0.0039, 7e-4, 1.44e-4, 0.37e-4,
        0.12e-4, 0.56e-5, 0.35e-5, 0.26e-5, 0.24e-5,
        2e-6, 2e-6,
    ])
    return 10**(np.interp(log10(f), log10(freq), log10(pressure_asd)))
