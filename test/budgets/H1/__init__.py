"""
Simplified ifo based off of H1's budget for testing
"""
import numpy as np
from gwinc import nb
import scipy.constants as scc
import gwinc.noise as noise
from os import path
from gwinc import Struct


class Shot(nb.Noise):
    style = dict(label='Shot noise')

    def calc(self):
        dcpd_mA = self.ifo.dcpd_mA
        shot_mA_rtHz = np.ones_like(self.freq) * np.sqrt(2 * scc.e * dcpd_mA * 1e-3) * 1e3
        sqzV = 10**(self.ifo.sqz_dB / 10)
        return shot_mA_rtHz**2 * sqzV


class RadiationPressure(nb.Noise):
    style = dict(label='Radiation pressure')

    def calc(self):
        arm_len = self.ifo.arm_len
        arm_trans = self.ifo.arm_trans
        srm_trans = self.ifo.srm_trans
        mass = self.ifo.mass
        wavelen = self.ifo.wavelen
        c_light = scc.c
        h_bar = scc.hbar
        omega = 2 * np.pi * self.freq

        prc_power = self.ifo.prc_power
        asqzV = 10**(self.ifo.asqz_dB / 10)

        arm_len = self.ifo.arm_len
        arm_trans = self.ifo.arm_trans
        srm_trans = self.ifo.srm_trans
        mass = self.ifo.mass
        wavelen = self.ifo.wavelen
        c_light = scc.c
        h_bar = scc.hbar
        omega = 2 * np.pi * self.freq

        prc_power = self.ifo.prc_power
        asqzV = 10**(self.ifo.asqz_dB / 10)

        gamma = arm_trans * c_light / (4 * arm_len)
        beta = np.arctan(omega / gamma)
        rho = (1 - srm_trans)**.5
        tau = (srm_trans)**.5
        Isql = mass * arm_len**2 * gamma**4 / (4 * 2 * np.pi * c_light / wavelen)
        kappa = 2 * (prc_power / Isql) * gamma**4 / (omega**2 * (gamma**2 + omega**2))
        kappa_b = kappa * tau**2 / (1 + rho**2 + 2 * rho * np.cos(2*beta))
        Ssql = 8 * h_bar / (mass * omega**2)
        return (Ssql / 2) * kappa_b * asqzV


class Coating(nb.Budget):
    """Coating Thermal

    """

    name = 'Coating'

    style = dict(
        label='Coating Thermal',
        color='#fe0002',
    )

    noises = [
        noise.coatingthermal.CoatingBrownian,
        noise.coatingthermal.CoatingThermoOptic,
    ]


class Substrate(nb.Budget):
    """Substrate Thermal

    """

    name = 'Substrate'

    style = dict(
        label='Substrate Thermal',
        color='#fb7d07',
    )

    noises = [
        noise.substratethermal.SubstrateBrownian,
        noise.substratethermal.SubstrateThermoElastic,
    ]


class Thermal(nb.Budget):
    noises = [
        Coating,
        Substrate,
        noise.suspensionthermal.SuspensionThermal,
    ]


class ThermalDict(nb.Budget):
    noises = {
        'Coating': Coating,
        'Substrate': Substrate,
        'Sus': noise.suspensionthermal.SuspensionThermal,
    }


class Sensing(nb.Calibration):
    """
    Simple single pole sensing function
    """
    def calc(self):
        sensing_mA_m = 5.15e12 / (1 + 1j * self.freq / 445)
        return 1 / np.abs(sensing_mA_m)**2


class SensingOpticalSpring(nb.Calibration):
    """
    Sensing function with an optical spring
    """
    def calc(self):
        darm_pole_Hz = 445
        spring_Hz = 10
        springQ = 30
        sensing_mA_m = 5.15e12 / (1 + 1j * self.freq / darm_pole_Hz)
        den = self.freq**2 + spring_Hz**2 - 1j * self.freq * spring_Hz / springQ
        sensing_mA_m *= self.freq**2 / den
        return 1 / np.abs(sensing_mA_m)**2


class Quantum(nb.Budget):
    noises = [
        (Shot, Sensing),
        RadiationPressure,
    ]


class QuantumDict(nb.Budget):
    noises = {
        'Shot': (Shot, Sensing),
        'RadiationPressure': RadiationPressure,
    }


class DARMMeasured(nb.Noise):
    style = dict(label='H1 reference')

    def load(self):
        bpath = self.load.__code__.co_filename
        fname = path.join(path.split(bpath)[0], 'lho_ref.txt')
        F_Hz, darm_m_rtHz = np.loadtxt(fname).T
        self.data = self.interpolate(F_Hz, darm_m_rtHz**2)

    def calc(self):
        return self.data


class DARMMeasuredO3(nb.Noise):
    style = dict(label='H1 O3')

    def load(self):
        bpath = self.load.__code__.co_filename
        fname = path.join(path.split(bpath)[0], 'lho_o3.txt')
        F_Hz, darm_m_rtHz = np.loadtxt(fname).T
        self.data = self.interpolate(F_Hz, darm_m_rtHz**2)

    def calc(self):
        return self.data


class H1(nb.Budget):
    noises = [
        (Shot, Sensing),
        RadiationPressure,
        Thermal,
        noise.seismic.Seismic,
    ]

    references = [
        DARMMeasured,
        DARMMeasuredO3,
    ]


class H1NoRefs(nb.Budget):
    noises = [
        (Shot, Sensing),
        RadiationPressure,
        Thermal,
        noise.seismic.Seismic,
    ]


class H1NoRefsForwardNoises(nb.Budget):
    noises = [
        noise.seismic.Seismic,
    ]

    noises_forward = [
        Quantum,
        Thermal,
    ]


class H1dict(nb.Budget):
    noises = {
        'Shot': (Shot, Sensing),
        'RadiationPressure': RadiationPressure,
        'Thermal': ThermalDict,
        'Seismic': noise.seismic.Seismic,
    }

    references = Struct(
        Reference = DARMMeasured,
    )


class H1dictNoRefs(nb.Budget):
    noises = {
        'Shot': (Shot, Sensing),
        'RadiationPressure': RadiationPressure,
        'Thermal': ThermalDict,
        'Seismic': noise.seismic.Seismic,
    }


class H1dictNoRefsForwardNoises(nb.Budget):
    noises = {
        'Seismic': noise.seismic.Seismic,
    }

    noises_forward = {
        'Quantum': QuantumDict,
        'Thermal': ThermalDict,
    }
