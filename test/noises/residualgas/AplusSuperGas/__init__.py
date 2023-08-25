from os import path

import numpy as np

from gwinc.ifo import PLOT_STYLE
from gwinc import noise
from gwinc import nb
from gwinc.ifo.noises import Strain, dhdl


class Quantum(nb.Budget):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    noises = [
        noise.quantum.AS,
        noise.quantum.Arm,
        noise.quantum.SEC,
        noise.quantum.FilterCavity,
        noise.quantum.Injection,
        noise.quantum.Readout,
        noise.quantum.QuadraturePhase,
    ]


def ResidualGasScattering_constructor(species_name, tube):
    """Residual gas scattering for a single species and a single arm

    """

    colors = dict(
        H2_Y='xkcd:red orange',
        N2_Y='xkcd:emerald',
        H2O_Y='xkcd:water blue',
        H2_X='xkcd:blood red',
        N2_X='xkcd:emerald',
        H2O_X='xkcd:dark blue',
    )

    style0 = dict(color=colors[species_name + '_' + tube])
    style0['label'] = tube + ' arm '
    style0['label'] += noise.residualgas.RESGAS_STYLES[species_name]['label']
    style0['label'] += ' scattering'

    if tube == 'Y':
        style0['linestyle'] = '-'
    elif tube == 'X':
        style0['linestyle'] = '-.'

    class GasScatteringSpeciesTube(nb.Noise):
        name = 'Scattering' + tube + 'arm' + species_name
        style = style0

        def load(self):
            bpath = self.load.__code__.co_filename
            fname = path.join(path.split(bpath)[0], 'beamtube_pressure.txt')
            dtype = [(name, float) for name in ('H2_Y', 'H2O_Y', 'N2_Y', 'H2_X', 'H2O_X', 'N2_X', 'position_m')]
            df = np.loadtxt(fname, dtype=dtype)
            pressure_torr = df[species_name + '_' + tube]
            self.tubepos_m = df['position_m']
            self.pressure_Pa = pressure_torr * 133.3

        def calc(self):
            cavity = noise.residualgas.arm_cavity(self.ifo)
            species = self.ifo.Infrastructure.ResidualGas[species_name]
            n = noise.residualgas.residual_gas_scattering_arm(
                self.freq, self.ifo, cavity, species, self.pressure_Pa,
                self.tubepos_m)
            dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
            # note that this is for a single arm, so no factor of 2
            return n / sinc_sqr

    return GasScatteringSpeciesTube


class ResidualGas(nb.Budget):
    """Residual Gas

    """
    name = 'ResidualGas'

    style = dict(
        label='Residual Gas',
        color='#add00d',
        linestyle='-',
    )

    noises = [
        ResidualGasScattering_constructor('H2', 'Y'),
        ResidualGasScattering_constructor('N2', 'Y'),
        ResidualGasScattering_constructor('H2O', 'Y'),
        ResidualGasScattering_constructor('H2', 'X'),
        ResidualGasScattering_constructor('N2', 'X'),
        ResidualGasScattering_constructor('H2O', 'X'),
        noise.residualgas.ResidualGasDamping_constructor('H2'),
        noise.residualgas.ResidualGasDamping_constructor('N2'),
        noise.residualgas.ResidualGasDamping_constructor('H2O'),
    ]


class AplusSuperGas(nb.Budget):

    name = 'A+'

    noises = [
        Quantum,
        noise.seismic.Seismic,
        noise.newtonian.Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        noise.coatingthermal.CoatingBrownian,
        noise.coatingthermal.CoatingThermoOptic,
        noise.substratethermal.SubstrateBrownian,
        noise.substratethermal.SubstrateThermoElastic,
        ResidualGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
