from gwinc.ifo import PLOT_STYLE
from gwinc import noise
from gwinc import nb
from gwinc.ifo.noises import Strain


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


class Newtonian(nb.Budget):
    """Newtonian Gravity

    """

    name = 'Newtonian'

    style = dict(
        label='Newtonian',
        color='#15b01a',
    )

    noises = [
        noise.newtonian.Rayleigh,
        noise.newtonian.Body,
        noise.newtonian.Infrasound,
    ]


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


class CE2silica(nb.Budget):

    name = 'Cosmic Explorer 2 (Silica)'

    noises = [
        Quantum,
        noise.seismic.Seismic,
        Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        Coating,
        Substrate,
        noise.residualgas.ResidualGas,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
