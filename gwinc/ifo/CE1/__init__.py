from gwinc.ifo import PLOT_STYLE
from gwinc import noise
from gwinc import nb
import gwinc.ifo.noises as calibrations


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


class CE1(nb.Budget):

    name = 'Cosmic Explorer 1'

    noises = [
        noise.quantum.Quantum,
        noise.seismic.Seismic,
        noise.newtonian.Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        Coating,
        Substrate,
        noise.residualgas.ResidualGas,
    ]

    calibrations = [
        calibrations.Strain,
    ]

    plot_style = PLOT_STYLE


class Displacement(CE1):
    calibrations = []


class Acceleration(CE1):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(CE1):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(CE1):
    calibrations = [
        calibrations.Force,
    ]
