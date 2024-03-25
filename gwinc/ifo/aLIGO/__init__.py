from gwinc.ifo import PLOT_STYLE
from gwinc import noise
from gwinc import nb
import gwinc.ifo.noises as calibrations


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
        noise.quantum.Readout,
    ]


class aLIGO(nb.Budget):

    name = 'Advanced LIGO'

    noises = [
        Quantum,
        noise.seismic.Seismic,
        noise.newtonian.Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        noise.coatingthermal.CoatingBrownian,
        noise.coatingthermal.CoatingThermoOptic,
        noise.substratethermal.SubstrateBrownian,
        noise.substratethermal.SubstrateThermoElastic,
        noise.residualgas.ResidualGas,
    ]

    calibrations = [
        calibrations.Strain,
    ]

    plot_style = PLOT_STYLE


class Displacement(aLIGO):
    calibrations = []


class Acceleration(aLIGO):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(aLIGO):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(aLIGO):
    calibrations = [
        calibrations.Force,
    ]
