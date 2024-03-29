from gwinc.ifo import PLOT_STYLE
from gwinc import noise
from gwinc import nb
import gwinc.ifo.noises as calibrations


class Voyager(nb.Budget):

    name = 'Voyager'

    noises = [
        noise.quantum.Quantum,
        noise.seismic.Seismic,
        noise.newtonian.Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        noise.coatingthermal.CoatingBrownian,
        noise.coatingthermal.CoatingThermoOptic,
        noise.substratethermal.ITMThermoRefractive,
        noise.substratethermal.SubstrateBrownian,
        noise.substratethermal.SubstrateThermoElastic,
        noise.residualgas.ResidualGas,
    ]

    calibrations = [
        calibrations.Strain,
    ]

    plot_style = PLOT_STYLE


class Displacement(Voyager):
    calibrations = []


class Acceleration(Voyager):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(Voyager):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(Voyager):
    calibrations = [
        calibrations.Force,
    ]
