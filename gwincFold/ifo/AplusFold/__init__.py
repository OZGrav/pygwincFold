from gwincFold.ifo import PLOT_STYLE
from gwincFold import noise
from gwincFold import nb
import gwincFold.ifo.noises as calibrations


class AplusFold(nb.Budget):

    name = 'A+Fold'

    noises = [
        noise.quantum.QuantumVacuum,
        noise.seismic.Seismic,
        noise.newtonian.Newtonian,
        noise.suspensionthermal.SuspensionThermal,
        noise.suspensionthermal.SuspensionThermalFolding,
        noise.coatingthermal.CoatingBrownian,
        noise.coatingthermal.CoatingThermoOptic,
        noise.substratethermal.SubstrateBrownian,
        noise.substratethermal.SubstrateThermoElastic,
        noise.residualgas.ResidualGas,
    ]

    calibrations = [
	calibrations.Strain
    ]



    plot_style = PLOT_STYLE


class Displacement(AplusFold):
    calibrations = []


class Acceleration(AplusFold):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(AplusFold):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(AplusFold):
    calibrations = [
        calibrations.Force,
    ]
