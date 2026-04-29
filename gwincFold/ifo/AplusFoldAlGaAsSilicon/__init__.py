from gwincFold.ifo import PLOT_STYLE
from gwincFold import noise
from gwincFold import nb
import gwincFold.ifo.noises as calibrations


class AplusFoldAlGaAsSilicon(nb.Budget):

    name = 'A+FoldAlGaAsSilicon'

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


class Displacement(AplusFoldAlGaAsSilicon):
    calibrations = []


class Acceleration(AplusFoldAlGaAsSilicon):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(AplusFoldAlGaAsSilicon):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(AplusFoldAlGaAsSilicon):
    calibrations = [
        calibrations.Force,
    ]
