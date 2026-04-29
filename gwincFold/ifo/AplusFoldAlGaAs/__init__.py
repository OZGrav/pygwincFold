from gwincFold.ifo import PLOT_STYLE
from gwincFold import noise
from gwincFold import nb
import gwincFold.ifo.noises as calibrations


class AplusFoldAlGaAs(nb.Budget):

    name = 'A+FoldAlGaAs'

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


class Displacement(AplusFoldAlGaAs):
    calibrations = []


class Acceleration(AplusFoldAlGaAs):
    calibrations = [
        calibrations.Acceleration,
    ]


class Velocity(AplusFoldAlGaAs):
    calibrations = [
        calibrations.Velocity,
    ]


class Force(AplusFoldAlGaAs):
    calibrations = [
        calibrations.Force,
    ]
