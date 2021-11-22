from gwinc.ifo.noises import *
from gwinc.ifo import PLOT_STYLE


class QuantumVacuum(nb.Budget):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    noises = [
        QuantumVacuumAS,
        QuantumVacuumArm,
        QuantumVacuumSEC,
        QuantumVacuumFilterCavity,
        QuantumVacuumInjection,
        QuantumVacuumReadout,
        QuantumVacuumQuadraturePhase,
    ]


class CustomBudget(nb.Budget):

    name = 'A custom budget'

    noises = [
        QuantumVacuum,
        Seismic,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
