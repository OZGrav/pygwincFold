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


class CustomBudget(nb.Budget):

    name = 'A custom budget'

    noises = [
        Quantum,
        noise.seismic.Seismic,
    ]

    calibrations = [
        Strain,
    ]

    plot_style = PLOT_STYLE
