"""
Unit tests for residual gas noise
"""
import numpy as np
import gwinc
import gwinc.noise.residualgas as resgas
from gwinc.ifo.noises import dhdl, arm_cavity
from gwinc import const
import pytest
from itertools import product
import matplotlib.pyplot as plt
from timeit import timeit
from importlib import import_module
from numpy import sqrt, log, pi


SPECIES = [
    'H2',
    'N2',
    'H2O',
    'O2',
]


def timer(func, *args, **kwargs):
    """
    Helper function for timing functions with timeit
    """
    def timeit_func():
        return func(*args, **kwargs)
    return timeit_func


def test_custom_beamtube_pressure(fpath_join, tpath_join):
    """Plot the AplusSuperGas budget specifying a custom pressure profile
    """
    F_Hz = np.logspace(np.log10(5), 5, 3000)
    budget = gwinc.load_budget(fpath_join('AplusSuperGas'), freq=F_Hz)
    traces = budget.run()
    fig_total = traces.plot()
    fig_resgas = traces.ResidualGas.plot()
    fig_total.savefig(tpath_join('total.pdf'))
    fig_resgas.savefig(tpath_join('residual_gas.pdf'))


@pytest.mark.parametrize('ifo_name', gwinc.IFOS)
def test_compare_scattering_budgets(ifo_name, tpath_join):
    """
    Compare the budgets for the exact and old approximate gas scattering
    calculations.
    """
    # import, initialize, and run the default exact budget
    budget = gwinc.load_budget(ifo_name)
    traces = budget.run()

    # import the default exact budget
    mod = import_module('gwinc.ifo.{:s}'.format(ifo_name))
    Budget_approx = getattr(mod, ifo_name)
    # replace the exact calculation with the approximate one
    Budget_approx.noises.pop(-1)
    Budget_approx.noises.append(ResidualGasApprox)
    # initialize and run the approximate budget with the default ifo
    budget_approx = Budget_approx(ifo=budget.ifo, freq=budget.freq)
    traces_approx = budget_approx.run()

    fig = traces.plot()
    fig_rg = traces.ResidualGas.plot()
    ylims = fig.gca().get_ylim()
    ylims_rg = fig_rg.gca().get_ylim()
    fig.gca().set_title('{:s} Exact Calculation'.format(ifo_name))
    fig_rg.gca().set_title('{:s} Exact Calculation'.format(ifo_name))
    fig.savefig(tpath_join('total.pdf'))
    fig_rg.savefig(tpath_join('resgas.pdf'))

    fig_approx = traces_approx.plot()
    fig_approx_rg = traces_approx.ResidualGasApprox.plot()
    fig_approx.gca().set_ylim(*ylims)
    fig_approx_rg.gca().set_ylim(*ylims_rg)
    fig_approx.gca().set_title('{:s} Approximate Calculation'.format(ifo_name))
    fig_approx_rg.gca().set_title('{:s} Approximate Calculation'.format(ifo_name))
    fig_approx.savefig(tpath_join('total_approx.pdf'))
    fig_approx_rg.savefig(tpath_join('resgas_approx.pdf'))


@pytest.mark.parametrize('species_name, ifo', product(SPECIES, gwinc.IFOS))
def test_scattering_calcs(species_name, ifo, tpath_join):
    """
    Compare the exact gas scattering calculations using different size pressure
    arrays with the old approximate calculation
    """
    budget = gwinc.load_budget(ifo)
    ifo = budget.ifo
    F_Hz = np.logspace(1, 4, 3000)
    cavity = resgas.arm_cavity(ifo)
    Larm_m = ifo.Infrastructure.Length
    species = ifo.Infrastructure.ResidualGas[species_name]
    pressure_Pa = species.BeamtubePressure
    approx = residual_gas_scattering_arm_approx(F_Hz, ifo, cavity, species)
    approx = np.sqrt(approx)

    def calc_exact(npts):
        position_m = np.linspace(0, Larm_m, npts)
        exact = resgas.residual_gas_scattering_arm(
            F_Hz, ifo, cavity, species, pressure_Pa, position_m)
        return np.sqrt(exact)

    exact1000 = calc_exact(1000)
    exact500  = calc_exact(500)
    exact200  = calc_exact(200)
    exact100  = calc_exact(100)
    exact50   = calc_exact(50)

    fig, ax = plt.subplots()
    ax.loglog(F_Hz, exact1000, label='exact')
    ax.loglog(F_Hz, approx, label='approximate')
    ax.set_xlim(F_Hz[0], F_Hz[-1])
    ax.set_ylim(5e-23, 5e-21)
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Strain noise [1/Hz$^{-1/2}$]')
    ax.set_title(species_name)
    ax.legend()
    fig.savefig(tpath_join('resgas.pdf'))

    fig_rat, ax_rat = plt.subplots()
    ax_rat.semilogx(F_Hz, np.abs(1 - exact500 / exact1000), label='500')
    ax_rat.semilogx(F_Hz, np.abs(1 - exact200 / exact1000), label='200')
    ax_rat.semilogx(F_Hz, np.abs(1 - exact100 / exact1000), label='100')
    ax_rat.semilogx(F_Hz, np.abs(1 - exact50 / exact1000), label='50')
    ax_rat.semilogx(F_Hz, np.abs(1 - approx / exact1000), label='approximate')
    ax_rat.set_yscale('log')
    ax_rat.set_xlim(F_Hz[0], F_Hz[-1])
    ax_rat.set_ylim(1e-7, 1e-2)
    ax_rat.set_xlabel('Frequency [Hz]')
    ax_rat.set_ylabel('Ratio with 1000 points')
    ax_rat.set_title(species_name)
    ax_rat.legend()
    ax_rat.grid(True, which='major', alpha=0.5)
    ax_rat.grid(True, which='minor', alpha=0.2)
    fig_rat.savefig(tpath_join('ratios.pdf'))


@pytest.mark.slow
def test_time(pprint):
    """
    Compare the new exact and the old approximate gas scattering calculations
    """
    niter = 500
    budget = gwinc.load_budget('Aplus')
    ifo = budget.ifo
    F_Hz = np.logspace(1, 4, 3000)
    cavity = resgas.arm_cavity(ifo)
    Larm_m = ifo.Infrastructure.Length
    species = ifo.Infrastructure.ResidualGas['N2']
    pressure_Pa = species.BeamtubePressure

    position_m = np.linspace(0, Larm_m, 100)
    time_exact = timeit(
        timer(
            resgas.residual_gas_scattering_arm,
            F_Hz, ifo, cavity, species, pressure_Pa, position_m
        ),
        number=niter
    ) / niter

    time_approx = timeit(
        timer(
            residual_gas_scattering_arm_approx,
            F_Hz, ifo, cavity, species
        ),
        number=niter
    ) / niter

    pprint('approximate: {:0.3f} ms'.format(1e3 * time_approx))
    pprint('exact: {:0.3f} ms'.format(1e3 * time_exact))
    pprint('relative change: {:0.2f}'.format(time_exact / time_approx))


############################################################
# Old approximate gas scattering functions for comparison
############################################################


def ResidualGasScatteringApprox_constructor(species_name):
    """Residual gas scattering for a single species

    """

    class GasScatteringSpecies(resgas.nb.Noise):
        name = 'Scattering' + species_name
        style = dict(
            label=resgas.RESGAS_STYLES[species_name]['label'] + ' scattering',
            color=resgas.RESGAS_STYLES[species_name]['color'],
            linestyle='-',
        )

        def calc(self):
            cavity = arm_cavity(self.ifo)
            species = self.ifo.Infrastructure.ResidualGas[species_name]
            n = residual_gas_scattering_arm_approx(
                self.freq, self.ifo, cavity, species)
            dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
            return n * 2 / sinc_sqr

    return GasScatteringSpecies


def residual_gas_scattering_arm_approx(f, ifo, cavity, species):
    """Residual gas noise strain spectrum due to scattering from one arm

    Noise caused by the passage of residual gas molecules through the
    laser beams in one arm cavity due to scattering.

    :f: frequency array in Hz
    :ifo: gwinc IFO structure
    :cavity: arm cavity structure
    :species: molecular species structure

    :returns: arm strain noise power spectrum at :f:

    The method used here is presented by Rainer Weiss, Micheal
    E. Zucker, and Stanley E. Whitcomb in their paper Optical
    Pathlength Noise in Sensitive Interferometers Due to Residual Gas.

    Added to Bench by Zhigang Pan, Summer 2006
    Cleaned up by PF, Apr 07
    Eliminated numerical integration and substituted first order
    expansion of exp, to speed it up.

    """
    L = ifo.Infrastructure.Length
    kT = ifo.Infrastructure.Temp * const.kB
    P = species.BeamtubePressure
    M = species.mass
    alpha = species.polarizability

    rho = P / (kT)                   # number density of Gas
    v0 = sqrt(2*kT / M)              # mean speed of Gas

    waist = cavity.w0                # Gaussian beam waist size
    zr = cavity.zr                   # Rayleigh range
    z1 = -cavity.zBeam_ITM           # location of ITM relative to the waist
    z2 = cavity.zBeam_ETM            # location of ETM relative to the waist

    # The exponential of Eq. 1 of P940008 is expanded to first order; this
    # can be integrated analytically
    zint = log(z2 + sqrt(z2**2 + zr**2)) - log(z1 + sqrt(z1**2 + zr**2))
    zint = zint * zr/waist
    zint = zint - 2*pi*L*f/v0
    # optical path length for one arm
    zint = zint*((4*rho*(2*pi*alpha)**2)/v0)
    # eliminate any negative values due to first order approx.
    zint[zint < 0] = 0

    return zint


class ResidualGasApprox(resgas.nb.Budget):
    """Residual Gas

    """
    style = dict(
        label='Residual Gas',
        color='#add00d',
        linestyle='-',
    )

    noises = [
        ResidualGasScatteringApprox_constructor('H2'),
        ResidualGasScatteringApprox_constructor('N2'),
        ResidualGasScatteringApprox_constructor('H2O'),
        ResidualGasScatteringApprox_constructor('O2'),
        resgas.ResidualGasDamping_constructor('H2'),
        resgas.ResidualGasDamping_constructor('N2'),
        resgas.ResidualGasDamping_constructor('H2O'),
        resgas.ResidualGasDamping_constructor('O2'),
    ]
