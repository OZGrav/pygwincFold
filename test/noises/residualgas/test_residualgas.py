"""
Unit tests for residual gas noise
"""
import numpy as np
import gwinc


def test_custom_beamtube_pressure(fpath_join, tpath_join):
    F_Hz = np.logspace(np.log10(5), 5, 3000)
    budget = gwinc.load_budget(fpath_join('AplusSuperGas'), freq=F_Hz)
    traces = budget.run()
    fig_total = traces.plot()
    fig_resgas = traces.ResidualGas.plot()
    fig_total.savefig(tpath_join('total.pdf'))
    fig_resgas.savefig(tpath_join('residual_gas.pdf'))
