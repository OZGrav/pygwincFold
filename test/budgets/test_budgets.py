"""
"""
import numpy as np
import gwinc
from gwinc import load_budget
import gwinc.io as io
from copy import deepcopy
import pytest


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_load(ifo, pprint, tpath_join, fpath_join):
    B = load_budget(ifo)
    trace = B.run()
    fig = trace.plot()
    fig.savefig(tpath_join('budget_{}.pdf'.format(ifo)))


@pytest.mark.generate
@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_save_budgets(ifo, fpath_join):
    B = load_budget(ifo)
    traces = B.run()
    io.save_hdf5(traces, fpath_join(ifo + '.h5'))


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_check_noise(ifo, fpath_join, compare_noise):
    try:
        ref_traces = io.load_hdf5(fpath_join(ifo + '.h5'))
    except OSError:
        return
    budget = load_budget(ifo, freq=ref_traces.freq)
    traces = budget.run()
    compare_noise(traces, ref_traces)


@pytest.mark.logic
@pytest.mark.fast
def test_update_ifo_struct():
    """
    Test that the noise is recalculated when the ifo struct is updated
    """
    budget = gwinc.load_budget('CE2silica')
    tr1 = budget.run()
    budget.ifo.Suspension.VHCoupling.theta *= 2
    tr2 = budget.run()
    assert np.all(
        tr2.Seismic.SeismicVertical.asd == 2*tr1.Seismic.SeismicVertical.asd)


@pytest.mark.logic
@pytest.mark.fast
def test_change_ifo_struct():
    """
    Test that the noise is recalculated when a new ifo struct is passed to run
    """
    budget = gwinc.load_budget('CE2silica')
    ifo1 = deepcopy(budget.ifo)
    ifo2 = deepcopy(budget.ifo)
    ifo2.Suspension.VHCoupling.theta *= 2
    tr1 = budget.run(ifo=ifo1)
    tr2 = budget.run(ifo=ifo2)
    tr3 = budget.run(ifo=ifo1)
    assert np.all(tr1.asd == tr3.asd)
    assert np.all(
        tr2.Seismic.SeismicVertical.asd == 2*tr1.Seismic.SeismicVertical.asd)

