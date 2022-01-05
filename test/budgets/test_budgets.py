"""
"""
import numpy as np
import gwinc
from gwinc import load_budget
from copy import deepcopy
import pytest


def test_load(pprint, tpath_join, fpath_join):
    pprint(gwinc.IFOS)
    for ifo in gwinc.IFOS:
        B = load_budget(ifo)
        trace = B.run()
        fig = trace.plot()
        fig.savefig(tpath_join('budget_{}.pdf'.format(ifo)))


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


@pytest.mark.logic
@pytest.mark.fast
def test_update_freq():
    """
    Test three methods of updating a Budget frequency
    """
    freq1 = np.logspace(1, 3, 10)
    freq2 = np.logspace(0.8, 3.5, 11)
    freq3 = np.logspace(0.5, 3.6, 12)
    budget = gwinc.load_budget('Aplus', freq=freq1)
    traces1 = budget.run()
    traces2 = budget.run(freq=freq2)
    budget.freq = freq3
    traces3 = budget.run()
    assert np.all(traces1.freq == freq1)
    assert np.all(traces2.freq == freq2)
    assert np.all(traces3.freq == freq3)


@pytest.mark.logic
@pytest.mark.fast
def test_freq_spec_error():
    """
    Test that three methods of setting Budget frequencies raise errors
    """
    freq = [1, 2, 3]
    with pytest.raises(gwinc.InvalidFrequencySpec):
        budget = gwinc.load_budget('Aplus', freq=freq)
    with pytest.raises(AssertionError):
        budget = gwinc.load_budget('Aplus')
        traces = budget.run(freq=freq)
    with pytest.raises(AssertionError):
        budget = gwinc.load_budget('Aplus')
        budget.freq = freq

