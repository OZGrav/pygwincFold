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

