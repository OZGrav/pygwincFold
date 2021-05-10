"""
"""
import gwinc
from gwinc import load_budget


def test_load(pprint, tpath_join, fpath_join):
    pprint(gwinc.IFOS)
    for ifo in gwinc.IFOS:
        B = load_budget(ifo)
        trace = B.run()
        fig = trace.plot()
        fig.savefig(tpath_join('budget_{}.pdf'.format(ifo)))

