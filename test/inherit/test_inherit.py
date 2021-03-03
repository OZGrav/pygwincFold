"""
"""
from gwinc import load_budget


def test_load(pprint, tpath_join, fpath_join):
    fpath = fpath_join('Aplus_mod.yaml')
    B_inherit = load_budget(fpath)
    B_orig = load_budget('Aplus')

    pprint(B_inherit.ifo)
    pprint("ACTUAL TEST")
    pprint(B_inherit.ifo.diff(B_orig.ifo))

    assert(
        sorted(B_inherit.ifo.diff(B_orig.ifo))
        == sorted([
            ('Suspension.Stage[3].Mass', 30, 22.1),
            ('Squeezer.AmplitudedB', 14, 12),
            ('Squeezer.InjectionLoss', 0.02, 0.05)])
    )

    fpath2 = fpath_join('Aplus_mod2.yaml')
    B_inherit2 = load_budget(fpath2)
    pprint(B_inherit2.ifo.diff(B_orig.ifo))
    assert(
        sorted(B_inherit2.ifo.diff(B_orig.ifo))
        == sorted([
            ('Suspension.Stage[2].Mass', 30, 21.8),
            ('Suspension.Stage[3].Mass', 30, 22.1),
            ('Squeezer.InjectionLoss', 0.02, 0.05)
        ])
    )

