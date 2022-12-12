"""
"""
import pytest

from gwinc import load_budget


@pytest.mark.fast
@pytest.mark.logic
def test_inherit_load(pprint, tpath_join, fpath_join):
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

    fpath3 = fpath_join('Aplus_mod3.yaml')
    B_inherit3 = load_budget(fpath3)
    pprint(B_inherit3.ifo.diff(B_orig.ifo))
    assert(
        B_inherit3.ifo.diff(B_orig.ifo)
        == [('Optics.Quadrature.dc', None, 1.5707963)]
    )


@pytest.mark.fast
@pytest.mark.logic
def test_inherit_custom_budget(fpath_join):
    """
    Test that inheritance works when the final budget is a custom budget in
    an arbitrary directory instead of one of the canonical budgets
    """
    B_inherit = load_budget(fpath_join('subdir/CustomBudget_mod.yaml'))
    B_orig = load_budget(fpath_join('subdir/CustomBudget'))

    noises = [noise.__name__ for noise in B_inherit.noises]

    assert B_inherit.name == 'A custom budget'
    assert sorted(noises) == ['Quantum', 'Seismic']
    assert(
        sorted(B_inherit.ifo.diff(B_orig.ifo))
        == sorted([
            ('Squeezer.AmplitudedB', 14, 12),
            ('Squeezer.FilterCavity.Lrt', 40e-6, 60e-6),
        ])
    )


@pytest.mark.fast
@pytest.mark.logic
def test_load_uninherited_yaml(fpath_join):
    """
    Test that a yaml file not specifying an inherited budget is loaded into
    the aLIGO budget
    """
    B_new = load_budget(fpath_join('new_ifo.yaml'))
    B_aLIGO = load_budget('aLIGO')

    assert B_new.name == 'Advanced LIGO'
    assert(
        sorted(B_new.ifo.diff(B_aLIGO.ifo))
        == sorted([
            ('Optics.Loss', 3.75e-05, 4e-05),
            ('Squeezer.AmplitudedB', 12, None),
            ('Squeezer.FilterCavity.L', 300, None),
        ])
    )


@pytest.mark.fast
@pytest.mark.logic
def test_inherit_fail(pprint, tpath_join, fpath_join):
    """
    This test shows that the interim logic that Struct.from_file checks for and fails when it gets "+inherit" keys.
    Those will be allowable at a later time.
    """
    fpath2 = fpath_join('inherit_fail.yaml')

    with pytest.raises(RuntimeError):
        B_inherit2 = load_budget(fpath2)


@pytest.mark.fast
@pytest.mark.logic
def test_load_fail_unknown_filetype(fpath_join):
    """
    Test that load_budget fails when given a file type not supported by Struct
    """
    with pytest.raises(RuntimeError):
        budget = load_budget(fpath_join('new_ifo.txt'))


@pytest.mark.fast
@pytest.mark.logic
def test_load_fail_unknown_ifo():
    """
    Test that load_budget fails when given an unknown ifo that doesn't exist
    """
    with pytest.raises(RuntimeError):
        budget = load_budget('not_a_real_ifo')

