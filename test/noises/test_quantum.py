"""
Unit tests for quantum noise
"""
import gwinc
from gwinc.noise.quantum import getSqzParams
from copy import deepcopy
import pytest


@pytest.mark.logic
@pytest.mark.fast
def test_no_squeezer_type():
    """Test that the appropriate options result in no squeezer
    """
    def assert_no_squeezer(params):
        assert params.sqzType == 'None'
        assert params.SQZ_DB == 0
        assert params.ANTISQZ_DB == 0
        assert params.lambda_in == 0
        assert params.etaRMS == 0

    budget = gwinc.load_budget('Aplus')
    ifo1 = deepcopy(budget.ifo)
    ifo2 = deepcopy(budget.ifo)
    ifo3 = deepcopy(budget.ifo)

    # there should be no squeezer if
    # the squeezer struct is missing
    del ifo1.Squeezer
    # or the squeezing amplitude is 0
    ifo2.Squeezer.AmplitudedB = 0
    # or the squeezer type is 'None'
    ifo3.Squeezer.Type = 'None'

    assert_no_squeezer(getSqzParams(ifo1))
    assert_no_squeezer(getSqzParams(ifo2))
    assert_no_squeezer(getSqzParams(ifo2))


@pytest.mark.logic
@pytest.mark.fast
@pytest.mark.skip(reason='Needs to be implemented')
def test_lo_params():
    """Test the logic for the various LO options
    """
    pass
