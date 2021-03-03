"""
"""
import numpy as np
from os import path
from gwinc import Struct
import pylab as pyl


def test_load(pprint, tpath_join, fpath_join):
    fpath = fpath_join('test_load.yml')
    yml = Struct.from_file(fpath)
    pprint("full yaml")
    pprint(yml)
    pprint("individual tests")
    pprint(yml.lists.test_nonelist)
    assert(yml.lists.test_nonelist == [None, None])
    pprint(yml.dicts.test_nonedict)
    S_cmp = Struct({'A': None, 'B': None})
    pprint(yml.dicts.test_nonedict.diff(S_cmp))
    assert(yml.dicts.test_nonedict == S_cmp)
