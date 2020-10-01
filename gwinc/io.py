import h5py
import yaml
import datetime

from . import logger
from . import Struct
from .trace import BudgetTrace


SCHEMA = 'GWINC noise budget'


def _write_trace_recursive(f, trace):
    f.attrs['style'] = yaml.dump(trace.style)
    f.create_dataset('PSD', data=trace.psd)
    if trace.budget:
        bgrp = f.create_group('budget')
    for t in trace:
        tgrp = bgrp.create_group(t.name)
        _write_trace_recursive(tgrp, t)


def save_hdf5(path, trace, ifo=None, plot_style=None, **kwargs):
    """Save GWINC budget data to an HDF5 file.

    The `freq` argument should be the frequency array, and `traces`
    should be the traces (recursive) dictionary.  Keyword arguments
    are stored in the HDF5 top level 'attrs' key-value store.  If an
    'ifo' keyword arg is supplied, it is assumed to be a Struct and
    will be serialized to YAML for storage.

    See HDF5_SCHEMA.

    """
    with h5py.File(path, 'w') as f:
        f.attrs['SCHEMA'] = SCHEMA
        f.attrs['SCHEMA_VERSION'] = 2
        # FIXME: add budget code hash or something
        f.attrs['date'] = datetime.datetime.now().isoformat()
        if ifo:
            f.attrs['ifo'] = ifo.to_yaml()
        if plot_style:
            f.attrs['plot_style'] = yaml.dump(plot_style)
        for key, val in kwargs.items():
            f.attrs[key] = val
        f.create_dataset('Freq', data=trace.freq)
        tgrp = f.create_group(trace.name)
        _write_trace_recursive(tgrp, trace)


def _load_hdf5_v1(f):
    def read_recursive(element):
        budget = []
        for name, item in element.items():
            if name == 'Total':
                total = item[:]
                continue
            if isinstance(item, h5py.Group):
                trace = read_recursive(item)
                trace.name = name
            else:
                trace = BudgetTrace(
                    name=name,
                    style=dict(item.attrs),
                    psd=item[:],
                    budget=[],
                )
            budget.append(trace)
        return BudgetTrace(
            psd=total,
            budget=budget,
        )
    trace = read_recursive(f['/traces'])
    trace.name = 'Total'
    trace._freq = f['Freq'][:]
    attrs = dict(f.attrs)
    if 'ifo' in attrs:
        try:
            ifo = Struct.from_yaml(attrs['ifo'])
            del attrs['ifo']
        except yaml.constructor.ConstructorError:
            logger.warning("HDF5 load warning: Could not de-serialize 'ifo' YAML attribute.")
            ifo = None
    trace.style = attrs
    trace.ifo = ifo
    return trace


def _load_hdf5_v2(f):
    def read_recursive(element):
        psd = element['PSD'][:]
        style = yaml.safe_load(element.attrs['style'])
        budget = []
        if 'budget' in element:
            for name, item in element['budget'].items():
                trace = read_recursive(item)
                trace.name = name
                budget.append(trace)
        return BudgetTrace(
            psd=psd,
            budget=budget,
            style=style,
        )
    for name, item in f.items():
        if name == 'Freq':
            freq = item[:]
        else:
            trace = read_recursive(item)
            trace.name = name
    trace._freq = freq
    trace.ifo = None
    attrs = dict(f.attrs)
    ifo = attrs.get('ifo')
    if ifo:
        try:
            trace.ifo = Struct.from_yaml(ifo)
            del attrs['ifo']
        except yaml.constructor.ConstructorError:
            logger.warning("HDF5 load warning: Could not de-serialize 'ifo' YAML attribute.")
    trace.plot_style = yaml.safe_load(attrs['plot_style'])
    return trace


def load_hdf5(path):
    """Load GWINC budget data from an HDF5 file.

    Returns BudgetTrace.  An 'ifo' attr will be de-serialized from
    YAML into a Struct object and assigned as an attribute to the
    BudgetTrace.

    See HDF5_SCHEMA.

    """
    loaders = {
        1: _load_hdf5_v1,
        2: _load_hdf5_v2,
    }
    with h5py.File(path, 'r') as f:
        version = f.attrs.get('SCHEMA_VERSION', 1)
        return loaders[version](f)
