import itertools
import collections
import numpy as np
import scipy.interpolate

from . import logger
from .trace import BudgetTrace


def precomp(*precomp_funcs, **precomp_fmaps):
    """BudgetItem.calc decorator to add pre-computed functions

    This is intended to decorate BudgetItem.calc() methods with
    functions that would update the BudgetItem.ifo attribute.  All
    precomp functions are collected, and executed after attribute
    update during BudgetItem.update() calls.  They are supplied with
    the `freq` array and `ifo` Struct attributes as arguments, and are
    intended to update the `ifo` with derived values needed during
    subsequent calc() calls.

    For example, if a calc method is defined as:

      @precomp(precomp_foo)
      @precomp(precomp_bar)
      def calc(self):
          ...

    then the update method will execute the following after attribute
    update:

      precomp_foo(self.freq, self.ifo)
      precomp_bar(self.freq, self.ifo)

    The execution state of precomp functions are cached so that the
    same function is not needlessly executed multiple times.

    """
    def decorator(func):
        if precomp_funcs:
            try:
                func._precomp_list.extend(precomp_funcs)
            except AttributeError:
                #probably don't need to copy the **kwargs dict
                func._precomp_list = list(precomp_funcs)

        if precomp_fmaps:
            try:
                func._precomp_mapped.update(precomp_fmaps)
            except AttributeError:
                #probably don't need to copy the **kwargs dict
                func._precomp_mapped = dict(precomp_fmaps)
        return func
    return decorator


def quadsum(data):
    """Calculate quadrature sum of list of data arrays.

    Provided data are assumed to be power-referred, so this is a
    simple point-by-point sum.

    NaNs in sum elements do not contribute to sum.

    """
    return np.nansum(data, 0)


def _precomp_recurse_mapping(func, freq, ifo, _precomp):
    """
    Recurses down functions which may themselves have precomp decorators, this
    builds the **kwarg mapping to pass to the function call, and this mapping
    is returned
    """
    #run the prerequisite precomps first. These typically modify the ifo Struct (yuck)
    plist_set = _precomp.setdefault("_precomp_list", set())
    for pc_func in getattr(func, '_precomp_list', []):
        if pc_func in plist_set:
            continue
        pc_map = _precomp_recurse_mapping(pc_func, freq, ifo, _precomp=_precomp)
        #now call the function with the built mapping
        pc_func(freq, ifo, **pc_map)
        plist_set.add(pc_func)

    #now run the prerequisite mappings. These return values which get mapped
    precomp_mapping = dict()
    for name, pc_func in getattr(func, '_precomp_mapped', {}).items():
        try:
            PC = _precomp[pc_func]
        except KeyError:
            #not in _precomp already
            pass
        else:
            precomp_mapping[name] = PC
            continue
        logger.debug("precomp {}".format(pc_func))
        #build the mapping for the requisite call
        pc_map = _precomp_recurse_mapping(pc_func, freq, ifo, _precomp=_precomp)
        #now call the function with the built mapping
        PC = pc_func(freq, ifo, **pc_map)
        precomp_mapping[name] = PC

    return precomp_mapping

class BudgetItem:
    """GWINC BudgetItem class

    """
    def load(self):
        """Overload method for initial loading of static data.

        """
        return None

    def update(self, _precomp=None, **kwargs):
        """Update parameters and execute precomp functions.

        The method does two things.  First, any keyword arguments
        provided are written directly as attribute variables to the
        class (as with __init__).

        Second, after attribute update, all functions provided by
        @precomp decorators to the calc() method will be executed,
        supplied with the `freq` and `ifo` attributes as arguments.
        See the `precomp` documentation for more information.

        The `_precomp` keyword argument is for internal use. If provided, it is
        assumed to be a dictionary of previously executed precomp functions
        mapped to their return values. Any function included in the dict will
        not be re-executed, and the dict will be updated with any newly executed
        functions.

        This method can be overridden, but if it is, it's important to
        make sure that the method defined in the base class is always
        executed with e.g.:

          super().update(**kwargs)

        """
        for key, val in kwargs.items():
            setattr(self, key, val)
        if _precomp is None:
            _precomp = dict()

        _PCmap = _precomp_recurse_mapping(self.calc, self.freq, self.ifo, _precomp=_precomp)
        #PCmap is not used for this "dry run" update. _precomp could be cached?
        _PCmap  # I just refer to _PCmap here to appease the linter
        return

    def calc(self):
        """Overload method for final PSD calculation.

        Should return an array of power-referenced values evaluated at
        all evaluation frequencies (self.freq).

        """
        return None

    ##########

    def __init__(self, freq=None, **kwargs):
        """Initialize budget item.

        The primary argument should be the evaluation frequency array.
        If not provided, then a pre-defined `freq` attribute of the
        BudgetItem class should exist.

        Additional keyword arguments are written as attribute
        variables to the initialized object.

        """
        if freq is not None:
            assert isinstance(freq, np.ndarray)
            self.freq = freq
        for key, val in kwargs.items():
            setattr(self, key, val)
        self._loaded = False

    @property
    def name(self):
        """"Name of this BudgetItem class."""
        return self.__class__.__name__

    def __str__(self):
        # FIXME: provide info on internal state (load/update/calc/etc.)
        return '<{} {}>'.format(
            ', '.join([c.__name__ for c in self.__class__.__bases__]),
            self.name,
        )

    def interpolate(self, freq, data):
        """Interpolate data to the evaluation frequencies.

        """
        func = scipy.interpolate.interp1d(
            freq, data,
            kind='nearest',
            copy=False,
            assume_sorted=True,
            bounds_error=False,
            fill_value=np.nan,
        )
        return func(self.freq)


class Calibration(BudgetItem):
    """GWINC Calibration class

    BudgetItem that represents a calibration transfer function for a
    Noise.  The calc() method should return a transfer function
    amplitude array evaluated at the evaluation frequencies supplied
    at initialization and available in the `freq` array attribute
    (self.freq).

    """
    def __call__(self, data):
        """Calibrate input data.

        Returns calibrated version of input data array,
        e.g. point-by-point product of data and calibration arrays.

        """
        cal = self.calc()
        assert data.shape == cal.shape, \
            "data shape does not match calibration ({} != {})".format(data.shape, cal.shape)
        return data * cal


class Noise(BudgetItem):
    """GWINC Noise class

    BudgetItem that represents a PSD noise calculation.  The calc()
    method should return the noise PSD spectrum array evaluated at the
    evaluation frequencies supplied at initialization and available in
    the `freq` array attribute (self.freq).

    """

    style = {}
    """Trace plot style dictionary"""

    def _make_trace(self, psd=None, budget=None):
        return BudgetTrace(
            name=self.name,
            style=self.style,
            freq=self.freq,
            psd=psd,
            budget=budget,
        )

    def calc_trace(self, calibration=1, calc=True, _precomp = None):
        """Calculate noise and return BudgetTrace object

        `calibration` should either be a scalar or a len(self.freq)
        array that will be multiplied to the output PSD of the budget
        and all sub noises.

        If `calc` is False, the noise will not be calculated and the
        trace PSD will be None.  This is useful for just getting the
        trace style info.

        """
        if _precomp is None:
            _precomp = dict()

        total = None
        if calc:
            PCmap = _precomp_recurse_mapping(self.calc, self.freq, self.ifo, _precomp)
            total = self.calc(**PCmap) * calibration

        return self._make_trace(psd=total)

    def run(self, **kwargs):
        """Calculate budget noise and return BudgetTrace object.

        Roughly the equivalent running load(), update(), and
        calc_trace() in sequence.  Keyword arguments are passed to the
        update() method.

        NOTE: The load status is cached such that subsequent calls to
        this method will not re-execute the load() method.

        NOTE: The update() method is only run if keyword arguments
        (`kwargs`) are supplied, or if the `ifo` attribute has
        changed.

        """
        if not self._loaded:
            self.load()
            self._loaded = True

        _precomp = dict()
        ifo = kwargs.get('ifo', getattr(self, 'ifo'))
        if ifo:
            if not hasattr(ifo, '_orig_keys'):
                logger.debug("new ifo detected")
                ifo._orig_keys = tuple(k for k, v in ifo.walk())
                ifo_hash = ifo.hash()
                kwargs['ifo'] = ifo
            else:
                ifo_hash = ifo.hash(ifo._orig_keys)
                if ifo_hash != getattr(self, '_ifo_hash', 0):
                    logger.debug("ifo hash change")
                    kwargs['ifo'] = self.ifo
            self._ifo_hash = ifo_hash
        if kwargs:
            self.update(_precomp = precomp, **kwargs)

        return self.calc_trace(_precomp = _precomp)


class Budget(Noise):
    """GWINC Budget class

    This is a Noise that represents a budget of multiple sub noises.

    The `noises` attribute of this class should list constituent Noise
    classes.  Each element can be either a single Noise class, or a
    tuple of (Noise, Calibration) classes, e.g.:

    noises = [
        Thermal,
        (Shot, Sensing),
    ]

    When this object is initialized, all sub noises and calibrations
    are initialized.  Pre-defined load() and update() methods call the
    load() and update() methods of all sub noises and calibrations.
    When calc() is called, the PSD is calculated for all sub noises,
    the relevant calibration is evaluated and applied, and the
    quadrature sum of all calibrated consituent noises is returned.

    Additionally, a `calibrations` attribute may define a list of
    common calibrations to apply to all noises, e.g.:

    calibrations = [
        Strain,
    ]

    Finally, a `references` attribute may be defined, similar to the
    `noises` attribute described above except that the specified
    noises do not contribute to the overall budget total, e.g.:

    references = [
        strain_data_20200120,
    ]

    NOTE: if an `ifo` attribute is defined it is always passed as an
    initialization argument to sub noises.

    """

    noises = []
    """List of constituent noise classes, or (noise, cal) tuples"""

    calibrations = []
    """List of calibrations to be applied to all budget noises (not references)"""

    references = []
    """List of reference noise classes, or (ref, cal) tuples"""

    def __init__(self, freq=None, noises=None, **kwargs):
        """Initialize Budget object.

        See BudgetItem for base initialization arguments.

        If a `noises` keyword argument is provided it should be an
        iterable of noise names (constituent or reference) which will
        be used to filter the noises initialized in this budget.

        """
        super().__init__(freq, **kwargs)
        # store kwargs for later use
        self.kwargs = kwargs
        # record the frequency array as a kwarg if it's definied as a
        # class attribute
        if freq is not None:
            self.kwargs['freq'] = freq
        else:
            self.kwargs['freq'] = getattr(self, 'freq', None)
        # FIXME: special casing the ifo kwarg here, in case it's
        # defined as a class attribute rather than passed at
        # initialization.  we do this because we're not defining a
        # standard way to extract IFO variables that get passed around
        # in a reasonable way.  how can we clarify this?
        if 'ifo' not in kwargs and getattr(self, 'ifo', None):
            self.kwargs['ifo'] = getattr(self, 'ifo', None)
        # all noise objects keyed by name
        self._noise_objs = collections.OrderedDict()
        # all cal objects keyed by name
        self._cal_objs = {}
        # set of calibration names to apply to noise
        self._noise_cals = collections.defaultdict(set)
        # set of all constituent budget noise names
        self._budget_noises = set()
        # initialize all noise objects
        for nc in self.noises:
            name = self.__init_noise(nc, noises)
            if name:
                self._budget_noises.add(name)
        # initialize common calibrations and add to all budget noises
        for cal in self.calibrations:
            self.__add_calibration(cal, self._budget_noises)
        # initialize references, without common calibrations
        for nc in self.references:
            self.__init_noise(nc, noises)
        # error if requested noise is not present
        if noises:
            sset = set(noises)
            nset = set([name for name in self._noise_objs.keys()])
            if not sset <= nset:
                raise AttributeError("unknown noise terms: {}".format(' '.join(sset-nset)))

    def __init_noise(self, nc, noise_filt):
        cal = None
        if isinstance(nc, (list, tuple)):
            noise = nc[0]
            cals = nc[1:]
        else:
            noise = nc
            cals = []
        noise_obj = noise(
            **self.kwargs
        )
        name = noise_obj.name
        if noise_filt and name not in noise_filt:
            return
        logger.debug("init {}".format(noise_obj))
        self._noise_objs[name] = noise_obj
        for cal in cals:
            self.__add_calibration(cal, [name])
        return name

    def __add_calibration(self, cal, noises):
        cal_obj = cal(
            **self.kwargs
        )
        name = cal_obj.name
        if name not in self._cal_objs:
            logger.debug("init {}".format(cal_obj))
            self._cal_objs[name] = cal_obj
        # register noises for this calibration
        for noise in noises:
            self._noise_cals[noise].add(name)
        return name

    def __getitem__(self, name):
        try:
            return self._noise_objs[name]
        except KeyError:
            try:
                return self._cal_objs[name]
            except KeyError:
                raise KeyError("unknown noise or cal name '{}".format(name))

    def keys(self):
        """Iterate over budget noise names."""
        return iter(self._noise_objs.keys())

    def values(self):
        """Iterate over budget noise objects."""
        return iter(self._noise_objs.values())

    def items(self):
        """Iterate over budget noise (name, object) tuples."""
        return iter(self._noise_objs.items())

    def __iter__(self):
        return iter(self.keys())

    def walk(self):
        """Walk recursively through every BudgetItem in the budget.

        This includes Noise, Calibration and Budget objects, as well
        as any decendents of Budget objects.

        For each leaf item yields a tuple of all ancestor objects,
        e.g.:

          (self)
          (self, BudgetItem)
          (self, ChildBudget1)
          (self, ChildBudget1, BudgetItem)
          ...

        """
        yield (self,)
        for item in itertools.chain(
                self._cal_objs.values(),
                self._noise_objs.values()):
            if isinstance(item, Budget):
                for i in item.walk():
                    yield (self,) + i
            else:
                yield (self, item)

    def load(self):
        """Load all noise and cal objects."""
        for name, item in itertools.chain(
                self._cal_objs.items(),
                self._noise_objs.items()):
            logger.debug("load {}".format(item))
            item.load()

    def update(self, _precomp=None, **kwargs):
        """Recursively update all noise and cal objects with supplied kwargs.

        See BudgetItem.update() for more info.

        """
        for key, val in kwargs.items():
            setattr(self, key, val)

        if _precomp is None:
            _precomp = dict()

        for name, item in itertools.chain(
                self._cal_objs.items(),
                self._noise_objs.items()):
            logger.debug("update {}".format(item))
            item.update(_precomp=_precomp, **kwargs)

    def calc_noise(self, name, calibration=1, calc=True, _cals=None, _precomp = None):
        """Return calibrated individual noise BudgetTrace.

        The noise and calibration transfer functions are calculated,
        and the calibrated noise BudgetTrace is returned.
        `calibration` is an overall calculated calibration to apply to
        the noise.

        """
        if _precomp is None:
            _precomp = dict()
        for cal in self._noise_cals[name]:
            if _cals:
                calibration *= _cals[cal]
            else:
                obj = self._cal_objs[cal]
                logger.debug("calc {}".format(obj))
                PCmap = _precomp_recurse_mapping(obj.calc, self.freq, self.ifo, _precomp)
                calibration *= obj.calc(**PCmap)
        noise = self._noise_objs[name]
        logger.debug("calc {}".format(noise))
        return noise.calc_trace(
            calibration=calibration,
            calc=calc,
            _precomp=_precomp,
        )

    def calc_trace(self, calibration=1, calc=True, _precomp=None):
        """Calculate all budget noises and return BudgetTrace object

        `calibration` should either be a scalar or a len(self.freq)
        array that will be multiplied to the output PSD of the budget
        and all sub noises.

        If `calc` is False, the noise will not be calculated and the
        trace PSD will be None.  This is useful for just getting the
        trace style info.

        """
        _cals = {}
        if _precomp is None:
            _precomp = dict()
        if calc:
            for name, cal in self._cal_objs.items():
                logger.debug("calc {}".format(cal))
                PCmap = _precomp_recurse_mapping(cal.calc, self.freq, self.ifo, _precomp)
                _cals[name] = cal.calc(**PCmap)

        budget = []
        for name in self._noise_objs:
            trace = self.calc_noise(
                name,
                calibration=calibration,
                calc=calc,
                _cals=_cals,
                _precomp=_precomp,
            )
            budget.append(trace)

        total = quadsum([trace.psd for trace in budget])
        return self._make_trace(
            psd=total, budget=budget
        )
