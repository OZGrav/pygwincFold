import numpy as np


class BudgetTrace:
    """Budget trace data calculated from a Noise or Budget


    """
    def __init__(self, name=None, style=None, freq=None, psd=None, budget=None):
        self.name = name
        self.style = style or {}
        self._freq = freq
        self._psd = psd
        self.budget = budget or []

    def __repr__(self):
        if self.budget:
            bs = ' [{}]'.format(', '.join([str(b.name) for b in self]))
        else:
            bs = ''
        return '<{} {}{}>'.format(
            self.__class__.__name__,
            self.name,
            bs,
        )

    @property
    def freq(self):
        """trace frequency array, in Hz"""
        return self._freq

    @property
    def psd(self):
        """trace power spectral density array"""
        return self._psd

    @property
    def asd(self):
        """trace amplitude spectral density array"""
        return np.sqrt(self._psd)

    def len(self):
        """Length of data"""
        return len(self._freq)

    def __iter__(self):
        """iterator of budget traces"""
        return iter(self.budget)

    @property
    def _bdict(self):
        bdict = {trace.name: trace for trace in self.budget}
        return bdict

    def __getattr__(self, name):
        return self._bdict[name]

    def __getitem__(self, name):
        """get budget trace by name"""
        return self._bdict[name]

    def items(self):
        """iterator of budget (name, trace) tuples"""
        return self._bdict.items()

    def keys(self):
        """iterator of budget trace names"""
        return self._bdict.keys()

    def values(self):
        """iterator of budget traces"""
        return self._bdict.values()

    def walk(self):
        """walk recursively through all traces"""
        yield self.name, self
        for trace in self:
            for t in trace.walk():
                yield t
