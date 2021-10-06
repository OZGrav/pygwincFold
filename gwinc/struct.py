"""
"""

from collections.abc import Mapping, Sequence, MutableSequence
from numbers import Number

import os
import re
import io
import yaml

import numpy as np
from scipy.io import loadmat
from scipy.io.matlab.mio5_params import mat_struct


class Struct(object):
    """Matlab struct-like object

    This is a simple implementation of a MATLAB struct-like object
    that stores values as attributes of a simple class: and allows
    assigning to attributes recursively, e.g.:

    >>> s = Struct()
    >>> s.a = 4
    >>> s.b = Struct()
    >>> s.b.c = 8

    Various classmethods allow creating one of these objects from YAML
    file, a nested dict, or a MATLAB struct object.

    """
    STRUCT_EXT = ['.yaml', '.yml', '.mat', '.m']
    """accepted extension types for struct files"""

    # FIXME: There should be a way to allow setting nested struct
    # attributes, e.g.:
    #
    # >>> s = Struct()
    # >>> s.a.b.c = 4
    #
    # Usage of __getattr__ like this is dangerous and creates
    # non-intuitive behavior (i.e. an empty struct is returned when
    # accessing attributes that don't exist).  Is there a way to
    # accomplish this without that adverse side affect?
    #
    # def __getattr__(self, name):
    #     if name not in self.__dict__:
    #         self.__dict__[name] = Struct()
    #     return self.__dict__[name]

    ##########

    def __init__(self, *args, **kwargs):
        """Initialize Struct object

        Initializes similar to dict(), taking a single dict or mapping
        argument, or keyword arguments to initially populate the
        Struct.

        """
        # TODO, should this use the more or less permissive allow_unknown_types?
        self.update(dict(*args, **kwargs), allow_unknown_types=True)

    def __getitem__(self, key):
        """Get a (possibly nested) value from the struct.

        """
        if '.' in key:
            k, r = key.split('.', 1)
            # FIXME: this is inelegant.  better done with regexp?
            if len(k.split('[')) > 1:
                kl, i = k.split('[')
                i = int(i.strip(']'))
                return self.__dict__[kl][i][r]
            return self.__dict__[k][r]
        else:
            return self.__dict__[key]

    def get(self, key, default=None):
        """Get a (possibly nested) value from the struct, or default.

        """
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key, value):
        if '.' in key:
            k, r = key.split('.', 1)
            self.__dict__[k][r] = value
        else:
            self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def setdefault(self, key, default):
        return self.__dict__.setdefault(key, default)

    def update(
            self, other,
            overwrite_atoms=False,
            clear_test=lambda v: False,
            allow_unknown_types=True,
    ):
        """Update Struct from other Struct or dict.
        This is *recursive* and will also update using lists, performing a
        deepcopy of the dict/list structure. It inspects the internal types to
        do this.

        None's are not inserted and are always overwritten.

        If a value of other returns true on clear_test(value) then that value is
        cleared in the updated self. override the argument
        clear_test=lambda v: v is None to clear null values.

        overwrite_atoms is an boolean argument which allows overwriting values
        with different types, like converting a float to a Struct. It defaults
        to False to disallow this often confusing behavior, but is sometimes
        necessary.

        allow_unknown_types allows updates and assignments of types other than
        Structs, Lists, floats, strings, and numpy arrays of elements. It
        defaults to True to be permissive.
        """
        kw = dict(
            overwrite_atoms=overwrite_atoms,
            clear_test=clear_test,
            allow_unknown_types=allow_unknown_types,
        )

        def update_element(self, k, other_v,):
            """
            type dispatch that assigns into self[k] based on the current type
            and the type of other_v
            """
            self_v = self[k]
            if clear_test(other_v):
                if isinstance(self, Mapping):
                    del self[k]
                else:
                    raise StructTypingError("clear_test deletions not allowed in sequences like lists")
            elif other_v is None:
                # don't update on None
                pass
            elif isinstance(other_v, VALUE_TYPES):
                # other is a value type, not a collection
                if isinstance(self_v, VALUE_TYPES):
                    self[k] = other_v
                elif isinstance(self_v, (Sequence, Mapping)):
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a value into a dict or list)")
                else:
                    if not allow_unknown_types:
                        raise StructTypingError("Unknown type assignment during recursive .update()")
                    else:
                        self[k] = other_v
            elif isinstance(other_v, Mapping):
                if isinstance(self_v, VALUE_TYPES):
                    if not overwrite_atoms:
                        raise StructTypingError("struct update is an incompatible storage type (e.g. updating a dict into a float)")
                    else:
                        self_v = self[k] = Struct()
                        self_v.update(other_v, **kw)
                elif isinstance(self_v, Sequence):
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a dict into a list)")
                elif isinstance(self_v, Mapping):
                    self[k].update(other_v, **kw)
                elif self_v is None:
                    self_v = self[k] = Struct()
                    self_v.update(other_v, **kw)
                else:
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a dict into a list)")
            elif isinstance(other_v, Sequence):
                # this check MUST come after VALUE_TYPES, or string is included

                # make mutable
                if not isinstance(self_v, MutableSequence):
                    self_v = list(self_v)

                if isinstance(self_v, VALUE_TYPES):
                    if not overwrite_atoms:
                        raise StructTypingError("struct update is an incompatible storage type (e.g. updating a dict into a string)")
                    else:
                        self_v = self[k] = other_v
                elif isinstance(self_v, Sequence):
                    # the string check MUST come before Sequence
                    list_update(self_v, other_v)
                elif isinstance(self_v, Mapping):
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a list into a dict)")
                elif self_v is None:
                    self_v = self[k] = other_v
                else:
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a value into a list)")
            else:
                # other is an unknown value type, not a collection
                if not allow_unknown_types:
                    raise StructTypingError("Unknown type assigned during recursive .update()")

                if isinstance(self_v, (Sequence, Mapping)):
                    raise StructTypingError("struct update is an incompatible storage type (e.g. updating a value into a dict or list)")
                else:
                    self[k] = other_v
            return

        def list_update(self_v, other_v,):
            """
            helper function for the recursive update
            """
            N_min = min(len(self_v), len(other_v))
            # make self as long as other, filled with None's so that assignment occurs
            self_v.extend([None] * (len(other_v) - N_min))
            for idx, sub_other_v in enumerate(other_v):
                update_element(self_v, idx, sub_other_v)
            return

        # actual code loop for the recursive update
        for k, other_v in other.items():
            if k in self:
                update_element(self, k, other_v)
            else:
                # k not in self, so just assign
                if clear_test(other_v):
                    pass
                elif isinstance(other_v, VALUE_TYPES):
                    # value type to directly assign
                    self[k] = other_v
                elif isinstance(other_v, Mapping):
                    self_v = self[k] = Struct()
                    # use update so that it is a deepcopy
                    self_v.update(other_v, **kw)
                elif isinstance(other_v, Sequence):
                    # MUST come after the value types check, or strings included
                    self_v = self[k] = []
                    list_update(self_v, other_v)
                else:
                    if not allow_unknown_types:
                        raise StructTypingError("Unknown type assigned during recursive .update()")
                    # value type to directly assign
                    self[k] = other_v

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def __contains__(self, key):
        return key in self.__dict__

    def to_dict(self, array=False):
        """Return nested dictionary representation of Struct.

        If `array` is True, any lists encountered will be turned into
        numpy arrays, and lists of Structs will be turned into record
        arrays.  This is needed to convert to structure arrays in
        matlab.

        """
        d = {}
        for k, v in self.__dict__.items():
            if k[0] == '_':
                continue
            if isinstance(v, Struct):
                d[k] = v.to_dict(array=array)
            else:
                if isinstance(v, list):
                    try:
                        # this should fail if the elements of v are
                        # not Struct
                        # FIXME: need cleaner way to do this
                        v = [i.to_dict(array=array) for i in v]
                        if array:
                            v = dictlist2recarray(v)
                    except AttributeError:
                        if array:
                            v = np.array(v)
                # FIXME: there must be a better way to just match all
                # numeric scalar types
                elif isinstance(v, Number):
                    v = float(v)
                d[k] = v
        return d

    def to_yaml(self, path=None):
        """Return YAML representation of Struct.

        Write YAML to `path` if specified.

        """
        y = yaml.dump(self.to_dict(), default_flow_style=False)
        if path:
            with open(path, 'w') as f:
                f.write(y)
        else:
            return y

    def __str__(self):
        return '<GWINC Struct: {}>'.format(list(self.__dict__.keys()))

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.__dict__)

    def _repr_pretty_(self, s, cycle):
        """
        This is the pretty print extension function for IPython's pretty printer
        """
        if cycle:
            s.text('GWINC Struct(...)')
            return
        s.begin_group(8, 'Struct({')
        for idx, (k, v) in enumerate(self.items()):
            s.pretty(k)
            s.text(': ')
            s.pretty(v)
            if idx + 1 < len(self):
                s.text(',')
                s.breakable()
        s.end_group(8, '})')
        return

    def __iter__(self):
        return iter(self.__dict__)

    def walk(self):
        """Iterate over all leaves in the struct tree.

        """
        for k, v in self.__dict__.items():
            if k[0] == '_':
                continue
            if isinstance(v, (dict, Struct)):
                for sk, sv in v.walk():
                    yield k + '.' + sk, sv
            else:
                try:
                    for i, vv in enumerate(v):
                        if isinstance(vv, dict):
                            vv = Struct(vv)
                        for sk, sv in vv.walk():
                            yield '{}[{}].{}'.format(k, i, sk), sv
                except (AttributeError, TypeError):
                    yield k, v

    def hash(self, keys=None):
        """Hash of Struct.walk() data (recursive)

        """
        def filter_keys(kv):
            k, v = kv
            if keys:
                return k in keys
            else:
                return True

        def map_tuple(kv):
            k, v = kv
            if isinstance(v, list):
                return k, tuple(v)
            else:
                return k, v

        return hash(tuple(sorted(
            map(map_tuple, filter(filter_keys, self.walk()))
        )))

    def diff(self, other):
        """Return tuple of differences between target IFO.

        Returns list of (key, value, other_value) tuples.  Value is
        None if key not present.

        Note: yaml also supports putting None into dictionaries by not supplying
        a value. The None values returned here and the "missing value" None's
        from yaml are not distinguished in this diff
        """
        diffs = []

        # this is just a funky structure that will be guaranteed unique
        UNIQUE = (lambda x: None, )

        if isinstance(other, dict):
            other = Struct(other)
        for k, ov in other.walk():
            try:
                v = self.get(k, UNIQUE)
                if ov != v and ov is not v:
                    if v is UNIQUE:
                        diffs.append((k, None, ov))
                    else:
                        diffs.append((k, v, ov))
            except TypeError:
                # sometimes the deep keys go through unmappable objects
                # which TypeError if indexed
                diffs.append((k, None, ov))
        for k, v in self.walk():
            try:
                ov = other.get(k, UNIQUE)
                if ov is UNIQUE:
                    diffs.append((k, v, None))
            except TypeError:
                diffs.append((k, v, None))
        return diffs

    def __eq__(self, other):
        """True if structs have all equal values"""
        return not bool(self.diff(other))

    def to_txt(self, path=None, fmt='0.6e', delimiter=': ', end=''):
        """Return text represenation of Struct, one element per line.

        Struct keys use '.' to indicate hierarchy.  The `fmt` keyword
        controls the formatting of numeric values.  MATLAB code can be
        generated with the following parameters:

        >>> ifo.to_txt(delimiter=' = ', end=';')

        Write text to `path` if specified.

        """
        txt = io.StringIO()
        for k, v in sorted(self.walk()):
            if isinstance(v, (int, float, complex)):
                base = fmt
            elif isinstance(v, (list, np.ndarray)):
                if isinstance(v, list):
                    v = np.array(v)
                v = np.array2string(
                    v,
                    separator='',
                    max_line_width=np.Inf,
                    formatter={'all': lambda x: "{:0.6e} ".format(x)}
                )
                base = 's'
            else:
                base = 's'
            txt.write(u'{key}{delimiter}{value:{base}}{end}\n'.format(
                key=k, value=v, base=base,
                delimiter=delimiter,
                end=end,
            ))
        if path:
            with open(path, 'w') as f:
                f.write(txt.getvalue())
        else:
            return txt.getvalue()

    @classmethod
    def from_yaml(cls, y):
        """Create Struct from YAML string.

        """
        d = yaml.load(y, Loader=YAML_LOADER) or {}
        return cls(d)

    @classmethod
    def from_matstruct(cls, s):
        """Create Struct from scipy.io.matlab mat_struct object.

        """
        c = cls()
        try:
            s = s['ifo']
        except Exception:
            pass
        for k, v in s.__dict__.items():
            if k in ['_fieldnames']:
                # skip these fields
                pass
            elif type(v) is mat_struct:
                c.__dict__[k] = Struct.from_matstruct(v)
            else:
                # handle lists of Structs
                try:
                    c.__dict__[k] = list(map(Struct.from_matstruct, v))
                except Exception:
                    c.__dict__[k] = v
                    # try:
                    #     c.__dict__[k] = float(v)
                    # except:
                    #     c.__dict__[k] = v
        return c

    @classmethod
    def from_file(cls, path, _pass_inherit=False):
        """Load Struct from .yaml or MATLAB .mat file.

        Accepted file types are .yaml, .mat, or .m.

        For .m files, the file is expected to include either an object
        or function that corresponds to the basename of the file.  The
        MATLAB engine will be invoked to execute the .m code and
        extract the resultant IFO data.

        If `path` is a tuple, all elements will be joined ala
        os.path.join, with the first element resolved to it's absolute
        dirname.  This is useful for loading package-relative files
        with e.g.:

          Struct.from_file((__file__, 'myifo.yaml'))

        the _pass_inherit is a special key enabling the "+inherit" feature from
        load_budget. It will change in future versions as that functionality is
        moved here.
        """
        if type(path) == tuple:
            path = os.path.join(os.path.abspath(os.path.dirname(path[0])), *path[1:])
        base, ext = os.path.splitext(path)

        if ext == '.m':
            from .gwinc_matlab import Matlab
            matlab = Matlab()
            matlab.addpath(os.path.dirname(path))
            func_name = os.path.basename(base)
            matlab.eval("ifo = {};".format(func_name), nargout=0)
            ifo = matlab.extract('ifo')
            val = Struct.from_matstruct(ifo)

        else:
            with open(path, 'r') as f:
                if ext in ['.yaml', '.yml']:
                    val = cls.from_yaml(f)
                elif ext == '.mat':
                    s = loadmat(f, squeeze_me=True, struct_as_record=False)
                    val = cls.from_matstruct(s)
                else:
                    raise IOError("Unknown file type: {}".format(ext))

        # now include code it enable or disable "+inherit" at this stage of loading
        # this will modify in place if +inherit loading is active
        def recurse_value(v):
            if isinstance(v, VALUE_TYPES):
                pass
            elif isinstance(v, Sequence):
                recurse_sequence(v)
            elif isinstance(v, Mapping):
                recurse_mapping(v)
            else:
                pass

        def recurse_mapping(dct):
            for k, v in dct.items():
                if k == '+inherit':
                    raise RuntimeError('The +inherit key is not supported (yet) using from_file')
                recurse_value(v)

        def recurse_sequence(lst):
            for v in lst:
                recurse_value(v)

        # the bottom loop is special cased, as currently +inherit is only allowed at root level
        for k, v in val.items():
            if k == '+inherit':
                if _pass_inherit:
                    pass
                else:
                    raise RuntimeError('The +inherit key is not supported (yet) using from_file')
                continue
            recurse_value(v)
        return val


# these are the leaf types for struct that are most officially supported
# str must be included to distinguish it from Sequences like lists.
VALUE_TYPES = (str, Number, np.ndarray),


def dictlist2recarray(lst):
    def dtype(v):
        if isinstance(v, int):
            return float
        else:
            return type(v)
    # get dtypes from first element dict
    dtypes = [(k, dtype(v)) for k, v in lst[0].items()]
    values = [tuple(el.values()) for el in lst]
    out = np.array(values, dtype=dtypes)
    return out.view(np.recarray)


# HACK: fix loading Number in scientific notation
#
# https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
#
# An apparent bug in python-yaml prevents it from recognizing
# scientific notation as a float.  The following is a modified version
# of the parser that recognize scientific notation appropriately.
YAML_LOADER = yaml.SafeLoader
YAML_LOADER.add_implicit_resolver(
    'tag:yaml.org,2002:float',
    re.compile('''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list('-+0123456789.'))


class StructTypingError(ValueError):
    """
    Exception class for specific errors related to typing during
    a recursive update to Struct.
    """
    pass


# add to the registry of mappings, as it is a useful way to write code to normalize between dicts and structs
Mapping.register(Struct)
