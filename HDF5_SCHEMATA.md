# HDF5 Schema for GWINC noise trace storage

This file describes a schemata for HDF5 storage of noise trace data
and plot styling GWINC noise budgets.  Python functions for writing
budget data to and reading budget data from this format are included
in the `gwinc.io` module.

HDF5 is a heirarchical, structured data storage format [0].  Content
is organized into a heirarchical folder-like structure, with two
types of named objects:

 * groups: holder of other objects (like folders)
 * datasets: holder of data arrays (like files)

Objects can also have attributes as (almost) arbitrary key:value
pairs.

Bindings are available for most platforms including Python [1] and
Matlab.

 * [0] https://en.wikipedia.org/wiki/Hierarchical_Data_Format
 * [1] http://www.h5py.org/


## schema

The following describes the noise budget schema.  Specific strings are
enclosed in single quotes (''), and variables are described in
brackets (<>).  Group objects are indicated by a closing '/', data
sets are indicated by a closing ':' followed by a specification of
their length and type (e.g. "(N),float"), and attributes are specified
in the .attrs[] dictionary format.  Optional elements are enclosed in
parentheses.


## top-level attributes

The following two root attributes expected: a string describing the schema,
and an int schema version number:
```
/.attrs['SCHEMA'] = 'GWINC Noise Budget'
/.attrs['SCHEMA_VERSION'] = 1
```

The following root root attributes are defined:
```
/.attrs['title'] = <experiment description string (e.g. 'H1 Strain Budget')>
/.attrs['date'] = <ISO-formatted string (e.g. '2015-10-24T20:30:00.000000Z')>
/.attrs['ifo'] = <IFO Struct object, YAML serialized>
```

The remaining root level attributes usually pertain to the plot style.

The budget frequency array is defined in a top level 'Freq' dataset:

```
/'Freq': (N),float
```

## version history

### v1

A single trace is a length N array (with optional plot style specified
in attributes:
```
/<trace>: (N),float
/<trace>.attrs['label'] = <label>
/<trace>.attrs['color] = <color>
...
```

A budget item, i.e. a collection of noises is structured as follows:
```
/<budget>/
/<budget>/Total': (N),float
/<budget>/<trace_0>: (N),float
/<budget>/...
```


The budget traces are defined a traces group.  The overall structure
looks something like this:
```
/traces/
/traces/'Total': (N),float
/traces/<noise_0>: (N),float
/traces/<noise_1>: (N),float
/traces/<noise_2>/
/traces/<noise_2>/'Total': (N),float
/traces/<noise_2>/<noise_3>: (N),float
/traces/<noise_2>/...
/traces/...
```


### v2

Each trace is given the following structure:
```
/<trace>/
/<trace>.attrs['style'] = <YAML trace style dict>
/<trace>/'PSD': (N),float
/<trace>/budget/
/<trace>/budget/<subtrace_0>/
/<trace>/budget/<subtrace_1>/
/<trace>/budget/...
```

The overall structure is:
```
/<budget>/
/<budget>/.attrs['plot_style'] = <YAML plot style dict>
/<budget>/.attrs['style'] = <YAML "Total" trace style dict>
/<budget>/.attrs[*] = <arbitrary data>
/<budget>/'Freq': (N),float
/<budget>/'PSD': (N),float
/<budget>/budget/
/<budget>/budget/<trace_0>/...
/<budget>/budget/<trace_1>/...
/<budget>/budget/...
```
