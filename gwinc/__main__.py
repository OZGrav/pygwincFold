from __future__ import print_function
import os
import signal
import logging
import argparse

from . import (
    __version__,
    IFOS,
    DEFAULT_FREQ,
    InvalidFrequencySpec,
    load_budget,
    logger,
)
from . import io

logger.setLevel(os.getenv('LOG_LEVEL', 'WARNING').upper())
formatter = logging.Formatter('%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

##################################################

description = """GWINC noise budget tool

IFOs can be specified by name of included canonical budget (see
below), or by path to a budget module (.py), description file
(.yaml/.mat/.m), or HDF5 data file (.hdf5/.h5).  Available included
IFOs are:

  {}

""".format(', '.join(["{}".format(ifo) for ifo in IFOS]))
# for ifo in available_ifos():
#     description += "  '{}'\n".format(ifo)
description += """

By default the noise budget of the specified IFO will be loaded and
plotted with an interactive plotter.  Individual IFO parameters can be
overriden with the --ifo option:

  gwinc --ifo Optics.SRM.Tunephase=3.14 ...

If the --save option is specified the plot will be saved directly to a
file (without display) (various file formats are supported, indicated
by file extension).  If the requested extension is 'hdf5' or 'h5' then
the noise traces and IFO parameters will be saved to an HDF5 file.

If the --range option is specified and the inspiral_range package is
available, various BNS (m1=m2=1.4 M_solar) range figures of merit will
be calculated for the resultant spectrum.  The default waveform
parameters can be overriden with the --waveform-parameter/-wp option:

  gwinc -r -wp m1=20 -wp m2=20 ...

See the inspiral_range package documentation for details.
"""

IFO = 'aLIGO'
RANGE_PARAMS = dict(m1=1.4, m2=1.4)
DATA_SAVE_FORMATS = ['.hdf5', '.h5']

parser = argparse.ArgumentParser(
    prog='gwinc',
    description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    '--version', '-v', action='version', version=__version__)
parser.add_argument(
    '--freq', '-f', metavar='FLO:[NPOINTS:]FHI',
    help="logarithmic frequency array specification in Hz [{}]".format(DEFAULT_FREQ))
parser.add_argument(
    '--ifo', '-o', metavar='PARAM=VAL', default=[],
    #nargs='+', action='extend',
    action='append',
    help="override budget IFO parameter (may be specified multiple times)")
parser.add_argument(
    '--title', '-t',
    help="plot title")
parser.add_argument(
    '--range', '-r', action='store_true',
    help="calculate inspiral ranges [m1=m2=1.4]")
parser.add_argument(
    '--waveform-parameter', '-wp', metavar='PARAM=VAL', default=[],
    action='append',
    help="specify inspiral range parameters (may be specified multiple times)")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    '--interactive', '-i', action='store_true',
    help="launch interactive shell after budget processing")
group.add_argument(
    '--save', '-s', metavar='PATH', action='append',
    help="save plot (.png/.pdf/.svg) or budget traces (.hdf5/.h5) to file (may be specified multiple times)")
group.add_argument(
    '--yaml', '-y', action='store_true',
    help="print IFO as yaml to stdout and exit (budget not calculated)")
group.add_argument(
    '--text', '-x', action='store_true',
    help="print IFO as text table to stdout and exit (budget not calculated)")
group.add_argument(
    '--diff', '-d', metavar='IFO',
    help="show difference table between IFO and another IFO description (name or path) and exit (budget not calculated)")
group.add_argument(
    '--list', '-l', action='store_true',
    help="list all elements of Budget (budget not calculated)")
parser.add_argument(
    '--no-plot', '-np', action='store_false', dest='plot',
    help="suppress plotting")
parser.add_argument(
    '--bname', '-b',
    help="name of top-level Budget class to load (defaults to IFO name)")
parser.add_argument(
    'IFO',
    help="IFO name or path")
parser.add_argument(
    'subbudget', metavar='SUBBUDGET', nargs='?',
    help="subbudget to plot; can be nested (e.g. 'Thermal.Substrate')")


def main():
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    args = parser.parse_args()

    ##########
    # initial arg processing

    if os.path.splitext(os.path.basename(args.IFO))[1] in io.DATA_SAVE_FORMATS:
        if args.freq:
            parser.exit(2, "Error: Frequency specification not allowed when loading traces from file.\n")
        if args.ifo:
            parser.exit(2, "Error: IFO parameter specification not allowed when loading traces from file.\n")
        from .io import load_hdf5
        budget = None
        name = args.IFO
        trace = load_hdf5(args.IFO)
        freq = trace.freq
        ifo = trace.ifo
        plot_style = trace.plot_style

    else:
        try:
            budget = load_budget(args.IFO, freq=args.freq, bname=args.bname)
        except InvalidFrequencySpec as e:
            parser.error(e)
        except RuntimeError as e:
            parser.exit(2, f"Error: {e}\n")
        name = budget.name
        ifo = budget.ifo
        freq = budget.freq
        plot_style = getattr(budget, 'plot_style', {})
        trace = None

    for paramval in args.ifo:
        try:
            param, val = paramval.split('=', 1)
            ifo[param] = float(val)
        except ValueError:
            parser.error(f"Improper IFO parameter specification: {paramval}")

    if args.yaml:
        if not ifo:
            parser.exit(2, "Error: IFO structure not provided.\n")
        print(ifo.to_yaml(), end='')
        return
    if args.text:
        if not ifo:
            parser.exit(2, "Error: IFO structure not provided.\n")
        print(ifo.to_txt(), end='')
        return
    if args.diff:
        if not ifo:
            parser.exit(2, "Error: IFO structure not provided.\n")
        dbudget = load_budget(args.diff)
        diffs = ifo.diff(dbudget.ifo)
        if diffs:
            w = max([len(d[0]) for d in diffs])
            fmt = '{{:{}}} {{:>20}} {{:>20}}'.format(w)
            print(fmt.format('', args.IFO, args.diff))
            print(fmt.format('', '-----', '-----'))
            for p in diffs:
                k = str(p[0])
                v = repr(p[1])
                ov = repr(p[2])
                print(fmt.format(k, v, ov))
        return
    if args.list:
        for i in budget.walk():
            name = '.'.join([n.__class__.__name__ for n in i])
            type = i[-1].__class__.__bases__[0].__name__
            print(f'{name} ({type})')
        return

    if args.subbudget:
        try:
            budget[args.subbudget]
        except KeyError:
            parser.exit(3, f"Error: Unknown budget item '{args.subbudget}'.\n")

    out_data_files = set()
    out_plot_files = set()
    if args.save:
        args.plot = False
        out_files = set(args.save)
        for path in out_files:
            if os.path.splitext(path)[1] in io.DATA_SAVE_FORMATS:
                out_data_files.add(path)
        out_plot_files = out_files - out_data_files

    if args.plot or out_plot_files:
        if out_plot_files:
            # FIXME: this silliness seems to be the only way to have
            # matplotlib usable on systems without a display.  There must
            # be a better way.  'AGG' is a backend that works without
            # displays.  but it has to be set before any other matplotlib
            # stuff is imported.  and we *don't* want it set if we do want
            # to show an interactive plot.  there doesn't seem a way to
            # set this opportunistically.
            import matplotlib
            matplotlib.use('AGG')
        try:
            from matplotlib import pyplot as plt
        except ImportError as e:
            parser.exit(5, f"ImportError: {e}\n")
        except RuntimeError:
            parser.exit(10, "Error: Could not open display for plotting.\n")

    if args.range:
        try:
            import inspiral_range
        except ImportError as e:
            parser.exit(5, f"ImportError: {e}\n")

        logger_ir = logging.getLogger('inspiral_range')
        logger_ir.setLevel(logger.getEffectiveLevel())
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logger_ir.addHandler(handler)

        for paramval in args.waveform_parameter:
            try:
                param, val = paramval.split('=')
                if not val:
                    raise ValueError
            except ValueError:
                parser.error(f"Improper range parameter specification: {paramval}")
            try:
                val = float(val)
            except ValueError:
                pass
            RANGE_PARAMS[param] = val

    ##########
    # main calculations

    if not trace:
        logger.info("calculating budget...")
        trace = budget.run()

    if args.range:
        logger.info("calculating inspiral ranges...")
        metrics, H = inspiral_range.all_ranges(freq, trace.psd, **RANGE_PARAMS)
        print(f"{H.params['approximant']} {H.params['m1']}/{H.params['m2']} M_solar:")
        for metric, (value, unit) in metrics.items():
            if unit is None:
                unit = ''
            print(f" {metric}: {value:0.1f} {unit}")
        range_func = 'range'
        subtitle = 'inspiral {func} {m1}/{m2} $\mathrm{{M}}_\odot$: {fom:.0f} {unit}'.format(
            func=range_func,
            m1=H.params['m1'],
            m2=H.params['m2'],
            fom=metrics[range_func][0],
            unit=metrics[range_func][1] or '',
        )
    else:
        subtitle = None

    if args.subbudget:
        trace = trace[args.subbudget]
        name += f': {args.subbudget}'

    if args.title:
        plot_style['title'] = args.title
    else:
        plot_style['title'] = "GWINC Noise Budget: {}".format(name)

    ##########
    # interactive

    # interactive shell plotting
    if args.interactive:
        banner = """GWINC interactive shell

The 'ifo' Struct, 'budget', and 'trace' objects are available for
inspection.  Use the 'whos' command to view the workspace.
"""
        if not args.plot:
            banner += """
You may plot the budget using the 'trace.plot()' method:

In [.]: trace.plot(**plot_style)
"""
        banner += """
You may interact with the plot using the 'plt' functions, e.g.:

In [.]: plt.title("foo")
In [.]: plt.savefig("foo.pdf")
"""
        from IPython.terminal.embed import InteractiveShellEmbed
        if subtitle:
            plot_style['title'] += '\n' + subtitle
        ipshell = InteractiveShellEmbed(
            banner1=banner,
            user_ns={
                'ifo': ifo,
                'budget': budget,
                'trace': trace,
                'plot_style': plot_style,
            },
        )
        ipshell.enable_pylab(import_all=False)
        if args.plot:
            ipshell.ex("fig = trace.plot(**plot_style)")
        ipshell()

    ##########
    # output

    # save noise trace to HDF5 file
    if out_data_files:
        for path in out_data_files:
            logger.info("saving budget trace: {}".format(path))
            io.save_hdf5(
                trace=trace,
                path=path,
                ifo=ifo,
                plot_style=plot_style,
            )

    # standard plotting
    if args.plot or out_plot_files:
        logger.debug("plotting noises...")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if subtitle:
            plot_style['title'] += '\n' + subtitle
        trace.plot(
            ax=ax,
            **plot_style
        )
        fig.tight_layout()
        if out_plot_files:
            for path in out_plot_files:
                logger.info("saving budget plot: {}".format(path))
                try:
                    fig.savefig(path)
                except Exception as e:
                    parser.exit(2, f"Error saving plot: {e}.\n")
        else:
            plt.show()


if __name__ == '__main__':
    main()
