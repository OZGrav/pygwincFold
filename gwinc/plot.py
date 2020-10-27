def plot_trace(
        trace,
        ax=None,
        **kwargs
):
    """Plot a GWINC BudgetTrace noise budget from calculated noises

    If an axis handle is provided it will be used for the plot.

    Returns the figure handle.

    """
    if ax is None:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    else:
        fig = ax.figure

    total = trace.asd
    ylim = [min(total)/10, max(total)]
    style = dict(
        color='#000000',
        alpha=0.6,
        lw=4,
    )
    style.update(getattr(trace, 'style', {}))
    if 'label' in style:
        style['label'] = 'Total ' + style['label']
    else:
        style['label'] = 'Total'
    ax.loglog(trace.freq, total, **style)

    for name, strace in trace.items():
        style = strace.style
        if 'label' not in style:
            style['label'] = name
        if 'linewidth' in style:
            style['lw'] = style['linewidth']
        elif 'lw' not in style:
            style['lw'] = 3
        ax.loglog(trace.freq, strace.asd, **style)

    ax.grid(
        True,
        which='both',
        lw=0.5,
        ls='-',
        alpha=0.5,
    )

    ax.legend(
        ncol=2,
        fontsize='small',
    )

    ax.autoscale(enable=True, axis='y', tight=True)
    ax.set_ylim(kwargs.get('ylim', ylim))
    ax.set_xlim(trace.freq[0], trace.freq[-1])
    ax.set_xlabel('Frequency [Hz]')
    if 'ylabel' in kwargs:
        ax.set_ylabel(kwargs['ylabel'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    return fig


# FIXME: deprecate
plot_noise = plot_trace
plot_budget = plot_trace
