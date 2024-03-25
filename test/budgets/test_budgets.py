"""
"""
import numpy as np
import gwinc
from gwinc import load_budget
import gwinc.io as io
from copy import deepcopy
from gwinc.ifo.noises import dhdl
import matplotlib.pyplot as plt
import pytest


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_load(ifo, pprint, tpath_join, fpath_join):
    B = load_budget(ifo)
    trace = B.run()
    fig = trace.plot()
    fig.savefig(tpath_join('budget_{}.pdf'.format(ifo)))


@pytest.mark.generate
@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_save_budgets(ifo, fpath_join):
    B = load_budget(ifo)
    traces = B.run()
    io.save_hdf5(traces, fpath_join(ifo + '.h5'))


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_check_noise(ifo, fpath_join, compare_noise):
    try:
        ref_traces = io.load_hdf5(fpath_join(ifo + '.h5'))
    except OSError:
        return
    budget = load_budget(ifo, freq=ref_traces.freq)
    traces = budget.run()
    compare_noise(traces, ref_traces)


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_calibrations(ifo, tpath_join):
    """Test that the standard calibrations are correct"""
    from gwinc.noise.coatingthermal import mirror_struct

    cals = ["Displacement", "Velocity", "Acceleration", "Force"]
    budgets = gwinc.Struct({cal: load_budget(ifo, bname=cal) for cal in cals})
    budgets["Strain"] = load_budget(ifo)
    traces = gwinc.Struct({cal: budget.run() for cal, budget in budgets.items()})
    freq = traces.Strain.freq
    dhdl_sqr, _ = dhdl(freq, budgets.Strain.ifo.Infrastructure.Length)
    disp = traces.Strain.asd / np.sqrt(dhdl_sqr)

    for power, (cal, trace) in enumerate(traces.items()):
        if cal == "Strain":
            continue
        # reference is the correct value
        reference = disp * (2 * np.pi * freq)**power
        if cal == "Force":
            mass = mirror_struct(budgets.Strain.ifo, "ETM").MirrorMass
            reference = disp * mass * (2 * np.pi * freq)**2
        fig = trace.plot()
        fig.gca().loglog(freq, reference, ls="--", c="xkcd:baby blue", lw=3)
        fig.savefig(tpath_join(f"Total{cal}.pdf"))
        assert np.allclose(trace.psd, reference**2, atol=0)


@pytest.mark.parametrize("ifo", gwinc.IFOS)
def test_sub_budgets(ifo, tpath_join):
    B = load_budget(ifo)
    traces = B.run()
    fig = traces.plot()
    fig.savefig(tpath_join('Total.pdf'))
    for trace in traces:
        if trace.budget:
            fig = trace.plot()
            fig.savefig(tpath_join(trace.name + '.pdf'))


@pytest.mark.logic
def test_budget_run_calc(tpath_join, pprint, compare_noise):
    """
    Tests that
      1) budget.run() calculates the entire budget including calibration
      2) budget.calc_noise(name) calculates the name sub-budget including calibration
      3) budget[name].calc_trace() calculates the name sub-budget not including calibration
      4) Both 1) and 2) are still true when called directly on sub-budgets
      5) Both budget[name].calc_trace(sub_trace) and budget[name][sub_trace].calc_trace()
         calculate the sub_trace trace of the name sub-budget without the calibration
         of the main budget
      6) budget[name].run() is the same as budget[name].calc_trace()
    """
    F_Hz = np.logspace(np.log10(5), 4, 3000)
    B = load_budget('CE2silica', freq=F_Hz)
    traces1 = B.run()
    traces2 = B.calc_noise('Quantum')
    traces3 = B['Quantum'].calc_trace()
    traces4 = B['Quantum'].calc_noise('AS')
    traces5 = B['Quantum']['AS'].calc_trace()
    traces6 = B['Quantum'].run()
    fig1 = traces1.Quantum.plot()
    fig2 = traces2.plot()
    fig3 = traces3.plot()
    fig1.savefig(tpath_join('run.pdf'))
    fig2.savefig(tpath_join('calc_noise.pdf'))
    fig3.savefig(tpath_join('calc_trace.pdf'))

    pprint('Testing that run() and calc_noise() do the same thing')
    compare_noise(traces1.Quantum, traces2)
    compare_noise(traces1.Quantum.AS, traces2.AS)
    pprint(
        "Testing that run() and calc_trace() on sub-budgets don't apply strain calibration")
    compare_noise(traces3, traces6)

    def update_total_psd(traces):
        total_psd = np.sum([trace.psd for trace in traces], axis=0)
        traces._psd = total_psd

    # divide displacement noise by arm length for ease of plotting
    Larm_m = B.ifo.Infrastructure.Length
    dhdl_sqr, sinc_sqr = dhdl(B.freq, Larm_m)
    for trace in traces3:
        trace._psd /= Larm_m**2
    update_total_psd(traces3)

    fig, ax = plt.subplots()
    ax.loglog(F_Hz, traces1.Quantum.asd, label='run')
    ax.loglog(F_Hz, traces2.asd, ls='--', label='calc_noise')
    ax.loglog(F_Hz, traces3.asd, ls='-.', label='calc_trace / $L_\mathrm{arm}$')

    # convert displacement noise into strain
    for trace in traces3:
        trace._psd *= sinc_sqr
    update_total_psd(traces3)
    ax.loglog(
        F_Hz, traces3.asd, ls=':',
        label='calc_trace / $(\mathrm{d}h/\mathrm{d}L_\mathrm{arm})$')

    ax.set_xlim(F_Hz[0], F_Hz[-1])
    ax.grid(True, which='major', alpha=0.5)
    ax.grid(True, which='minor', alpha=0.2)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Strain noise [1/Hz$^{-1/2}$]')
    ax.legend()
    fig.savefig(tpath_join('comparison.pdf'))

    pprint('Testing that calc_trace() is the same as calc_noise() without calibration')
    compare_noise(traces2, traces3)
    pprint('Testing that calc_noise() and calc_trace() on sub-budgets are right')
    compare_noise(traces4, traces5)


@pytest.mark.logic
def test_budget_cals_refs(fpath_join, tpath_join, compare_noise):
    """
    Test calibration specifications and reference plotting

    Tests the (Noise, Calibration) calculations not used in the canonical budgets
    as well as specification of reference traces not included in budget
    """
    F_Hz = np.logspace(1, 4, 1000)
    budget = load_budget(fpath_join('H1'), freq=F_Hz)
    budget_no_ref = load_budget(fpath_join('H1'), freq=F_Hz, bname='H1NoRefs')
    traces = budget.run()
    traces_no_ref = budget_no_ref.run()
    compare_noise(traces_no_ref, traces)

    fig = traces.plot()
    fig_no_ref = traces_no_ref.plot()
    fig.savefig(tpath_join('budget.pdf'))
    fig_no_ref.savefig(tpath_join('budget_no_ref.pdf'))


@pytest.mark.logic
def test_budget_dict_attributes(fpath_join, tpath_join):
    """
    Test that noises and references can be specified by dicts or Structs
    """
    import H1
    F_Hz = np.logspace(1, 4, 1000)
    budget = load_budget(fpath_join('H1'))
    budget_dict0 = H1.H1dict(freq=F_Hz, ifo=budget.ifo)
    H1.H1dict.noises['Shot'] = (H1.Shot, H1.SensingOpticalSpring)
    H1.H1dict.references.Reference = H1.DARMMeasuredO3
    budget_dict1 = H1.H1dict(freq=F_Hz, ifo=budget.ifo)
    traces_dict0 = budget_dict0.run()
    traces_dict1 = budget_dict1.run()

    fig_dict0 = traces_dict0.plot()
    fig_dict0.savefig(tpath_join('budget_dict0.pdf'))
    fig_dict1 = traces_dict1.plot()
    fig_dict1.savefig(tpath_join('budget_dict1.pdf'))


@pytest.mark.logic
def test_forward_noises(fpath_join, tpath_join, compare_noise):
    B = load_budget(fpath_join('H1'), bname='H1NoRefs')
    B_frwd = load_budget(fpath_join('H1'), bname='H1NoRefsForwardNoises')
    B_dict = load_budget(fpath_join('H1'), bname='H1dictNoRefs')
    B_dict_frwd = load_budget(
        fpath_join('H1'), bname='H1dictNoRefsForwardNoises')

    tr = B.run()
    tr_frwd = B_frwd.run()
    tr_dict = B_dict.run()
    tr_dict_frwd = B_dict_frwd.run()

    fig = tr.plot()
    fig.savefig(tpath_join('budget.pdf'))
    fig_frwd = tr_frwd.plot()
    fig_frwd.savefig(tpath_join('budget_frwd.pdf'))

    fig_dict = tr_dict.plot()
    fig_dict.savefig(tpath_join('budget_dict.pdf'))
    fig_dict_frwd = tr_dict_frwd.plot()
    fig_dict_frwd.savefig(tpath_join('budget_dict_frwd.pdf'))

    compare_noise(tr.Thermal.SuspensionThermal, tr_frwd.SuspensionThermal)
    compare_noise(tr_dict.ThermalDict.SuspensionThermal, tr_dict_frwd.SuspensionThermal)
    compare_noise(tr, tr_dict)
    compare_noise(tr_frwd, tr_dict_frwd)


@pytest.mark.logic
@pytest.mark.fast
def test_update_ifo_struct():
    """
    Test that the noise is recalculated when the ifo struct is updated
    """
    budget = gwinc.load_budget('CE2silica')
    tr1 = budget.run()
    budget.ifo.Suspension.VHCoupling.theta *= 2
    tr2 = budget.run()
    assert np.all(
        tr2.Seismic.Vertical.asd == 2*tr1.Seismic.Vertical.asd)


@pytest.mark.logic
@pytest.mark.fast
def test_change_ifo_struct():
    """
    Test that the noise is recalculated when a new ifo struct is passed to run
    """
    budget = gwinc.load_budget('CE2silica')
    ifo1 = deepcopy(budget.ifo)
    ifo2 = deepcopy(budget.ifo)
    ifo2.Suspension.VHCoupling.theta *= 2
    tr1 = budget.run(ifo=ifo1)
    tr2 = budget.run(ifo=ifo2)
    tr3 = budget.run(ifo=ifo1)
    assert np.all(tr1.asd == tr3.asd)
    assert np.all(
        tr2.Seismic.Vertical.asd == 2*tr1.Seismic.Vertical.asd)


@pytest.mark.logic
@pytest.mark.fast
def test_update_freq():
    """
    Test three methods of updating a Budget frequency
    """
    freq1 = np.logspace(1, 3, 10)
    freq2 = np.logspace(0.8, 3.5, 11)
    freq3 = np.logspace(0.5, 3.6, 12)
    budget = gwinc.load_budget('Aplus', freq=freq1)
    traces1 = budget.run()
    traces2 = budget.run(freq=freq2)
    budget.freq = freq3
    traces3 = budget.run()
    assert np.all(traces1.freq == freq1)
    assert np.all(traces2.freq == freq2)
    assert np.all(traces3.freq == freq3)


@pytest.mark.logic
@pytest.mark.fast
def test_freq_spec_error():
    """
    Test that three methods of setting Budget frequencies raise errors
    """
    freq = [1, 2, 3]
    with pytest.raises(gwinc.InvalidFrequencySpec):
        budget = gwinc.load_budget('Aplus', freq=freq)
    with pytest.raises(AssertionError):
        budget = gwinc.load_budget('Aplus')
        traces = budget.run(freq=freq)
    with pytest.raises(AssertionError):
        budget = gwinc.load_budget('Aplus')
        budget.freq = freq

