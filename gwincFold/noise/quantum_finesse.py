import numpy as np
import copy
import scipy
from ..squeeze import computeFCParams
from ..suspension import precomp_suspension, precomp_suspension_f
from ..ifo.noises import ifo_power
from .. import nb
from vectfit3 import vectfit
from vectfit3 import opts
import finesse
from finesse import Model
from finesse.components import Laser, Mirror, Space, Beamsplitter,SuspensionZPK, SignalGenerator, Squeezer, Isolator, DirectionalBeamsplitter,DegreeOfFreedom
from finesse.detectors import PowerDetector, QuantumNoiseDetector


class Quantum(nb.Noise):
    """Quantum Vacuum

    """
    style = dict(
        label='Quantum Vacuum',
        color='#ad03de',
    )

    def calc(self):
        psd1, freq = quantum_noise(self.ifo)
        dhdl_sqr, sinc_sqr = dhdl(self.freq, self.ifo.Infrastructure.Length)
        psd_out = np.interp(self.freq, freq, psd1)/dhdl_sqr
        return psd_out



def tf_to_zpk(func, ifo):
    N = 10000
    n = 20
    start_f = 0.1
    stop_f = 500
    pc_sus = func(np.logspace(np.log10(start_f), np.log10(stop_f), N), ifo=ifo)
    f= pc_sus.tst_suscept      
    s = 2j*np.pi*np.logspace(np.log10(start_f), np.log10(stop_f),N,dtype=np.complex128)
    weights=np.ones(N,dtype=np.float64)  
    poles=-2*np.pi*np.logspace(np.log10(start_f), np.log10(stop_f ),n,dtype=np.complex128) 
    opts["skip_res"]=True  # Modified to skip residue computation during the iterative execution of vector fitting
    opts["spy2"]=False     # Modified to omit graphs generation into the iterative application of vectfit

    Niter=5 #number of iterations
    for itr in range(Niter):
        if itr==Niter-1:
            opts["spy2"]= False       #enabling graphs for the results
            opts["skip_res"]=False  #enabling residue computation in the final iteration
        (SER,poles,rmserr,fit)=vectfit(f,s,poles,weights,opts)
    out = scipy.signal.ss2zpk(SER["A"], SER["B"], SER["C"], SER["D"])
    return out

def quantum_noise(ifo):
    power = ifo_power(ifo, PRfixed=False)
    model = Model()

    zpk_plant_1 = {}
    zpk_plant_1['z', 'F_z'] = tf_to_zpk(precomp_suspension, ifo)

    zpk_plant_2 = {}
    zpk_plant_2['z', 'F_z'] = tf_to_zpk(precomp_suspension_f, ifo)

    l1 = model.add(Laser('L1', P=ifo.Laser.Power))
    prm = model.add(Mirror('PRM', T=power.Tpr, L=ifo.Optics.Loss, phi=90))
    bs = model.add(Beamsplitter('BS', T=0.5 - ifo.Optics.BSLoss/2, R=0.5 - ifo.Optics.BSLoss/2, alpha=45))
    itmx = model.add(Mirror('ITMX', T=ifo.Optics.ITM.Transmittance, L=ifo.Optics.Loss, phi=90))
    itmy = model.add(Mirror('ITMY', T=ifo.Optics.ITM.Transmittance, L=ifo.Optics.Loss, phi=0))
    etmx = model.add(Mirror('ETMX', T=ifo.Optics.ETM.Transmittance, L=ifo.Optics.Loss, phi=90-0.000125))          
    etmy = model.add(Mirror('ETMY', T=ifo.Optics.ETM.Transmittance, L=ifo.Optics.Loss, phi=0.000125)) 
    fmx = model.add(Beamsplitter('FMX', T=ifo.Optics.FM.Transmittance, L=ifo.Optics.Loss, alpha=ifo.Infrastructure.FoldAngle))
    fmy = model.add(Beamsplitter('FMY', T=ifo.Optics.FM.Transmittance, L=ifo.Optics.Loss, alpha=ifo.Infrastructure.FoldAngle))
    srm  = model.add(Mirror('SRM', T=ifo.Optics.SRM.Transmittance, L=ifo.Optics.Loss, phi=-90+ifo.Optics.SRM.Tunephase))
    out  = model.add(Mirror('out', R=0, L=(1-ifo.Optics.PhotoDetectorEfficiency)))

    OFI = model.add(DirectionalBeamsplitter('OFI'))

    sqz = model.add(Squeezer('sqz', db=10, angle=0))

    sqz3 = model.add(Space('sqz3', portA=sqz.p1, portB=OFI.p2))

    itmx_sus = model.add(SuspensionZPK('itmx_sus', itmx.mech, zpk_plant_1))
    itmy_sus = model.add(SuspensionZPK('itmy_sus', itmy.mech, zpk_plant_1))
    etmx_sus = model.add(SuspensionZPK('etmx_sus', etmx.mech, zpk_plant_1))
    etmy_sus = model.add(SuspensionZPK('etmy_sus', etmy.mech, zpk_plant_1))

    fmx_sus = model.add(SuspensionZPK('fmx_sus', fmx.mech, zpk_plant_2))
    fmy_sus = model.add(SuspensionZPK('fmy_sus', fmy.mech, zpk_plant_2))

    s1 = model.add(Space('s1', portA=l1.p1, portB=prm.p1, L=1,))
    prc = model.add(Space('prc', portA=prm.p2, portB=bs.p1, L=53))
    lx = model.add(Space('lx', portA=bs.p3, portB=itmx.p1, L=4.5))
    ly = model.add(Space('ly', portA=bs.p2, portB=itmy.p1, L=4.45))
    Lx1 = model.add(Space('lx1', portA=itmx.p2, portB=fmx.p1, L=ifo.Infrastructure.FoldedLength))
    Lx2 = model.add(Space('lx2', portA=fmx.p2, portB=etmx.p1, L=ifo.Infrastructure.FoldedLength))
    Ly1 = model.add(Space('ly1', portA=itmy.p2, portB=fmy.p1, L=ifo.Infrastructure.FoldedLength))
    Ly2 = model.add(Space('ly2', portA=fmy.p2, portB=etmy.p1, L=ifo.Infrastructure.FoldedLength))
    src = model.add(Space('src', portA=bs.p4, portB=srm.p1, L=ifo.Optics.SRM.CavityLength))
    srout = model.add(Space('srout', portA=srm.p2, portB=OFI.p1))
    srout3 = model.add(Space('srout3', portA=OFI.p3, portB=out.p1))
    model.fsig.f=1

    darmx1 = model.add(SignalGenerator('darmx1', Lx1.h))
    darmx2 = model.add(SignalGenerator('darmx2', Lx2.h))
    darmy1 = model.add(SignalGenerator('darmy1', Ly1.h, phase=180))
    darmy2 = model.add(SignalGenerator('darmy2', Ly2.h, phase=180))
    qdetector = model.add(QuantumNoiseDetector('QuantumNoise', out.p2.o, nsr=True))
    sol_opt_sqz = model.run("xaxis(fsig, log, 1, 6k, 1000, pre_step=minimize(QuantumNoise, sqz.angle, tol=1e-9))")
    psd = (np.abs(np.array(sol_opt_sqz[:], dtype=np.complex128()))*1)**2 
    return psd, np.logspace(np.log10(1), np.log10(6e3), 1001)


def dhdl(f, armlen):
    """Strain to length conversion for noise power spetra

    This takes into account the GW wavelength and is only important
    when this is comparable to the detector arm length.

    From R. Schilling, CQG 14 (1997) 1513-1519, equation 5,
    with n = 1, nu = 0.05, ignoring overall phase and cos(nu)^2.
    A small value of nu is used instead of zero to avoid infinities.

    Returns the square of the dh/dL function, and the same divided by
    the arm length squared.

    """
    c = 3e8
    nu_small = 15*np.pi/180
    omega_arm = np.pi * f * armlen / c
    omega_arm_f = (1 - np.sin(nu_small)) * np.pi * f * armlen / c
    omega_arm_b = (1 + np.sin(nu_small)) * np.pi * f * armlen / c
    sinc_sqr = 4 / np.abs(np.sin(omega_arm_f) * np.exp(-1j * omega_arm) / omega_arm_f
                       + np.sin(omega_arm_b) * np.exp(1j * omega_arm) / omega_arm_b)**2
    dhdl_sqr = sinc_sqr / armlen**2
    return dhdl_sqr, sinc_sqr
