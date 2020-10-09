'''Functions to calculate suspension thermal noise

'''
from __future__ import division
import numpy as np
from numpy import pi, imag

from ..const import kB
from ..suspension import getJointParams


def suspension_thermal(f, sus, sustf):
    """Suspension thermal noise.

    :f: frequency array in Hz
    :sus: gwinc suspension structure
    :sustf: sus transfer function Struct

    :returns: displacement noise power spectrum at :f:

    :Temp: must either be set for each stage individually, or globally
    in :sus.Temp:.  If both are specified, :sus.Temp: is interpreted as
    the temperature of the upper joint of the top stage.

    Assumes suspension transfer functions and V-H coupling have been
    pre-calculated and populated into the relevant `sus` fields.

    """
    # suspension TFs
    hForce = sustf.hForce
    vForce = sustf.vForce
    # and vertical to beamline coupling angle
    theta = sustf.VHCoupling.theta

    noise = np.zeros_like(f)

    # Here the suspension stage list is reversed so that the last stage
    # in the list is the test mass
    stages = sus.Stage[::-1]
    w = 2*pi*f
    dxdF = np.zeros_like(hForce)  # complex valued
    for n, stage in enumerate(stages):
        # add up the contribution from each joint
        jointParams = getJointParams(sus, n)
        for (isLower, Temp, wireMat, bladeMat) in jointParams:
            # convert to beam line motion.  theta is squared because
            # we rotate by theta into the suspension basis, and by
            # theta to rotate back to the beam line basis
            m = 2*n + isLower
            dxdF[m, :] = hForce[m, :] + theta**2 * vForce[m, :]
            noise += 4 * kB * Temp * abs(imag(dxdF[m, :])) / w

    # thermal noise (m^2/Hz) for one suspension
    return noise
