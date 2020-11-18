'''Functions to calculate suspension thermal noise

'''
from __future__ import division
import numpy as np
from numpy import pi, imag

from ..const import kB
from ..suspension import getJointParams


def getJointTempSusceptibility(sus, sustf, stage_num, isUpper, direction):
    """Get the product of the temperature and the susceptibility of a joint

    :sus: the suspension struct
    :sustf: suspension transfer function struct
    :stage_num: stage number with 0 at the top
    :isUpper: whether to get the upper or lower joint
    :direction: either 'horiz' or 'vert'
    """
    jointParams = getJointParams(sus, stage_num)
    for (isLower, Temp, wireMat, bladeMat) in jointParams:
        if isUpper == isLower:
            continue

        ind = 2*stage_num + isLower
        # Compute the force on the TM along the beamline
        if direction == 'horiz':
            dxdF = sustf.hForce[ind]

        elif direction == 'vert':
            # vertical to beamline coupling angle
            theta = sustf.VHCoupling.theta

            # convert to beam line motion. theta is squared because
            # we rotate by theta into the suspension basis, and by
            # theta to rotate back to the beam line basis
            dxdF = theta**2 * sustf.vForce[ind]

        return Temp * abs(imag(dxdF))


def susptherm_stage(f, sus, sustf, stage_num, direction):
    """Suspension thermal noise of a single stage

    :f: frequency array in Hz
    :sus: gwinc suspension struct
    :sustf: suspension transfer function struct
    :stage_num: stage number with 0 at the top
    :direction: either 'horiz' for horizontal or 'vert' for vertical
    :comp: component of the loss 0) bulk, 1) surface, 2) TE
    :returns: displacement noise power spectrum at :f:

    :Temp: must either be set for each stage individually, or globally
    in :sus.Temp:.  If both are specified, :sus.Temp: is interpreted as
    the temperature of the upper joint of the top stage.

    Assumes suspension transfer functions and V-H coupling have been
    pre-calculated and populated into the relevant `sus` fields.

    """

    w = 2*pi*f
    last_stage = len(sus.Stage) - 1

    # get the noise from the lower joint
    noise = getJointTempSusceptibility(sus, sustf, stage_num, False, direction)

    # if this is the top mass, also get the noise from the upper joint
    if stage_num == 0:
        noise += getJointTempSusceptibility(sus, sustf, 0, True, direction)

    # if this isn't the test mass, also get the noise from the upper joint of
    # the mass below this one
    if stage_num < last_stage:
        noise += getJointTempSusceptibility(
            sus, sustf, stage_num + 1, True, direction)

    # thermal noise (m^2/Hz) for one suspension
    noise = 4 * kB * noise / w

    return noise


def suspension_thermal(f, sus, sustf):
    """Total suspension thermal noise of one suspension

    :f: frequency array in Hz
    :sus: gwinc suspension structure
    :sustf: suspension transfer function struct
    :returns: displacement noise power spectrum at :f:

    :Temp: must either be set for each stage individually, or globally
    in :sus.Temp:.  If both are specified, :sus.Temp: is interpreted as
    the temperature of the upper joint of the top stage.

    Assumes suspension transfer functions and V-H coupling have been
    pre-calculated and populated into the relevant `sus` fields.

    """
    nstage = len(sus.Stage)
    noise = np.zeros_like(f)
    for stage_num in range(nstage):
        noise += susptherm_stage(f, sus, sustf, stage_num, 'horiz')
        noise += susptherm_stage(f, sus, sustf, stage_num, 'vert')
    return abs(noise)
