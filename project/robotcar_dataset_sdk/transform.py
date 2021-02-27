################################################################################
#
# Copyright (c) 2017 University of Oxford
# Authors:
#  Geoff Pascoe (gmp@robots.ox.ac.uk)
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
#
################################################################################

import numpy as np
import numpy.matlib as matlib
from math import sin, cos, atan2, sqrt

MATRIX_MATCH_TOLERANCE = 1e-4


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    """
    if len(xyzrpy) != 6:
        raise ValueError("Must supply 6 values to build transform")

    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3

def build_se3_transform_new(xyzrpy):
    se3 = np.zeros((len(xyzrpy.index),16))
    # se3[:, [0:3,4:7,8:11]] = euler_to_so3_new(xyzrpy[3:])
    se3[:, [0,1,2,4,5,6,8,9,10]] = euler_to_so3_new(xyzrpy[3:])
    se3[:, [3,7,11]] = xyzrpy[:3]
    return se3


def euler_to_so3_new(rpy):
    sin_0, cos_0 = np.sin(rpy[0]), np.cos(rpy[0])
    sin_1, cos_1 = np.sin(rpy[1]), np.cos(rpy[1])
    sin_2, cos_2 = np.sin(rpy[2]), np.cos(rpy[2])
    
    R00 = cos_2 * cos_1
    R01 = -sin_2 * cos_0
    R02 = cos_2 * sin_1 * sin_0
    R10 = sin_2 * cos_1
    R11 = cos_2*cos_0 + sin_0*sin_1*sin_2
    R12 = -sin_0*cos_2 + cos_0*sin_1*sin_2
    R20 = -sin_1
    R21 = sin_0*cos_1
    R22 = cos_0*cos_1
    
    return np.stack([R00,R01,R02,R10,R11,R12,R20,R21,R22]).transpose()
    
def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx


def so3_to_euler(so3):
    """Converts an SO3 rotation matrix to Euler angles

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of Euler angles (size 3)

    Raises:
        ValueError: if so3 is not 3x3
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")
    roll = atan2(so3[2, 1], so3[2, 2])
    yaw = atan2(so3[1, 0], so3[0, 0])
    denom = sqrt(so3[0, 0] ** 2 + so3[1, 0] ** 2)
    pitch_poss = [atan2(-so3[2, 0], denom), atan2(-so3[2, 0], -denom)]

    R = euler_to_so3((roll, pitch_poss[0], yaw))

    if (so3 - R).sum() < MATRIX_MATCH_TOLERANCE:
        return np.matrix([roll, pitch_poss[0], yaw])
    else:
        R = euler_to_so3((roll, pitch_poss[1], yaw))
        if (so3 - R).sum() > MATRIX_MATCH_TOLERANCE:
            raise ValueError("Could not find valid pitch angle")
        return np.matrix([roll, pitch_poss[1], yaw])

def so3_to_quaternion_new(so3):
    R_xx = so3[:, 0]
    R_xy = so3[:, 1]
    R_xz = so3[:, 2]
    R_yx = so3[:, 3]
    R_yy = so3[:, 4]
    R_yz = so3[:, 5]
    R_zx = so3[:, 6]
    R_zy = so3[:, 7]
    R_zz = so3[:, 8]
    

    so3_trace = R_xx + R_yy + Y_zz + 1
    w=np.zeros(R_xx.shape[0],1)
    w = np.sqrt(so3_trace, where=so3_trace>=0) / 2
    
    x = 1 + R_xx - R_yy - R_zz
    x[x<0] = 0
    x = np.sqrt(x) / 2
    
    y = 1 + R_yy - R_xx - R_zz
    y[y<0] = 0
    y = np.sqrt(y) / 2
    
    z = 1 + R_zz - R_yy - R_xx
    z[z<0] = 0
    z = np.sqrt(z) / 2

    max_indexes = np.argmax(np.stack([w,x,y,z]), axis=1)
    max_0 = np.where(max_indexes==0)[0]
    max_1 = np.where(max_indexes==1)[0]
    max_2 = np.where(max_indexes==2)[0]
    max_3 = np.where(max_indexes==3)[0]
    
    x[max_0] = (R_zy[max_0] - R_yz[max_0]) / (4 * w[max_0])
    y[max_0] = (R_xz[max_0] - R_zx[max_0]) / (4 * w[max_0])
    z[max_0] = (R_yx[max_0] - R_xy[max_0]) / (4 * w[max_0])
    
    w[max_1] = (R_zy[max_1] - R_yz[max_1]) / (4 * x[max_1])
    y[max_1] = (R_xy[max_1] + R_yx[max_1]) / (4 * x[max_1])
    z[max_1] = (R_zx[max_1] + R_xz[max_1]) / (4 * x[max_1])
    
    w[max_2] = (R_xz[max_2] - R_zx[max_2]) / (4 * y[max_2])
    x[max_2] = (R_xy[max_2] + R_yx[max_2]) / (4 * y[max_2])
    z[max_2] = (R_yz[max_2] + R_zy[max_2]) / (4 * y[max_2])

    w[max_3] = (R_yx[max_3] - R_xy[max_3]) / (4 * z[max_3])
    x[max_3] = (R_zx[max_3] + R_xz[max_3]) / (4 * z[max_3])
    y[max_3] = (R_yz[max_3] + R_zy[max_3]) / (4 * z[max_3])
    
    return np.array([w, x, y, z])
    
    
def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    Args:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    # Due to numerical precision the value passed to `sqrt` may be a negative of the order 1e-15.
    # To avoid this error we clip these values to a minimum value of 0.
    x = sqrt(max(1 + R_xx - R_yy - R_zz, 0)) / 2
    y = sqrt(max(1 + R_yy - R_xx - R_zz, 0)) / 2
    z = sqrt(max(1 + R_zz - R_yy - R_xx, 0)) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def se3_to_components(se3):
    """Converts an SE3 rotation matrix to linear translation and Euler angles

    Args:
        se3: 4x4 transformation matrix

    Returns:
        numpy.matrixlib.defmatrix.matrix: list of [x, y, z, roll, pitch, yaw]

    Raises:
        ValueError: if se3 is not 4x4
        ValueError: if a valid Euler parametrisation cannot be found

    """
    if se3.shape != (4, 4):
        raise ValueError("SE3 transform must be a 4x4 matrix")
    xyzrpy = np.empty(6)
    xyzrpy[0:3] = se3[0:3, 3].transpose()
    xyzrpy[3:6] = so3_to_euler(se3[0:3, 0:3])
    return xyzrpy
