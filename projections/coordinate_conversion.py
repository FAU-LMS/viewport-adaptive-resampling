# BSD 3-Clause License
#
# Copyright (c) 2023, Friedrich-Alexander-Universität Erlangen-Nürnberg.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from typing import Union, Tuple


def cartesian_to_polar(
        y: Union[float, np.ndarray],
        x: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert cartesian coordinates [y, x] to polar coordinates [r, phi]."""
    r = np.sqrt(np.square(y) + np.square(x))
    phi = np.arctan2(y, x)
    return r, phi


def polar_to_cartesian(
        r: Union[float, np.ndarray],
        phi: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert polar coordinates [r, phi] to cartesian coordinates [y, x]."""
    y = r * np.sin(phi)
    x = r * np.cos(phi)
    return y, x


def cartesian_to_spherical(
        x: Union[float, np.ndarray],
        y: Union[float, np.ndarray],
        z: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert cartesian coordinates [x, y, z] to spherical coordinates [r, theta, phi]."""
    r = np.sqrt(np.square(x) + np.square(y) + np.square(z))
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(
        r: Union[float, np.ndarray],
        theta: Union[float, np.ndarray],
        phi: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Convert spherical coordinates [r, theta, phi] to cartesian coordinates [x, y, z]."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z
