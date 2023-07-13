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


from typing import Tuple, Union
import numpy as np
from . import Projection
from . import coordinate_conversion


class EquirectangularProjection(Projection):

    def __init__(self, size: Union[Tuple[int, int], np.ndarray]):
        self._size = size

    @property
    def focal_length(self):
        return 1 / np.tan(np.pi/self._size[0])

    @property
    def size(self):
        return self._size

    def to_sphere(self,
                  y: Union[float, np.ndarray],
                  x: Union[float, np.ndarray],
                  *args, **kwargs) -> Union[Tuple[float, float, float],
                                            Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        phi_s = -((x + 0.5) / self._size[1]) * 2 * np.pi
        theta_s = ((y + 0.5) / self._size[0]) * np.pi
        xs, ys, zs = coordinate_conversion.spherical_to_cartesian(1, theta_s, phi_s)
        return xs, ys, zs

    def from_sphere(self,
                    xs: Union[float, np.ndarray],
                    ys: Union[float, np.ndarray],
                    zs: Union[float, np.ndarray]) -> Union[Tuple[float, float],
                                                           Tuple[np.ndarray, np.ndarray]]:
        _, theta_s, phi_s = coordinate_conversion.cartesian_to_spherical(xs, ys, zs)
        phi_s = np.where(phi_s > 0, phi_s - 2 * np.pi, phi_s)
        y = (theta_s / np.pi) * self._size[0] - 0.5
        x = -(phi_s / (2 * np.pi)) * self._size[1] - 0.5
        return y, x
