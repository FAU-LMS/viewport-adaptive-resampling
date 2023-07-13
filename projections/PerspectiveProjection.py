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


from . import RadialProjection
from typing import Optional, Union, Tuple
import numpy as np
from . import coordinate_conversion


class PerspectiveProjection(RadialProjection):
    """
    Defines the perspective projection model.
    """

    @property
    def max_fov(self):
        return np.pi

    def radius(self, theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return self.focal_length * np.tan(theta)

    def theta(self, radius: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return np.arctan(radius/self.focal_length)

    def to_sphere(self,
                  y: Union[float, np.ndarray],
                  x: Union[float, np.ndarray],
                  vip: Optional[Union[bool, np.ndarray]] = None,
                  *args, **kwargs) -> Union[Tuple[float, float, float],
                                            Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        r, phi_s = coordinate_conversion.cartesian_to_polar(y - self.optical_center[0],
                                                            x - self.optical_center[1])
        theta_s = self.theta(r)
        # Virtual Image Plane Compensation (VIPC)
        if vip is not None:
            phi_s = phi_s - vip * np.pi
            theta_s = theta_s - vip * (2 * theta_s - np.pi)
        xsr, ysr, zsr = coordinate_conversion.spherical_to_cartesian(1, theta_s, phi_s)
        xs = -zsr
        ys = xsr
        zs = -ysr
        return xs, ys, zs

    def from_sphere(self,
                    xs: Union[float, np.ndarray],
                    ys: Union[float, np.ndarray],
                    zs: Union[float, np.ndarray]) -> Union[Tuple[float, float, bool],
                                                           Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        _, theta_s, phi_s = coordinate_conversion.cartesian_to_spherical(ys, -zs, -xs)
        r = self.radius(theta_s)
        y, x = coordinate_conversion.polar_to_cartesian(r, phi_s)
        return y + self.optical_center[0], x + self.optical_center[1], r < 0
