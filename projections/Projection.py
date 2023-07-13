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


from abc import ABC, abstractmethod
from typing import Union, Tuple
import numpy as np


class Projection(ABC):
    """
    Provides forward and backward projection methods.
    """

    @property
    @abstractmethod
    def focal_length(self):
        """
        Get the focal length.
        """
        pass

    @abstractmethod
    def to_sphere(self,
                  y: Union[float, np.ndarray],
                  x: Union[float, np.ndarray],
                  *args, **kwargs) -> Union[Tuple[float, float, float],
                                            Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Project coordinates to unit sphere.
        :param y: y coordinates (scalar or array)
        :param x: x coordinates (scalar or array)
        :return: corresponding coordinates on unit sphere (xs, ys, zs)
        """
        pass

    @abstractmethod
    def from_sphere(self,
                    xs: Union[float, np.ndarray],
                    ys: Union[float, np.ndarray],
                    zs: Union[float, np.ndarray]) -> Union[Tuple[float, float],
                                                           Tuple[np.ndarray, np.ndarray]]:
        """
        Reproject coordinates from unit sphere.
        :param xs: x coordinates (scalar or array)
        :param ys: y coordinates (scalar or array)
        :param zs: z coordinates (scalar or array)
        :return: reprojected coordinates (y, x)
        """
        pass
