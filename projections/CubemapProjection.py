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


from . import Projection
from typing import Union, Tuple
import numpy as np
from . import coordinate_conversion


class CubemapProjection(Projection):
    """
    Cubemap Projection (CMP)
    """

    class CoordMap:
        def __init__(self, interval_2d, interval_3d, axis_3d):
            if axis_3d not in ('x', 'y', 'z'):
                raise Exception("Unknown 3d axis '{}'".format(axis_3d))
            self.interval_2d = interval_2d
            self.interval_3d = interval_3d
            self.axis_3d = axis_3d

        def val3d_for_2d(self, val2d):
            return self.lerp(val2d, self.interval_2d[0], self.interval_2d[1], self.interval_3d[0], self.interval_3d[1])

        def val2d_for_3d(self, val3d):
            return self.lerp(val3d, self.interval_3d[0], self.interval_3d[1], self.interval_2d[0], self.interval_2d[1])

        @staticmethod
        def lerp(val, a1, b1, a2, b2):
            return (val - a1) * ((b2 - a2) / (b1 - a1)) + a2

    class Region:
        def __init__(self, x_map: 'CubemapProjection.CoordMap',
                     y_map: 'CubemapProjection.CoordMap',
                     plane_val_3d: float):
            """
            :param x_map: coordinate map from 2d x-axis to corresponding 3d axis
            :param y_map: coordinate map from 2d y-axis to corresponding 3d axis
            :param plane_val_3d: coordinate value for image plane on 3d plane axis.
            """
            if x_map.axis_3d == y_map.axis_3d:
                raise Exception("2d coordinates x and y are mapped to same 3d axis '{}'.".format(x_map.axis_3d))
            self.x_map = x_map
            self.y_map = y_map
            self.plane_axis_3d = next(filter(lambda axis_3d: axis_3d not in (x_map.axis_3d, y_map.axis_3d),
                                             ('x', 'y', 'z')))
            self.plane_val_3d = plane_val_3d

        def within_region_mask(self, y, x):
            return (y >= min(self.y_map.interval_2d)) & (y <= max(self.y_map.interval_2d)) & \
                   (x >= min(self.x_map.interval_2d)) & (x <= max(self.x_map.interval_2d))

        def within_cubeface_mask(self, xs, ys, zs):
            plane_ax_idx = self._ax_to_idx(self.plane_axis_3d)
            other_ax_idxs = (self._ax_to_idx(self.x_map.axis_3d), self._ax_to_idx(self.y_map.axis_3d))
            vals = (xs, ys, zs)
            abs_vals = (np.abs(xs), np.abs(ys), np.abs(zs))
            if np.sign(self.plane_val_3d) > 0:
                mask = vals[plane_ax_idx] > 0
            else:
                mask = vals[plane_ax_idx] < 0
            mask = mask & (abs_vals[plane_ax_idx] > abs_vals[other_ax_idxs[0]]) & \
                   (abs_vals[plane_ax_idx] > abs_vals[other_ax_idxs[1]])
            return mask

        def __contains__(self, item):
            y = item[0]
            x = item[1]
            return np.all(self.within_region_mask(y, x))

        @staticmethod
        def _ax_to_idx(ax):
            return ('x', 'y', 'z').index(ax)

        def to_3d(self, y, x):
            if (y, x) not in self:
                raise Exception("Not all requested y, x values lie within the region.")
            shape = y.shape
            y, x = np.asarray(y).reshape(-1), np.asarray(x).reshape(-1)
            coords_3d = np.empty((3, y.shape[0]))
            coords_3d[self._ax_to_idx(self.y_map.axis_3d)] = self.y_map.val3d_for_2d(y)
            coords_3d[self._ax_to_idx(self.x_map.axis_3d)] = self.x_map.val3d_for_2d(x)
            coords_3d[self._ax_to_idx(self.plane_axis_3d)] = self.plane_val_3d
            return coords_3d.reshape(np.append(3, shape))

        def to_2d(self, xs, ys, zs):
            if not np.all(self.within_cubeface_mask(xs, ys, zs)):
                raise Exception("Not all requested xs, ys, zs values lie within the cubeface.")
            shape = xs.shape
            xs, ys, zs = xs.reshape(-1), ys.reshape(-1), zs.reshape(-1)
            r, theta, phi = coordinate_conversion.cartesian_to_spherical(xs, ys, zs)
            if self.plane_axis_3d == 'x':
                r = self.plane_val_3d / (np.sin(theta) * np.cos(phi))
            elif self.plane_axis_3d == 'y':
                r = self.plane_val_3d / (np.sin(theta) * np.sin(phi))
            elif self.plane_axis_3d == 'z':
                r = self.plane_val_3d / np.cos(theta)
            xs, ys, zs = coordinate_conversion.spherical_to_cartesian(r, theta, phi)
            coords_3d = (xs, ys, zs)

            coords_2d = np.empty((2, xs.shape[0]))
            coords_2d[0] = self.y_map.val2d_for_3d(coords_3d[self._ax_to_idx(self.y_map.axis_3d)])
            coords_2d[1] = self.x_map.val2d_for_3d(coords_3d[self._ax_to_idx(self.x_map.axis_3d)])
            return coords_2d.reshape(np.append(2, shape))

    def __init__(self, size):
        self._size = size
        Region = CubemapProjection.Region
        CoordMap = CubemapProjection.CoordMap

        self.top_region = Region(
            x_map=CoordMap((0, 1/3), (-1, 1), 'x'),
            y_map=CoordMap((0.5, 1), (-1, 1), 'y'),
            plane_val_3d=1
        )
        self.left_region = Region(
            x_map=CoordMap((0, 1/3), (1, -1), 'x'),
            y_map=CoordMap((0, 0.5), (1, -1), 'z'),
            plane_val_3d=-1
        )
        self.front_region = Region(
            x_map=CoordMap((1/3, 2/3), (-1, 1), 'y'),
            y_map=CoordMap((0, 0.5), (1, -1), 'z'),
            plane_val_3d=-1
        )
        self.right_region = Region(
            x_map=CoordMap((2/3, 1), (-1, 1), 'x'),
            y_map=CoordMap((0, 0.5), (1, -1), 'z'),
            plane_val_3d=1
        )
        self.back_region = Region(
            x_map=CoordMap((1/3, 2/3), (1, -1), 'z'),
            y_map=CoordMap((0.5, 1), (-1, 1), 'y'),
            plane_val_3d=1
        )
        self.bottom_region = Region(
            x_map=CoordMap((2/3, 1), (1, -1), 'x'),
            y_map=CoordMap((0.5, 1), (-1, 1), 'y'),
            plane_val_3d=-1
        )

    @property
    def focal_length(self):
        return 1 / np.tan(np.pi / self._size[0])

    @property
    def size(self):
        return self._size

    def to_sphere(self,
                  y: Union[float, np.ndarray],
                  x: Union[float, np.ndarray],
                  *args, **kwargs) -> Union[Tuple[float, float, float],
                                            Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        y = (y+0.5) / self._size[0]
        x = (x+0.5) / self._size[1]

        coords_3d = np.ones((3, y.shape[0], y.shape[1])) * np.nan
        for region in (self.top_region, self.left_region, self.front_region, self.right_region,
                       self.back_region, self.bottom_region):
            mask = region.within_region_mask(y, x)
            coords_3d[:, mask] = region.to_3d(y[mask], x[mask])

        isvalid = ~(np.isnan(coords_3d[0]) | np.isnan(coords_3d[1]) | np.isnan(coords_3d[2]))
        coords_3d[:, isvalid] = coords_3d[:, isvalid] / np.linalg.norm(coords_3d[:, isvalid], axis=0)
        return coords_3d[0], coords_3d[1], coords_3d[2]

    def from_sphere(self,
                    xs: Union[float, np.ndarray],
                    ys: Union[float, np.ndarray],
                    zs: Union[float, np.ndarray]) -> Union[Tuple[float, float],
                                                           Tuple[np.ndarray, np.ndarray]]:
        coords_2d = np.ones((2, xs.shape[0], xs.shape[1])) * np.nan
        for region in (self.top_region, self.left_region, self.front_region, self.right_region,
                       self.back_region, self.bottom_region):
            mask = region.within_cubeface_mask(xs, ys, zs)
            coords_2d[:, mask] = region.to_2d(xs[mask], ys[mask], zs[mask])

        coords_2d[0] = (coords_2d[0] * self._size[0]) - 0.5
        coords_2d[1] = (coords_2d[1] * self._size[1]) - 0.5
        return coords_2d[0], coords_2d[1]
