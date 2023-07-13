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


from typing import Tuple, Callable
import numpy as np
from tqdm import tqdm
from projections import Projection, PerspectiveProjection


class ViewportAdaptiveResampler:

    def __init__(self,
                 size_src: Tuple[int, int],
                 projection_src: Projection,
                 size_tar: Tuple[int, int],
                 projection_tar: Projection,
                 mesh_to_mesh_resampler: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                 blocksize: int,
                 incident_angle_factor: float):
        """
        Initialize Viewport-Adaptive Resampler.

        :param size_src: size of image in source projection format
        :param projection_src: source projection
        :param size_tar: size of image in target projection format
        :param projection_tar: target projection
        :param mesh_to_mesh_resampler: mesh-to-mesh capable resampler as
               (sample_pos_src [N, 2], sample_val_src [N], sample_pos_tar [L, 2]) -> sample_val_tar [L]
        :param blocksize: block size used for resampling (default: 8)
        :param incident_angle_factor: incident angle factor for source sample range selection (default: 2)
        """
        self._size_src = size_src
        self._projection_src = projection_src
        self._size_tar = size_tar
        self._projection_tar = projection_tar
        self._mesh_to_mesh_resampler = mesh_to_mesh_resampler
        self._blocksize = blocksize
        self._incident_angle_factor = incident_angle_factor
        self._check_config()

        y_src, x_src = np.mgrid[:size_src[0], :size_src[1]]
        xs_src, ys_src, zs_src = self._projection_src.to_sphere(y_src, x_src)
        self._s_src_o = np.stack((xs_src, ys_src, zs_src))

        y_tar, x_tar = np.mgrid[:size_tar[0], :size_tar[1]]
        xs_tar, ys_tar, zs_tar = self._projection_tar.to_sphere(y_tar, x_tar)
        self._s_tar_o = np.stack((xs_tar, ys_tar, zs_tar))

        self._projection_perspective = PerspectiveProjection(self._projection_src.focal_length, (0, 0))

    def resample(self, image_src, progress=True):
        """
        Resample image with initialized configuration.

        :param image_src: image in source projection format
        :param progress: whether to show current progress during resampling (default: true)
        :return: image in target projection format
        """
        if image_src.shape != self._size_src:
            raise ValueError(f"Image size {image_src.shape} does not "
                             f"match configured size {self._size_src}.")
        blocks_i = self._size_tar[0] // self._blocksize
        blocks_j = self._size_tar[1] // self._blocksize
        image_tar = np.empty(self._size_tar)
        t = None
        if progress:
            t = tqdm(total=blocks_i * blocks_j)
        for i in range(blocks_i):
            for j in range(blocks_j):
                block_tar = self._resample_block(i, j, image_src)
                image_tar[i * self._blocksize:(i + 1) * self._blocksize,
                j * self._blocksize:(j + 1) * self._blocksize] = block_tar
                if progress:
                    t.update(1)
        if progress:
            t.close()
        return image_tar

    def _check_config(self):
        if self._size_tar[0] % self._blocksize != 0 or self._size_tar[1] % self._blocksize != 0:
            raise ValueError("Support for partial blocks not implemented.")

    def _resample_block(self, i: int, j: int, image_src: np.ndarray):
        # Extract target block coordinates
        s_blk_tar_o = self._s_tar_o[:,
                      i * self._blocksize:(i + 1) * self._blocksize,
                      j * self._blocksize:(j + 1) * self._blocksize]

        # Calculate rotation matrix based on block center
        y_tar_c, x_tar_c = np.array([(i + 0.5) * self._blocksize,
                                     (j + 0.5) * self._blocksize]) - 0.5
        xs_tar_c, ys_tar_c, zs_tar_c = self._projection_tar.to_sphere(np.array([[y_tar_c]]),
                                                                      np.array([[x_tar_c]]))
        rotation_matrix = self._rotation_matrix(xs_tar_c[0, 0], ys_tar_c[0, 0], zs_tar_c[0, 0])

        # Viewport rotation
        s_src_r = np.tensordot(rotation_matrix, self._s_src_o, axes=(1, 0))
        s_blk_tar_r = np.tensordot(rotation_matrix, s_blk_tar_o, axes=(1, 0))

        # Select reference samples
        max_theta = self._incident_angle_factor * np.max(np.arccos(-s_blk_tar_r[0]))
        if max_theta > np.pi / 2:
            raise Exception("Maximum incident angle > pi/2. This is not allowed.")
        theta_src = np.arccos(-s_src_r[0])
        mask_src = theta_src < max_theta
        s_masked_src_r = s_src_r[:, mask_src]
        image_masked_src = image_src[mask_src]

        # Project to perspective image plane
        y_src_p, x_src_p, _ = self._projection_perspective.from_sphere(s_masked_src_r[0],
                                                                       s_masked_src_r[1],
                                                                       s_masked_src_r[2])
        y_blk_tar_p, x_blk_tar_p, _ = self._projection_perspective.from_sphere(s_blk_tar_r[0],
                                                                               s_blk_tar_r[1],
                                                                               s_blk_tar_r[2])

        # Prepare memory layout for resampling
        p_src_p = np.stack((x_src_p, y_src_p), axis=1)
        p_blk_tar_p = np.stack((x_blk_tar_p.reshape(-1), y_blk_tar_p.reshape(-1)), axis=1)

        # Resample using given mesh-to-mesh resampler
        val_blk_tar = self._mesh_to_mesh_resampler(p_src_p, image_masked_src, p_blk_tar_p)
        return val_blk_tar.reshape((self._blocksize, self._blocksize))

    @staticmethod
    def _rotation_matrix(x, y, z):
        gamma = np.pi - np.arctan2(y, x)
        rotation_matrix_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                                      [np.sin(gamma), np.cos(gamma), 0],
                                      [0, 0, 1]])
        other_v = rotation_matrix_z.dot(np.stack((x, y, z)))

        beta = -np.arctan2(other_v[2], np.abs(other_v[0]))
        rotation_matrix_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                                      [0, 1, 0],
                                      [-np.sin(beta), 0, np.cos(beta)]])

        return rotation_matrix_y.dot(rotation_matrix_z)
