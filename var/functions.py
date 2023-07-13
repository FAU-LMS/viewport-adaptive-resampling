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
from . import ViewportAdaptiveResampler
from projections import Projection
import numpy as np


def resample(image_src: np.ndarray,
             projection_src: Projection,
             size_tar: Tuple[int, int],
             projection_tar: Projection,
             mesh_to_mesh_resampler: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
             blocksize: int = 8,
             incident_angle_factor: float = 2,
             progress=True):
    """
    Resample image in source projection format to target projection format using viewport-adaptive resampling.

    :param image_src: image in source projection format
    :param projection_src: source projection
    :param size_tar: size of image in target projection format
    :param projection_tar: target projection
    :param mesh_to_mesh_resampler: mesh-to-mesh capable resampler as
           (sample_pos_src [N, 2], sample_val_src [N], sample_pos_tar [L, 2]) -> sample_val_tar [L]
    :param blocksize: block size used for resampling (default: 8)
    :param incident_angle_factor: incident angle factor for source sample range selection (default: 2)
    :param progress: whether to show current progress during resampling (default: true)
    :return: image in target projection format
    """
    var_resampler = ViewportAdaptiveResampler(image_src.shape, projection_src,
                                              size_tar, projection_tar,
                                              mesh_to_mesh_resampler,
                                              blocksize, incident_angle_factor)
    image_tar = var_resampler.resample(image_src, progress=progress)
    return image_tar
