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


from functools import partial
from scipy import interpolate
import numpy as np
import fsmr


def cmp_size(erp_size, block_size=32):
    """
    Calculate cubemap size with number of samples close to erp.

    :param erp_size: size of image in equirectangular projection
    :param block_size: block size each face shall be dividable by
    :return: cubemap size
    """
    v = np.floor(np.sqrt(erp_size[0] * erp_size[1] / 6))
    block_residual = v % block_size
    if block_residual < block_size/2:
        cubeface_res = v + (block_size - block_residual)
    else:
        cubeface_res = v - block_residual
    return int(cubeface_res * 2), int(cubeface_res * 3)


def get_resampler(method):
    """
    Convenience function to get a preconfigured resampler for the desired method.

    :param method: resampling method ('nearest', 'linear', 'cubic', 'fsmr')
    :return: preconfigured resampler
    """
    if method in ['nearest', 'linear', 'cubic']:
        return partial(interpolate.griddata, method=method, fill_value=0)
    elif method == 'fsmr':
        return partial(fsmr.resample_fsmr, transform_length=32, odc=0.5, sigma=0.93, shift=16, max_iterations=1000)
    else:
        raise ValueError(f"Unknown resampling method '{method}'.")
