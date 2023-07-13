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
from . import fsmr


def resample_fsmr(source_mesh, source_val, target_mesh, transform_length, odc, sigma, shift, max_iterations,
                  spatial_weighting=None):
    """
    Resample from source_mesh to target_mesh using FSMR.

    :param source_mesh: source sample positions [N, 2]
    :param source_val: source sample values [N]
    :param target_mesh: target sample positions [L, 2]
    :param transform_length: DCT transform length
    :param odc: orthogonality deficiency compensation factor
    :param sigma: frequency weight decay
    :param shift: source/target mesh offset
    :param max_iterations: number of iterations
    :param spatial_weighting: spatial weighting factor
    :return: target sample values [L]
    """
    source_mesh += shift
    target_mesh += shift

    basis_functions_source_mesh = fsmr.dct_basis_dict(source_mesh[:, 0], source_mesh[:, 1], transform_length)
    frequency_weighting = fsmr.dct_frequency_weighting(transform_length, sigma)

    if spatial_weighting is None:
        spatial_weighting = np.ones(source_mesh.shape[0])

    coeffs = fsmr.fsmr(source_val, basis_functions_source_mesh,
                       spatial_weighting, frequency_weighting,
                       odc, max_iterations)

    basis_functions_target_mesh = fsmr.dct_basis_dict(target_mesh[:, 0], target_mesh[:, 1], transform_length)
    target_val = basis_functions_target_mesh.transpose().dot(coeffs)
    return target_val
