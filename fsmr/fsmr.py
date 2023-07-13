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
from numba import jit


def dct_basis_dict(x, y, transform_length):
    """
    Setup DCT basis function dictionary.

    :param x: Mesh positions x [N]
    :param y: Mesh positions y [N]
    :param transform_length: DCT transform length K
    :return: DCT basis function dictionary [K^2, N]
    """
    weights = np.empty((2, 2))
    weights[0, 0] = 1 / transform_length
    weights[0, 1] = np.sqrt(2) / transform_length
    weights[1, 0] = weights[0, 1]
    weights[1, 1] = 2 / transform_length

    k, l = np.mgrid[:transform_length, :transform_length]
    k = k.flatten()
    l = l.flatten()
    cos1 = np.cos((np.pi / transform_length) * (y.reshape(1, -1) - 0.5) * k.reshape(-1, 1))
    cos2 = np.cos((np.pi / transform_length) * (x.reshape(1, -1) - 0.5) * l.reshape(-1, 1))
    basis_functions = cos1 * cos2 * weights[(k == 0).astype(int), (l == 0).astype(int)].reshape(-1, 1)
    return basis_functions


def dct_frequency_weighting(transform_length, sigma):
    """
    Setup DCT frequency weighting.

    :param transform_length: transform length
    :param sigma: frequency weighting factor
    :return: frequency weighting [K^2]
    """
    k, l = np.mgrid[:transform_length, :transform_length]
    k = k.flatten()
    l = l.flatten()
    return np.power(sigma, np.sqrt(np.square(k) + np.square(l)))


@jit(nopython=True)
def fsmr(mesh_val, basis_dict, spatial_weighting, freq_weighting, odc, max_iterations):
    """
    Frequency-selective matching pursuit

    :param mesh_val: signal values at mesh points [N]
    :param basis_dict: basis functions evaluated at mesh points [K^2, N]
    :param spatial_weighting: spatial weights at mesh points [N]
    :param freq_weighting: frequency weighting for basis functions [K^2]
    :param odc: orthogonality deficiency compensation (scalar)
    :param max_iterations: maximum number of iterations (scalar)
    :return: model weights [K^2]
    """
    L, MN = basis_dict.shape
    residual = np.copy(mesh_val)
    coeffs = np.zeros(L)

    D = np.square(basis_dict).dot(spatial_weighting)

    for i in range(max_iterations):
        projected_residual = basis_dict.dot(residual * spatial_weighting)
        obj = freq_weighting * (np.square(projected_residual) / D)
        idx = np.argmax(obj)

        c = projected_residual[idx] / D[idx]

        coeffs[idx] = coeffs[idx] + odc * c
        residual = residual - odc * basis_dict[idx] * c

    return coeffs
