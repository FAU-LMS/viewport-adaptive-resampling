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
import matplotlib.pyplot as plt
import projections
import var
from skimage import img_as_float, io, color, metrics
from misc import cmp_size, get_resampler

# Parameters
image_path = "studio_country_hall.png"
method = 'fsmr'

# Read input
image_erp = color.rgb2gray(img_as_float(io.imread(image_path)))
plt.imshow(image_erp, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title("Original equirectangular")
plt.show()

# Configuration
size_src = image_erp.shape
size_tar = cmp_size(size_src, 32)
erp = projections.EquirectangularProjection(size_src)
cmp = projections.CubemapProjection(size_tar)
blocksize = 8 if method == 'fsmr' else 32
resampler = get_resampler(method)
incident_angle_factor = 2
print(f"Configuration: {size_src=}, {size_tar=}, {erp=}, {cmp=}, {blocksize=}, {resampler=}, {incident_angle_factor=}")

# Perform viewport-adaptive resampling (ERP->CMP)
print(f"Resample to cubemap projection using va-{method}")
image_cmp = var.resample(
    image_src=image_erp,
    projection_src=erp,
    size_tar=size_tar,
    projection_tar=cmp,
    mesh_to_mesh_resampler=resampler,
    blocksize=blocksize,
    incident_angle_factor=incident_angle_factor
)
image_cmp = np.clip(image_cmp, 0, 1)
plt.imshow(image_cmp, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title("Resampled cubemap")
plt.show()

# Perform viewport-adaptive resampling (CMP->ERP)
print(f"Resample back to equirectangular projection using va-{method}")
image_erp_back = var.resample(
    image_src=image_cmp,
    projection_src=cmp,
    size_tar=size_src,
    projection_tar=erp,
    mesh_to_mesh_resampler=resampler,
    blocksize=blocksize,
    incident_angle_factor=incident_angle_factor
)
image_erp_back = np.clip(image_erp_back, 0, 1)
plt.imshow(image_erp_back, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.title("Resampled equirectangular")
plt.show()

print(f"PSNR (va-{method}):", metrics.peak_signal_noise_ratio(image_erp, image_erp_back))
