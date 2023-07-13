# Viewport-Adaptive Spherical Image Resampling

Python packages for Viewport-Adaptive Resampling and Keypoint-Agnostic Frequency-Selective Mesh-to-Mesh Resampling.

* A. Regensky, V. Heimann, R. Zhang, and A. Kaup, "Improving Spherical Image Resampling through Viewport-Adaptivity," accepted for *IEEE International Conference on Image Processing*, Oct. 2023, preprint available: [arxiv:2306.13692](https://arxiv.org/abs/2306.13692).
* V. Heimann, N. Genser, and A. Kaup, "Key Point Agnostic Frequency-Selective Mesh-to-Grid Image Resampling using Spectral Weighting," in *Proceedings of the IEEE 22nd International Workshop on Multimedia Signal Processing*, Sep. 2020, doi: [10.1109/MMSP48831.2020.9287096](https://doi.org/10.1109/MMSP48831.2020.9287096).

## Setup

Setup is possible in two ways.
Depending on your use case, you may either want to **clone** the project and run the provided example script or you may want to **install** the available packages in your python environment (side-packages) in order to use the provided APIs in your own projects.

### Clone

To run the provided example script, clone the project and install the required dependencies.
It is recommended to install the dependencies in a dedicated python environment.

```shell
git clone https://github.com/fau-lms/viewport-adaptive-resampling.git
cd viewport-adaptive-resampling
... # Activate your python environment
pip install numpy numba tqdm scikit-image scipy matplotlib
```

Execute `example_var.py` to run the example script.

```shell 
python example_var.py
```

### Install

To use the provided APIs in your own projects, install the available packages in your python environment using
```shell
pip install git+https://github.com/fau-lms/viewport-adaptive-resampling.git
```

or if you have already cloned the project in a previous step
```shell
cd viewport-adaptive-resampling
pip install .
```

For usage of the provided APIs, please see [Usage](#usage).

## Usage

The following script shows an examplary use case for resampling an image in equirectangular projection format to cubemap projection format.

```python
import var
import fsmr
import projections
from functools import partial

# Setup 
image_erp = # Read image in erp format
erp = projections.EquirectangularProjection(image_erp.shape)
size_tar = (640, 960)
cmp = projections.CubemapProjection(size_tar)
blocksize = 8

# Configure the mesh-to-mesh resampler
resampler = partial(fsmr.resample_fsmr,
                    transform_length=32,
                    odc=0.5,
                    sigma=0.93,
                    shift=16,
                    max_iterations=1000)

# Resample using viewport-adaptive resampling
image_cmp = var.resample(
    image_src=image_erp,
    projection_src=erp,
    size_tar=size_tar,
    projection_tar=cmp,
    mesh_to_mesh_resampler=resampler,
    blocksize=blocksize
)
```

The `mesh_to_mesh_resampler` given to `var.resample` can be any mesh-to-mesh capable resampling technique with io format `(sample_pos_src, sample_val_src, sample_pos_tar) -> sample_val_tar`:
* `sample_pos_src`: source sample positions [N, 2]
* `sample_val_src`: source sample values [N]
* `sample_pos_tar`: target sample positions [L, 2]
* `sample_val_tar`: target sample values [L]

## License

BSD 3-Clause License. For details, see [LICENSE](LICENSE).

## Citation

If you use this software in your work, please cite

```bibtex
% Viewport-Adaptive Resampling (VAR)
@inproceedings{Regensky23_VAR,
    title={Improving Spherical Image Resampling through Viewport-Adaptivity}, 
    author={Andy Regensky and Viktoria Heimann and Ruoyu Zhang and Andr\'{e} Kaup},
    booktitle={Proceedings of the IEEE International Conference on Image Processing},
    year={2023},
    month = oct,
    pages={tba},
    doi={tba}
}

% Keypoint Agnostic Frequency-Selective Mesh-to-Grid Resampling (FSMR)
@inproceedings{Heimann20_FSMR,
    title={Key Point Agnostic Frequency-Selective Mesh-to-Grid Image Resampling using Spectral Weighting}, 
    author={Viktoria Heimann and Nils Genser and Andr\'{e} Kaup},
    booktitle={Proceedings of the IEEE 22nd International Workshop on Multimedia Signal Processing},
    year={2020},
    month = sep,
    pages={1--6},
    doi={10.1109/MMSP48831.2020.9287096}
}
```
