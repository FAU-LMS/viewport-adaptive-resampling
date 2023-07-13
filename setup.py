from setuptools import setup, find_packages

setup(name='var',
      version='1.0.0',
      description='Viewport-adaptive resampling.',
      long_description='Viewport-adaptive resampling for 360-degree projection format conversion.',
      author='Andy Regensky',
      author_email='andy.regensky@fau.de',
      packages=find_packages(exclude=['doc']),
      install_requires=[
          'numpy',
          'numba',
          'tqdm',
      ],
      python_requires='>=3.6,<3.11',
      include_package_data=False)
