# pyqt

A quantum transport library based on the non equilibrium Green's function (NEGF) formalism written in Python.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

    > Ase https://wiki.fysik.dtu.dk/ase/
    > numpy https://github.com/numpy


### Installing

pip install qtpyt

## Running the tests

pytest subfolder/tests

### Upload

# change version number.

twine upload --repository testpypi dist/qtpyt-0.0.tar.gz

### Register

python setup.py sdist

### Build Cython

python setup.py build_ext --inplace

### Install develop

pip install -e .
