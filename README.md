# BV Simulation Prep
Scripts to generate a 3D geometry for cardiac MRI

# Installation
1. Create an environment and activate it. If using conda, 
```
conda create -n bvprep
conda activate bvprep
```
2. Install FEniCSx: 
```
conda install -c conda-forge fenics-dolfinx mpich h5py cffi python
```
3. Install necessary packages and modules,
```
python -m pip install -e .
```
4. Install cheart-python-io. Follow the instructions [here](https://gitlab.eecs.umich.edu/jilberto/cheart-python-io) (you will need access to Gitlab).
