# Data preparation

## Prerequests 
Several additional packages are required to a dataset from depth frames: 

* Connected Components package: (``pip install connected-components-3d``)
* Install Cython: (``conda install Cython``)
* Distance transform package: (```pip install git+https://github.com/markomih/distance-transform```)
* Voxel fusion: (follow the [instructions](https://github.com/markomih/voxel_fusion))
```bash
mkdir external
cd exteranl && git clone https://github.com/markomih/voxel_fusion && cd voxel_fusion
mkdir build && cd build
cmake ..
make 
export LD_LIBRARY_PATH="$PWD:$LD_LIBRARY_PATH"

cd ..
pip install .
```

## Prepare data
Run the following script to prepare a low res DeepSurfels scene:
```
python from_depth_frames.py
```
See the script for more details.