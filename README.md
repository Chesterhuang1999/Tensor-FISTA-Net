# Tensor-FISTA-Net

## Requirements
`
Tensorflow 2.0 
h5py
`
## Installation
Download the file through Git or directly download the zip file:

`
git clone https://github.com/Chesterhuang1999/Tensor-FISTA-Net.git
`
## Download the data 

The training, testing and mask data can be download through [Baidu Drive](https://pan.baidu.com/s/1yL8rKvpSQ_Gx8JAR41-MTw).(Key:gsni)

## Usage 

### Training CACTI datasets 

1. Open DefineParam.py and change the variable 'name' into 'Kobe' / 'Park' / 'Vehicle'.
2. Open the directory 'maskData' and rename the mask for CACTI  from 'mask_cacti.mat' to 'mask256.mat' 
3. Train the model through 
`
python Train.py
`
### Training CASSI datasets

1. Open DefineParam.py and change the variable 'name' into 'Spectral' 
2. Open the directory 'maskData' and rename the mask for CASSI from 'mask_cassi.mat' to 'mask256.mat' 
3. Train the model through 
`
python Train.py
`
### Testing CACTI datasets
1. Open DefineParam.py and change the variable 'name' into 'Kobe' / 'Park' / 'Vehicle'.
2. Open the directory 'maskData' and rename the mask for CACTI  from 'mask_cacti.mat' to 'mask256.mat' 
3. Test the model through 
`
python Reconstruction.py
`
### Testing CASSI datasets

1. Open DefineParam.py and change the variable 'name' into 'Spectral' 
2. Open the directory 'maskData' and rename the mask for CASSI from 'mask_cassi.mat' to 'mask256.mat' 
3. Test the model through 
`
python Test.py
`
