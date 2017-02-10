# Blind Regression

In cuda_implementation.py, there is an example taking a data matrix as input and returning the estimated result. (with the CUDA part left to be implemented)

## Setup Python Virtual Environment
It's recommended to setup a virtual environment by virtualenv.
Create a virtual environment
```
cd Path_to/gpu_projects/cuda/blind_regression/python
```
```
virtualenv -p python3 reg_venv
```
Activate the virtual environment so that whatever python packages you are going to install will only live in this environment.
```
source reg_venv/bin/activate
```
Install the necessary packages
```
pip3 install numpy
```

## Download the Dataset
```
hadoop fs -get /user/ykao/dataset/blind_regression/Microcontrollers_data_matrix.p Path_to/gpu_projects/cuda/blind_regression/python
```

## Run the Example
```
cd Path_to/gpu_projects/cuda/blind_regression/python
```
```
source reg_venv/bin/activate
```
```
python3 cuda_implementation.py
```
