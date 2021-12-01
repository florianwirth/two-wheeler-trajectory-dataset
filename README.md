# Two-Wheeler-Trajectory Dataset
This repo contains trajectory data of 3 different single-track two-wheelers in highly dynamic scenarios, between 1-1.5 min each, labelled with 2 Hz. The labels are stored in `/data/labels`. This development kit loads the dataset and visualizes it for inspection. Use this dataset for your individual research projects all around prediction and safety of two-wheelers.

## Setup
The package is developed and tested in Python 3.7. To install, clone this repository and install the required packages. 
Run the example file in python-sdk.
This script creates plots of the trajectory of a motorcyclist, a cyclist and an electric scooter.
Color encode different dynamic features: velocity and roll angle. However, there are more dynamic features like yaw angle and some derivatives stored in the `dataLoader`. 

In short:
```bash
git clone https://github.com/florianwirth/two-wheeler-trajectory-dataset
cd two-wheeler-trajectory-dataset/python-sdk
python3 example.py
```

## Contribution / Contact
Feel free to contact me, if you like to use our dataset:

Florian Wirth `florian.wirth@kit.edu`

#### Student Assistant:

Julian Wadephul

## Citation

This work is handed in as a contribution for the IEEE International Conference of Robotics and Automation (ICRA). 
It is currently under review. 
If you use our dataset for your research, we are happy when you cite our paper in case it gets accepted. 
