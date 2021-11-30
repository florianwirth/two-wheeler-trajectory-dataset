# two-wheeler-trajectory-dataset
Contains trajectory data of 3 different single-track two-wheelers in dynamic scenarios, 1-1.5 min each. The labels are saved to /data/labels. This development kit loads the dataset to visualize, analyze and use them for individual projects.

## Development kit setup
The package is developed and tested in pyhton 3.7. To install clone this git repository and install the required packages.  

Run the example file to get in touch with this development kit. This script creates plots of the three vehicles, shows and saves them. The dataLoader class loads and give you access to the dataset.

If you want to visualiue the lidar point cloud data you have to download them to /data/pcls/ and set the boolean _usePointClouds_ to True.