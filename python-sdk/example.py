
import st2w_lib
## Initialization
stw2 = st2w_lib.st2w(dataroot='../data/')

## Print the labels from the three vehicles
print(stw2.getLabel('bike'))
print(stw2.getLabel('motorbike'))
print(stw2.getLabel('scooter'))

## Print only the labels from the three vehicles for an explicit step
step = 0 # with 2 Hz step = 10 is equivalent to 5 seconds
print(stw2.getLabel('bike', step))
print(stw2.getLabel('motorbike', step))
print(stw2.getLabel('scooter', step))


## Visualize the dataset


## Visualize the lidar-data with the labeled bounding-box
stw2.Visualizer.view_pointcloud(stw2.pcds['bike'][step], stw2.bboxes['bike'][step])


## Plot the trajecotory of the bike with the velocity colored
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('bike'))
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('bike'), 'bike_velocity')

## Plot the trajecotory of the bike with the roll angle colored
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('bike'))
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('bike'), 'bike_rollangle')

## Plot the trajecotory of the motorbike with the velocity colored
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('motorbike'))
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('motorbike'), 'motorbike_velocity')

## Plot the trajecotory of the motorbike with the roll angle colored
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('motorbike'))
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('motorbike'), 'motorbike_rollangle')

## Plot the trajecotory of the scooter with the velocity colored
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('scooter'))
stw2.Visualizer.plot_trajectory_velocity(stw2.getLabel('scooter'), 'scooter_velocity')

## Plot the trajecotory of the scooter with the roll angle colored
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('scooter'))
stw2.Visualizer.plot_trajectory_rollangle(stw2.getLabel('scooter'), 'scooter_rollangle')

