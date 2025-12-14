
# ROS Modified Pure Pursuit for a Differential Robot
Reference generation, pure pursuit and nonlinear control for path tracking of a differential robot.

In this project, I propose a simple approach that applies nonlinear control for pure pursuit using two strategies for reference generation.

<img src="docs/demo_720p.gif" width="720" />

The details are provided in `docs/documentation.md`, as weel as on Medium:
[Medium Post](https://medium.com/@chaabeni.ilyes2002/pure-pursuit-reference-generation-and-control-for-a-differential-robot-321452e04a6e)

## Overview
The goal is to compute the reference velocities $v_r$ and $\omega_r$ and reference pose $(x_r, y_r, \theta_r)$ using the lookahead point and current position, then apply any type of controller (low level) for speed control. 

For both approaches, the reference point $(x_r, y_r)$ is taken as the lookahead point. A cubic spline is used to find the lookahead point. 

- **Approach 1:**   
Uses a Lyapunov control law for speed control.      
$v_r$ and $\omega_r$ are computed based on the error on orientation $e_\theta$.     
$\theta_r$ is computed as the angle to the lookahead point.     
 
 - **Approach 2:**      
 Can use any type of controller. Implemented a Lyapunov controller from the article [A stable tracking control method for an autonomous mobile robot](https://doi.org/10.1109/ROBOT.1990.126006) by Y. Kanayama, Y. Kimura, F. Miyazaki and T. Noguchi, and sliding mode controller from the article [Nonlinear Path Control for a Differential Drive Mobile Robot](https://api.semanticscholar.org/CorpusID:11832344).       
 $v_r$ and $\theta_r$ are computed using the forward model of the robot and the estimated linear reference velocities $\hat{\dot x}_r$ and $\hat{\dot y}_r$ that are computed from the cubic spline derivatives.     
 $\omega_r$ is computed using $v_r/R$ with $R$ and the curvature radius.     
  
The details and math behind the two approaches will be shared on an article on Medium, as well as a PDF version.

## Installation and Usage
### Dependencies and Installation
ROS Noetic on Ubuntu 20.04, with Gazebo installed and Python3.      
To install the pacakge:     

Clone the repository inside your ROS workspace:
```bash
cd ~/ros_ws/src
git clone https://github.com/Ilyes-Origamist/modified_pure_pursuit_ros.git
```
Compile the project:
```bash
cd .. && catkin_make --pkg modified_pure_pursuit_ros
```
Don't forget to make the files as executable (to run as standalone):
```bash
cd src && sudo chmod +x *.py
```

### Running Python simulation
You can run Python simulation for tests and validation (without ROS) using the scripts `approach1_simulation.py` and `approach2_simulation.py`:
```bash
cd ~/ros_ws/src
python approach1_simulation.py
```

### Running the project in ROS
A launch file is provided to launch Gazebo simulation, spawn the robot (Turtlebot3), launch rviz and `robot_state_publisher`. 

Start a ROS master:
```bash
roscore
```
Open a new terminal and launch Gazebo simulation:
```bash
roslaunch modified_pure_pursuit_ros turtlebot3_gazebo.launch
```
Open a new temrinal and run the modified pure pursuit controller:   
For the first approach:     
```bash
rosrun modified_pure_pursuit_ros approach1_ros.py
```
For the second approach:
```bash
rosrun modified_pure_pursuit_ros approach2_ros.py
```     

### Parameters
- **Reference Path**: The reference path (waypoint) is defined as two numpy arrays in `self.xr` and `self.yr` which you can change      
- **Controller Type**: The controller type can be selected as `lyapunov` or `sliding_mode` for the second appraoch, only in Python simulation `approach2_simulation.py`. In the ROS file, the `sliding_mode` controller has not been implemented yet.
- **Other parameters**: parameters like the lookahead  distance, the goal tolerance and physical limits of the robots are defined within scripts. TODO: use the config file to set them.
- **Config file (TODO)**: The config file `modified_pure_pursuit.yaml` contains parameters `strategy` to select one of the two approaches, `controller_type` to select the controller type and `gains` to set the controller gains. They haven't been used with the programs yet. 