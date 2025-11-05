import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

""" 
    @file approach1_ros.py
    @brief Performs Python simulation for validation of the first approach of the modified pure pursuit.
    @details Uses a Lyapunov control law for speed control.
            xr, yr are taken as the lookahead point.
            vr and wr are computed based on the error on orientation e_θ.
            θr is computed as the angle to the lookahead point. 
    @author Ilyes Chaabeni
    @date 2025-01-05
"""

# Parameters
LOOKAHEAD_DISTANCE = 0.2  # Lookahead distance for Pure Pursuit
GOAL_TOLERANCE = 0.03
DT = 0.1  # Time step for simulation

# Control gains for Lyapunov control law
K1, K2, K3 = 0.05, 0.1, 8.0
# proportional gain for omega_r computation (from angular error)
KP = 0.05

# Robot state [x, y, theta]
robot_state = np.array([0.0, 0.0, 0.0])
# Robot limits
v_max = 0.8
v_min = 0.1
omega_max = 4.5

# Reference trajectory (discrete waypoints)
xr = np.array([0, 0.5, 1.2, 1.8, 1.4, 1.0, 0.6, 0, -0.4])
yr = np.array([0, 0.5, 0.8, 1.6, 2.5, 2.3, 3.5, 3.6, 3.4])

# Interpolate the trajectory using cubic spline
spline_x = CubicSpline(np.arange(len(xr)), xr)
spline_y = CubicSpline(np.arange(len(yr)), yr)

# Lyapunov control law
def lyapunov_control(vr, omega_r, e1, e2, e3):
    v = vr * np.cos(e3) + K1 * e1
    omega = omega_r + vr * (K2 * e2 + K3 * np.sin(e3))
    return v, omega

# find the closest point on the trajectory
def closest_point_on_trajectory(robot_x, robot_y, spline_x, spline_y, t_range):
    t_values = np.linspace(t_range[0], t_range[1], 100)
    trajectory_points = np.vstack([spline_x(t_values), spline_y(t_values)]).T
    distances = np.linalg.norm(trajectory_points - np.array([robot_x, robot_y]), axis=1)
    closest_idx = np.argmin(distances)
    return spline_x(t_values[closest_idx]), spline_y(t_values[closest_idx]), t_values[closest_idx]

# find the lookahead point
def find_lookahead(robot_state, spline_x, spline_y, t_range):
    # Find the closest point on the trajectory
    x_closest, y_closest, t_closest = closest_point_on_trajectory(robot_state[0], robot_state[1], spline_x, spline_y, t_range)
    
    # Initialize t_lookahead
    t_lookahead = t_closest
    while True:
        # Compute the lookahead point
        x_lookahead = spline_x(t_lookahead)
        y_lookahead = spline_y(t_lookahead)
        
        # Compute the Euclidean distance
        distance = np.sqrt((x_lookahead - x_closest)**2 + (y_lookahead - y_closest)**2)
        
        # Check if the distance is approximately equal to the lookahead distance
        if np.isclose(distance, LOOKAHEAD_DISTANCE, atol=0.005):
            break
        
        # Increment t_lookahead to move further along the trajectory
        t_lookahead += 0.005  # Adjust the step size as needed
    
    return x_lookahead, y_lookahead, t_lookahead

## Simulation loop
t_range = [0, len(xr) - 1]  # Range of the trajectory parameter
trajectory_points = []  # To store the robot's path
v_control_sig = [] 
w_control_sig = []
N_steps = 500

for _ in range(N_steps):  # Simulate for 500 steps and stop if goal is reached
    # Get the lookahead point using Pure Pursuit
    x_lookahead, y_lookahead, t_lookahead = find_lookahead(robot_state, spline_x, spline_y, t_range)

    distance_to_goal = np.linalg.norm([robot_state[0] - xr[-1], robot_state[1] - yr[-1]])
    if distance_to_goal <= LOOKAHEAD_DISTANCE:
        # if the robot has reached the goal, send v=omega=0
        if distance_to_goal <= GOAL_TOLERANCE:
            x_lookahead, y_lookahead = 0, 0
        else:
            # the distance to the goal is smaller than the lookahead distance
            # but greater than tolerance
            # then take the goal as the lookahead point
            x_lookahead, y_lookahead = xr[-1], yr[-1]
    
    # Compute errors in the robot frame
    ex = x_lookahead - robot_state[0]
    ey = y_lookahead - robot_state[1]
    theta_ref = np.arctan2(ey, ex)
    e_theta = theta_ref - robot_state[2]

    # Transform errors to the robot frame
    e1 = np.cos(robot_state[2]) * ex + np.sin(robot_state[2]) * ey
    e2 = -np.sin(robot_state[2]) * ex + np.cos(robot_state[2]) * ey
    e3 = e_theta
    
    # Generate reference velocities
    vr = 1.0 # you can set this to a constant value or use a function of the errors
    weight = np.abs(np.sin(e_theta))
    vr = v_max * (1 - weight) + v_min * weight  # Reference linear velocity
    
    omega_r = - KP * e_theta
    omega_r = omega_max * np.sin(e_theta)
    
    # Compute control inputs using Lyapunov control law
    v, omega = lyapunov_control(vr, omega_r, e1, e2, e3)
    # Limit control inputs
    v = np.clip(v, 0.0, v_max)
    omega = np.clip(omega, -omega_max, omega_max)
    
    # tight_turn = e_theta > np.pi / 12.0
    # vr = v_max * (not tight_turn) + v_min * tight_turn
    # omega_r = -0.1 * e_theta

    v_control_sig.append(v)
    w_control_sig.append(omega)
    
    # Update robot state using the kinematic model
    robot_state[0] += v * np.cos(robot_state[2]) * DT
    robot_state[1] += v * np.sin(robot_state[2]) * DT
    robot_state[2] += omega * DT
    
    # Store the robot's position for visualization
    trajectory_points.append([robot_state[0], robot_state[1]])
    
    distance_to_goal = np.linalg.norm([robot_state[0] - xr[-1], robot_state[1] - yr[-1]])
    # Break if the robot reaches the end of the trajectory
    if distance_to_goal <= GOAL_TOLERANCE:
        break

# Plot the results
trajectory_points = np.array(trajectory_points)
plt.figure(figsize=(10, 6))
plt.plot(spline_x(np.linspace(t_range[0], t_range[1], 100)), spline_y(np.linspace(t_range[0], t_range[1], 100)), 'r--', label="Reference Trajectory")
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'b-', label="Robot Path")
plt.scatter(xr, yr, c='green', label="Waypoints")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Modified Pure Pursuit with Lyapunov Control (Proposition 1)")
plt.legend()
plt.grid()
plt.axis("equal")

plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(v_control_sig)) * DT, v_control_sig, label="v")
plt.plot(np.arange(len(w_control_sig)) * DT, w_control_sig, label="w")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Control")
plt.title("Control Signals v and w")
plt.grid()

plt.show()
