import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

""" 
    @file approach2_simulation.py
    @brief Performs Python simulation for validation of the second approach of the modified pure pursuit.
    @details Can use a Lyapunov controller or sliding mode controller.
            You can also add any other controller (for instance, PID) as well.
            vr and θr are computed using the estimated linear reference velocities that are
            computed from spline derivatives. wr is computed using vr/R with R and the curvature radius. 
    @author Ilyes Chaabeni
    @date 2025-01-05
    @note You can define your own reference generation and control strategies.
"""

# Parameters
LOOKAHEAD_DISTANCE = 0.15  # Lookahead distance for Pure Pursuit
GOAL_TOLERANCE = 0.05
DT = 0.1  # Time step for simulation

# Lyapunov control gains
K1, K2, K3 = 0.05, 0.1, 3.5
# SMC control gains
LAMBDA = 0.5 # Sliding surface gain
K = 4.0 # Switching control gain
SMC_A = 0.05 # SMC boundary layer width

# Controller selection: 'lyapunov' or 'sliding_mode'
controller_type = 'sliding_mode'  

# Robot state [x, y, theta]
robot_state = np.array([0.0, 0.0, 0.0])
# Robot limits
v_max = 0.8
omega_max = 4.5
# Reference trajectory (discrete waypoints)
xr = np.array([0, 0.5, 1.2, 1.8, 1.4, 1.0, 0.6, 0, -0.4])
yr = np.array([0, 0.5, 0.8, 1.6, 2.5, 2.3, 3.5, 3.6, 3.4])

# Interpolate the trajectory using cubic spline
spline_x = CubicSpline(np.arange(len(xr)), xr)
spline_y = CubicSpline(np.arange(len(yr)), yr)
# Compute the derivatives of the splines
spline_x_derivative = spline_x.derivative()
spline_y_derivative = spline_y.derivative()

# Function to compute the closest point on the trajectory
def closest_point_on_trajectory(robot_x, robot_y, spline_x, spline_y, t_range):
    t_values = np.linspace(t_range[0], t_range[1], 100)
    trajectory_points = np.vstack([spline_x(t_values), spline_y(t_values)]).T
    distances = np.linalg.norm(trajectory_points - np.array([robot_x, robot_y]), axis=1)
    closest_idx = np.argmin(distances)
    return spline_x(t_values[closest_idx]), spline_y(t_values[closest_idx]), t_values[closest_idx]

# Pure Pursuit algorithm
def pure_pursuit(robot_state, spline_x, spline_y, t_range):
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
        t_lookahead += 0.005
        # Check if t_lookahead exceeds the trajectory range
        if t_lookahead > t_range[1]:
            t_lookahead = t_range[1]
            break
        
    return x_lookahead, y_lookahead, t_lookahead, x_closest, y_closest, t_closest

# Reference generation
def reference_generation_lyapunov(spline_x_derivative, spline_y_derivative, curvature):
    dx_dt = spline_x_derivative
    dy_dt = spline_y_derivative
    # Compute the reference linear and angular velocity
    vr = np.sqrt(dx_dt**2 + dy_dt**2)
    omega_r = curvature * vr
    return vr, omega_r

def lyapunov_control(vr, omega_r, errors):
    e1 = errors[0]; e2 = errors[1]; e3 = errors[2]
    # compute the control law
    v = vr * np.cos(e3) + K1 * e1
    omega = omega_r + vr * (K2 * e2 + K3 * np.sin(e3))
    return v, omega

# Reference generation
def reference_generation_smc(spline_x_derivative, spline_y_derivative, curvature, errors, v_max):
    ey_R = errors[0]
    e3 = errors[1]
    dx_dt = spline_x_derivative
    dy_dt = spline_y_derivative
    # Compute the reference linear and angular velocity
    vr = np.sqrt(dx_dt**2 + dy_dt**2)
    # Compute the reference linear velocity control input    
    v = vr * np.cos(e_theta)
    # limit the linear velocity
    v = np.clip(v, 0.0, v_max)
    # Compute the reference angular velocity
    omega_r = (curvature * v * np.cos(e3)) / (1 + curvature * ey_R) 
    return v, omega_r

def smc_control(v, omega_r, e2, e3):
    z1 = e2
    z2 = np.sin(e3)
    S = z2 + LAMBDA * z1
    u_sw = -K * np.clip(S, -SMC_A, SMC_A)
    u = -LAMBDA * v * z2 + u_sw
    omega = omega_r - u / np.cos(e3)
    return omega

# Simulation loop
t_range = [0, len(xr) - 1]
trajectory_points = []
v_control_sig = []
w_control_sig = []
goal_reached = False
N_steps = 500

for _ in range(N_steps): # Simulate for N_steps and stop if goal is reached
    # Find the lookahead point and the closest point on the trajectory
    x_lookahead, y_lookahead, t_lookahead, x_closest, y_closest, t_closest = pure_pursuit(robot_state, spline_x, spline_y, t_range)
    # Current distance from the goal point
    distance_to_goal = np.linalg.norm([robot_state[0] - xr[-1], robot_state[1] - yr[-1]])
    if distance_to_goal <= LOOKAHEAD_DISTANCE:
        # if the robot has reached the goal, send v=omega=0
        if distance_to_goal <= GOAL_TOLERANCE:
            goal_reached = True
        else:
            # the distance to the goal is smaller than the lookahead distance
            # but greater than tolerance
            # then take the goal as the lookahead point
            x_lookahead, y_lookahead = xr[-1], yr[-1]
            t_lookahead = t_range[1]

    # t_index at which to evaluate the derivatives
    elif (controller_type == 'sliding_mode'):
        # due to different frame and control strategy,
        # we need to use the closest point instead of the lookahead
        t_index = t_closest
    else:
        t_index = t_lookahead

    # Evaluate the derivatives at t_index
    dx_dt = spline_x_derivative(t_index)
    dy_dt = spline_y_derivative(t_index)

    # reference angle
    theta_ref = np.arctan2(dy_dt, dx_dt)
    # compute the absolute errors to lookahead point
    ex = x_lookahead - robot_state[0]
    ey = y_lookahead - robot_state[1]
    e_theta = theta_ref - robot_state[2] 

    # compute curvature (pure pursuit)
    alpha = np.arctan2(ey, ex) - robot_state[2] 
    curvature = 2*np.sin(alpha) / LOOKAHEAD_DISTANCE
    
    ## Generate reference velocities and compute control inputs
    if controller_type == 'lyapunov':
        # Reference generation
        vr, omega_r = reference_generation_lyapunov(dx_dt, dy_dt, curvature)
        # Compute errors in the robot frame
        e1 = np.cos(robot_state[2]) * ex + np.sin(robot_state[2]) * ey
        e2 = -np.sin(robot_state[2]) * ex + np.cos(robot_state[2]) * ey
        e3 = e_theta
        # Compute control inputs
        v, omega = lyapunov_control(vr, omega_r, [e1, e2, e3])
        # Limit control inputs
        v = np.clip(v, 0.0, v_max)
        omega = np.clip(omega_r, -omega_max, omega_max)
        
    elif controller_type == 'sliding_mode':
        # Compute errors in the robot frame
        ey_R = x_closest - robot_state[0]
        e1 = np.cos(robot_state[2]) * 0 + np.sin(robot_state[2]) * ey_R
        e2 = - np.sin(robot_state[2]) * 0 + np.cos(robot_state[2]) * ey_R
        e3 = - e_theta
        # Compute the reference linear velocity
        vr = np.sqrt(dx_dt**2 + dy_dt**2)
        # Compute the linear velocity control input    
        v = vr * np.cos(e_theta)
        # limit the linear velocity
        v = np.clip(v, 0.0, v_max)
        # Compute the reference angular velocity
        omega_r = (curvature * v * np.cos(e3)) / (1 + curvature * ey_R) 
        # Compute omega control input
        omega = smc_control(v, omega_r, e2, e3)
        # Limit control inputs
        omega = np.clip(omega_r, -omega_max, omega_max)
        
    else:
        raise ValueError("Unknown controller_type")
    
    # Store control signals
    v_control_sig.append(v)
    w_control_sig.append(omega)
    # Update robot state
    robot_state[0] += v * np.cos(robot_state[2]) * DT
    robot_state[1] += v * np.sin(robot_state[2]) * DT
    robot_state[2] += omega * DT
    # Store the robot's position for visualization
    trajectory_points.append([robot_state[0], robot_state[1]])
    # Break if the robot reaches the end of the trajectory
    if goal_reached:
        break

## Plotting
# Plot the trajectories and waypoints
trajectory_points = np.array(trajectory_points)
plt.figure(figsize=(10, 6))
plt.plot(spline_x(np.linspace(t_range[0], t_range[1], 100)), spline_y(np.linspace(t_range[0], t_range[1], 100)), 'r--', label="Reference Trajectory")
plt.plot(trajectory_points[:, 0], trajectory_points[:, 1], 'b-', label="Robot Path")
plt.scatter(xr, yr, c='green', label="Waypoints")
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Modified Pure Pursuit with {'Lyapunov' if controller_type=='lyapunov' else 'Sliding Mode'} Control")
plt.legend()
plt.grid()
plt.axis("equal")

# Plot control signals
plt.figure(figsize=(10, 6))
plt.plot(np.arange(len(v_control_sig)) * DT, v_control_sig, label="v")
plt.plot(np.arange(len(w_control_sig)) * DT, w_control_sig, label="w")
plt.legend()
plt.xlabel("Time (s)")
plt.ylabel("Control")
plt.title("Control Signals v and w")
plt.xlim(0, 10.0)
plt.grid()

# Show figures
plt.show()
