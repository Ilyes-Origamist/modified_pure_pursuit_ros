#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import tf
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import subprocess, time, csv, os

""" 
    @file approach1_ros.py
    @brief ROS Implementation of the first approach of the modified pure pursuit.
    @details Uses a Lyapunov control law for speed control.
            xr, yr are taken as the lookahead point.
            vr and wr are computed based on the error on orientation e_θ.
            θr is computed as the angle to the lookahead point. 
    @author Ilyes Chaabeni
    @date 2025-01-05
    @todo Use the parameter 'strategy' to select the pursuit strategy, 'gains' to set the controller gains. Add other parameters like lookahead distance, goal tolerance, robot parameters.
"""

class ModifiedPurePursuit:
    def __init__(self):
        rospy.init_node('modified_pure_pursuit', anonymous = False)
        self.odom_sub = rospy.Subscriber('odom', Odometry, self.odom_callback )
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10 )
        self.path_pub = rospy.Publisher('path', Path, queue_size=10)
        self.lookahead_pub = rospy.Publisher('lookahead_point', PoseStamped, queue_size=10)
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
        
        self.rate = 4  # Hz
        # Path Parameters
        self.LOOKAHEAD_DISTANCE = 0.2
        self.goal_tolerance = 0.04
        # Robot Paramterers
        self.vr = 0.3 # Reference linear velocity
        max_tr_s = 0.5 # max rotational speed of the robot in tr/s
        self.omega_max = 2 * np.pi * max_tr_s 
        self.v_max = 0.5 # max linear velocity
        self.v_min = 0.1 # min linear velocity

        # Lyapunov Controller Gains
        self.K1, self.K2, self.K3 = 0.05, 0.1, 8.0
        # Proportional Controller gain for omega_ref
        self.Kp = 0.1
        
        # Trajectory to follow
        # Reference trajectory (discrete waypoints)
        self.xr = np.array([0, 0.5, 1.2, 1.8, 1.4, 1.0, 0.6, 0, -0.4])
        self.yr = np.array([0, 0.5, 0.8, 1.6, 2.5, 2.3, 3.5, 3.6, 3.4])

        # Interpolate the trajectory using cubic spline
        self.spline_x = CubicSpline(np.arange(len(self.xr)), self.xr)
        self.spline_y = CubicSpline(np.arange(len(self.yr)), self.yr)
        self.t_range = [0, len(self.xr) - 1]  # Range of the trajectory parameter

        self.timestamps = []

        # Control Signals for plotting
        self.v_control_sig = []
        self.w_control_sig = []

        # Error Signal for plotting
        self.error1 = []
        self.error2 = []
        self.error3 = []
        self.error = []
        
        self.prev_e_theta = 0.0

    def modified_pursuit_controller(self):
        ## Compute Reference
        x_r, y_r, t_lookahead = self.find_lookahead()
        
        # Check if the robot is close to the goal
        distance_to_goal = np.linalg.norm([self.x - self.xr[-1], self.y - self.yr[-1]])
        if distance_to_goal <= self.LOOKAHEAD_DISTANCE:
            # if the robot has reached the goal, send v=omega=0
            if distance_to_goal <= self.goal_tolerance:
                rospy.loginfo(f"Goal reached: v_r: {0}, omega_r: {0}")
                return 0, 0
            else:
                # the distance to the goal is smaller than the lookahead distance
                # but greater than tolerance
                # then take the goal as the lookahead point
                x_r, y_r = self.xr[-1], self.yr[-1]
        
        ## Compute errors 
        # Errors in the robot frame
        ex = x_r - self.x
        ey = y_r - self.y
        # Compute theta_ref
        theta_ref = np.arctan2(ey, ex)
        e_theta = theta_ref - self.theta

        # Transform errors to the robot frame
        e1 = np.cos(self.theta) * ex + np.sin(self.theta) * ey
        e2 = -np.sin(self.theta) * ex + np.cos(self.theta) * ey
        e3 = e_theta
        
        # store errors for plotting
        self.error1.append(e1)
        self.error2.append(e2)
        self.error3.append(e3)
        
        ## Compute vr and omega_r
        # Compute v_ref
        # ponderation = np.sin(e_theta)
        # vr = self.v_max * (1 - np.abs(ponderation)) + self.v_min * np.abs(ponderation) # Reference linear velocity
        tight_turn = np.abs(e_theta) > np.pi / 12.0
        vr = self.vr * (not tight_turn) + self.v_min * tight_turn
        
        # Compute omega_ref  
        # omega_r = 0
        # omega_r = (theta_ref - robot_state[2]) / DT  # Reference angular velocity (assumed zero for simplicity)
        # omega_r = self.omega_max * ponderation # accounts for sign
        omega_r = - self.Kp * e_theta

        rospy.loginfo(f"v_r: {vr}, omega_r: {omega_r}")
        
        ## Lyapunov Controller
        v = vr * np.cos(e3) + self.K1 * e1
        omega = omega_r + vr * (self.K2 * e2 + self.K3 * np.sin(e3))
        
        # Limits
        if v > self.v_max:
            v = self.v_max
        if v < 0.0:
            v = 0.0
        if np.abs(omega) > self.omega_max:
            omega = np.sign(omega) * self.omega_max

        # Store control signals for plotting
        self.v_control_sig.append(v)
        self.w_control_sig.append(omega)
        # rospy.loginfo("v= %.2f, omega= %.2f", v, omega)
        # self.timestamps.append(rospy.Time.now().to_sec())
        # rospy.loginfo("timestamp: %.2f", self.timestamps[-1])
        # # Return control inputs
        return v, omega
    

    def find_lookahead(self):
        # Find closest point on trajectory
        t_values = np.linspace(self.t_range[0], self.t_range[1], 100)
        trajectory_points = np.vstack([self.spline_x(t_values), self.spline_y(t_values)]).T
        distances = np.linalg.norm(trajectory_points - np.array([self.x, self.y]), axis=1)
        closest_idx = np.argmin(distances)
        x_closest, y_closest, t_closest = self.spline_x(t_values[closest_idx]), self.spline_y(t_values[closest_idx]), t_values[closest_idx]

        # update the sideway error
        self.error.append(y_closest - self.y)
        # rospy.loginfo('error: %.2f', self.error[-1])
        
        # Find lookahead point
        # Initialize t_lookahead
        t_lookahead = t_closest
        while True:
            # Compute the lookahead point
            x_lookahead = self.spline_x(t_lookahead)
            y_lookahead = self.spline_y(t_lookahead)
            
            # Compute the Euclidean distance
            distance = np.sqrt((x_lookahead - x_closest)**2 + (y_lookahead - y_closest)**2)
            
            # Check if the distance is approximately equal to the lookahead distance
            if np.isclose(distance, self.LOOKAHEAD_DISTANCE, atol=0.005):
                break
            
            # Increment t_lookahead to move further along the trajectory
            t_lookahead += 0.005  # Adjust the step size as needed
        
        return x_lookahead, y_lookahead, t_lookahead

        
    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y 

        # Get the orientation (theta) from the quaternion
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, theta) = euler_from_quaternion(orientation_list)
        self.theta = theta

    # Publish path to visualize in rviz
    def publish_path(self):
        path_msg = Path()
        path_msg.header.frame_id = "odom"
        path_msg.header.stamp = rospy.Time.now()

        t_values = np.linspace(self.t_range[0], self.t_range[1], 100)
        for t in t_values:
            pose = PoseStamped()
            pose.pose.position.x = self.spline_x(t)
            pose.pose.position.y = self.spline_y(t)
            pose.pose.position.z = 0
            pose.pose.orientation.w = 1.0  # No orientation
            path_msg.poses.append(pose)

        self.path_pub.publish(path_msg)

    # Publish lookahead point as Pose for visualization
    def publish_lookahead_point(self, x_lookahead, y_lookahead, theta_ref):
        lookahead_msg = PoseStamped()
        lookahead_msg.header.frame_id = "odom"
        lookahead_msg.header.stamp = rospy.Time.now()
        lookahead_msg.pose.position.x = x_lookahead
        lookahead_msg.pose.position.y = y_lookahead
        lookahead_msg.pose.position.z = 0

        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta_ref)
        lookahead_msg.pose.orientation.x = quaternion[0]
        lookahead_msg.pose.orientation.y = quaternion[1]
        lookahead_msg.pose.orientation.z = quaternion[2]
        lookahead_msg.pose.orientation.w = quaternion[3]

        self.lookahead_pub.publish(lookahead_msg)

    # function that runs the whole algorithm
    def run(self):
        rate = rospy.Rate(self.rate)  # rate in Hz
        while not rospy.is_shutdown():
            v, omega = self.modified_pursuit_controller()
            cmd = Twist()
            cmd.linear.x = v
            cmd.angular.z = omega
            self.cmd_pub.publish(cmd)
            self.publish_path()  # Publish the path

            # Publish the lookahead point
            x_r, y_r, t_lookahead = self.find_lookahead()
            theta_ref = np.arctan2(y_r - self.y, x_r - self.x)
            self.publish_lookahead_point(x_r, y_r, theta_ref)

            rate.sleep()

    # Plot results after killing the node   
    def save_plot_signals(self):
        rospy.loginfo("Saving control signals...")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Move up one level to the root directory (~/ros_ws/modified_pure_pursuit/src)
        root_dir = os.path.abspath(os.path.join(current_dir, '..')) 
        # Path to the other folder for exporting .csv files
        target_dir = os.path.join(root_dir, 'output')
        # Ensure the target directory exists
        os.makedirs(target_dir, exist_ok=True)
        filename = f"control_signals_{int(time.time())}.csv"
        filepath = os.path.join(target_dir, filename)
        
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "v_control_sig", "w_control_sig", "error", "error1", "error2", "error3"])
            for i in range(len(self.v_control_sig)):
                writer.writerow([float (i) / self.rate , self.v_control_sig[i], self.w_control_sig[i], self.error[i], self.error1[i], self.error2[i], self.error3[i]])
        
        rospy.loginfo(f"Control and error signals saved to /output/{filename}")
        
        # Run the plotting script
        try:
            plot_script_path = os.path.join(current_dir, "plot_signals.py")
            # Run the plotting script to display the signals previously saved in the CSV file
            subprocess.run(["python3", plot_script_path, filepath], check=True)

        except subprocess.CalledProcessError as e:
            rospy.logerr(f"Failed to run plotting script: {e}")

if __name__ == '__main__':
    try:
        mpp = ModifiedPurePursuit()
        rospy.on_shutdown(mpp.save_plot_signals)
        mpp.run()
    except rospy.ROSInterruptException:
        pass