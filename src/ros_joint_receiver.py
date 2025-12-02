#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
import socket
import json
import threading
import time

class ROSJointReceiver:
    def __init__(self, host='127.0.0.1', port=8888):
        # ROS parameters
        self.ros_node_initialized = False
        self.latest_left_joint_positions = None
        self.latest_right_joint_positions = None
        self.left_gripper_position = None
        self.right_gripper_position = None
        self.joint_positions_lock = threading.Lock()
        
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        self.start_socket_server()
        
        self.initialize_ros()
    
    def start_socket_server(self):
        """Start socket server for communication with Isaac Sim"""
        def socket_server():
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            try:
                self.socket.bind((self.host, self.port))
                self.socket.listen(1)
                rospy.loginfo(f"Socket server started on {self.host}:{self.port}")
                
                while not rospy.is_shutdown():
                    try:
                        client_socket, address = self.socket.accept()
                        rospy.loginfo(f"Connection from {address}")
                        self.connected = True
                        
                        # Handle client communication
                        while self.connected and not rospy.is_shutdown():
                            try:
                                # Send joint data when available
                                if (self.latest_left_joint_positions is not None and 
                                    self.latest_right_joint_positions is not None and
                                    self.left_gripper_position is not None and
                                    self.right_gripper_position is not None):
                                    
                                    with self.joint_positions_lock:
                                        # Combine all joint positions: 
                                        # left arm (6) + left gripper (1) + right arm (6) + right gripper (1) = 14 dimensions
                                        combined_positions = (
                                            self.latest_left_joint_positions[:6] + 
                                            [self.left_gripper_position] +
                                            self.latest_right_joint_positions[:6] + 
                                            [self.right_gripper_position]
                                        )
                                        data = {
                                            'joint_positions': combined_positions,
                                            'timestamp': time.time(),
                                            'dimensions': len(combined_positions)
                                        }
                                        json_data = json.dumps(data) + '\n'
                                        client_socket.sendall(json_data.encode('utf-8'))
                                
                                time.sleep(0.016)  # ~60Hz
                                
                            except (socket.error, ConnectionResetError):
                                rospy.logwarn("Client disconnected")
                                self.connected = False
                                break
                                
                    except socket.error as e:
                        rospy.logerr(f"Socket error: {e}")
                        time.sleep(1)
                        
            except Exception as e:
                rospy.logerr(f"Socket server failed: {e}")
        
        # Start socket server thread
        socket_thread = threading.Thread(target=socket_server)
        socket_thread.daemon = True
        socket_thread.start()
    
    def initialize_ros(self):
        """Initialize ROS node and subscribers"""
        rospy.init_node('ros_joint_receiver', anonymous=True)
        
        # Subscribe to left arm joint state topic (6 dimensions)
        rospy.Subscriber(
            '/tabletop/hdas/feedback_arm_left', 
            JointState, 
            self.left_joint_state_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Subscribe to left gripper position topic (1 dimension)
        rospy.Subscriber(
            '/motion_target/target_position_gripper_left',
            JointState,
            self.left_gripper_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Subscribe to right arm joint state topic (6 dimensions)
        rospy.Subscriber(
            '/tabletop/hdas/feedback_arm_right', 
            JointState, 
            self.right_joint_state_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Subscribe to right gripper position topic (1 dimension)
        rospy.Subscriber(
            '/motion_target/target_position_gripper_right',
            JointState,
            self.right_gripper_callback,
            queue_size=1,
            tcp_nodelay=True
        )
        
        self.ros_node_initialized = True
        rospy.loginfo("ROS Joint Receiver initialized")
        rospy.loginfo("Listening to:")
        rospy.loginfo("  /tabletop/hdas/feedback_arm_left (6D)")
        rospy.loginfo("  /motion_target/target_position_gripper_left (1D)")
        rospy.loginfo("  /tabletop/hdas/feedback_arm_right (6D)")
        rospy.loginfo("  /motion_target/target_position_gripper_right (1D)")
        rospy.loginfo("Total dimensions: 14")
    
    def left_joint_state_callback(self, msg):
        """ROS topic callback function for left 6-DOF arm joints"""
        try:
            if len(msg.position) >= 6:
                with self.joint_positions_lock:
                    self.latest_left_joint_positions = list(msg.position)
                
                # Throttled debug info
                if hasattr(self, 'last_left_log_time') and rospy.Time.now() - self.last_left_log_time > rospy.Duration(0.3):
                    rospy.loginfo(f"Received LEFT arm joint positions: {[f'{x:.3f}' for x in msg.position[:6]]}")
                    self.last_left_log_time = rospy.Time.now()
                elif not hasattr(self, 'last_left_log_time'):
                    self.last_left_log_time = rospy.Time.now()
                    
        except Exception as e:
            rospy.logerr(f"Error processing left joint state message: {e}")
    
    def right_joint_state_callback(self, msg):
        """ROS topic callback function for right 6-DOF arm joints"""
        try:
            if len(msg.position) >= 6:
                with self.joint_positions_lock:
                    self.latest_right_joint_positions = list(msg.position)
                
                # Throttled debug info
                if hasattr(self, 'last_right_log_time') and rospy.Time.now() - self.last_right_log_time > rospy.Duration(0.3):
                    rospy.loginfo(f"Received RIGHT arm joint positions: {[f'{x:.3f}' for x in msg.position[:6]]}")
                    self.last_right_log_time = rospy.Time.now()
                elif not hasattr(self, 'last_right_log_time'):
                    self.last_right_log_time = rospy.Time.now()
                    
        except Exception as e:
            rospy.logerr(f"Error processing right joint state message: {e}")
    
    def left_gripper_callback(self, msg):
        """ROS topic callback function for left gripper position"""
        try:
            if len(msg.position) > 0:
                self.left_gripper_position = msg.position[0]  # Take the first position value
                
                # Throttled debug info
                if hasattr(self, 'last_left_gripper_log_time') and rospy.Time.now() - self.last_left_gripper_log_time > rospy.Duration(0.3):
                    rospy.loginfo(f"Received LEFT gripper position: {self.left_gripper_position:.3f}")
                    self.last_left_gripper_log_time = rospy.Time.now()
                elif not hasattr(self, 'last_left_gripper_log_time'):
                    self.last_left_gripper_log_time = rospy.Time.now()
                    
        except Exception as e:
            rospy.logerr(f"Error processing left gripper message: {e}")
    
    def right_gripper_callback(self, msg):
        """ROS topic callback function for right gripper position"""
        try:
            if len(msg.position) > 0:
                self.right_gripper_position = msg.position[0]  # Take the first position value
                
                # Throttled debug info
                if hasattr(self, 'last_right_gripper_log_time') and rospy.Time.now() - self.last_right_gripper_log_time > rospy.Duration(0.3):
                    rospy.loginfo(f"Received RIGHT gripper position: {self.right_gripper_position:.3f}")
                    self.last_right_gripper_log_time = rospy.Time.now()
                elif not hasattr(self, 'last_right_gripper_log_time'):
                    self.last_right_gripper_log_time = rospy.Time.now()
                    
        except Exception as e:
            rospy.logerr(f"Error processing right gripper message: {e}")
    
    def run(self):
        """Main run loop"""
        rospy.loginfo("Starting ROS Joint Receiver...")
        rospy.spin()

def main():
    try:
        receiver = ROSJointReceiver()
        receiver.run()
        
    except KeyboardInterrupt:
        rospy.loginfo("Program interrupted by user")
    except Exception as e:
        rospy.logerr(f"Program runtime error: {e}")
    finally:
        rospy.loginfo("ROS Joint Receiver ended")

if __name__ == "__main__":
    main()