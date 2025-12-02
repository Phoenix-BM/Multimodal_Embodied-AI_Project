#!/usr/bin/env python3

import omni
import carb
from pxr import PhysxSchema, Usd, UsdPhysics, UsdGeom, Gf
import numpy as np
import threading
import socket
import json
import time
import atexit
import carb

class IsaacArmController:
    def __init__(self, host='127.0.0.1', port=8888):
        self.stage = omni.usd.get_context().get_stage()
        self.robot_path = "/World/mmp_revB_invconfig_upright_a1x"
        self.robot_prim = None
        
        # 通信参数
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
        # 数据缓冲与锁
        self.receive_buffer = ""
        self.latest_positions = None
        self.lock = threading.Lock()
        self.stop_flag = False
        
        # 关节映射
        self.left_joint_mapping = {}
        self.right_joint_mapping = {}
        self.left_gripper_joints = []  # 存储左夹爪关节路径
        self.right_gripper_joints = []  # 存储右夹爪关节路径

        # 物体抓取状态
        self.duck_grabbed = {"left": False, "right": False}
        self.duck_01_grabbed = {"left": False, "right": False}

        self.initialize_controller()
        self.start_socket_thread()
        atexit.register(self.cleanup)

    def initialize_controller(self):
        self.robot_prim = self.stage.GetPrimAtPath(self.robot_path)
        if not self.robot_prim.IsValid():
            print(f"[Init] Robot not found at {self.robot_path}")
            return
        print(f"[Init] Found robot: {self.robot_path}")
        self.find_all_joints()

    def find_all_joints(self):
        joints = []
        self.find_joints_recursive(self.robot_prim, joints)

        left_arm_joints = []
        right_arm_joints = []
        
        for joint in joints:
            joint_name = joint.GetName()
            joint_path = joint.GetPath().pathString
            
            if "left_arm_joint" in joint_name:
                left_arm_joints.append((joint_name, joint_path))
            elif "right_arm_joint" in joint_name:
                right_arm_joints.append((joint_name, joint_path))
            elif "left_gripper_finger_joint" in joint_name:
                self.left_gripper_joints.append(joint_path)
                print(f"[Left Gripper] Found gripper joint: {joint_name} -> {joint_path}")
            elif "right_gripper_finger_joint" in joint_name:
                self.right_gripper_joints.append(joint_path)
                print(f"[Right Gripper] Found gripper joint: {joint_name} -> {joint_path}")

        # 排序并映射左臂关节
        left_arm_joints.sort(key=lambda x: int(x[0].replace("left_arm_joint", "")))
        for i, (joint_name, joint_path) in enumerate(left_arm_joints[:6]):
            self.left_joint_mapping[joint_path] = i
            print(f"[Left Joint] {joint_name} -> ROS index {i}")

        # 排序并映射右臂关节
        right_arm_joints.sort(key=lambda x: int(x[0].replace("right_arm_joint", "")))
        for i, (joint_name, joint_path) in enumerate(right_arm_joints[:6]):
            self.right_joint_mapping[joint_path] = i + 7  # 从第8维开始
            print(f"[Right Joint] {joint_name} -> ROS index {i + 7}")
        
        print(f"[Left Joint] Total left arm joints mapped: {len(self.left_joint_mapping)}")
        print(f"[Right Joint] Total right arm joints mapped: {len(self.right_joint_mapping)}")
        print(f"[Left Gripper] Total left gripper joints found: {len(self.left_gripper_joints)}")
        print(f"[Right Gripper] Total right gripper joints found: {len(self.right_gripper_joints)}")

    def find_joints_recursive(self, prim, joints_list):
        if prim.IsA(UsdPhysics.Joint):
            joints_list.append(prim)
        for child in prim.GetChildren():
            self.find_joints_recursive(child, joints_list)

    # Socket 通信
    def socket_loop(self):
        """后台线程循环监听 socket"""
        while not self.stop_flag:
            if not self.connected:
                self.connect_socket()
                time.sleep(1)
                continue

            try:
                data = self.socket.recv(1024)
                if not data:
                    print("[Socket] Disconnected by server")
                    self.connected = False
                    self.socket.close()
                    self.socket = None
                    continue

                self.receive_buffer += data.decode('utf-8')
                while '\n' in self.receive_buffer:
                    line, self.receive_buffer = self.receive_buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        joint_positions = msg.get("joint_positions", [])
                        if len(joint_positions) >= 14:  # 改为检查14维数据
                            with self.lock:
                                self.latest_positions = joint_positions
                        else:
                            print(f"[Socket] Warning: Expected 14 dimensions, got {len(joint_positions)}")
                    except json.JSONDecodeError as e:
                        print(f"[Socket] JSON error: {e}")

            except socket.timeout:
                continue
            except Exception as e:
                print(f"[Socket] Receive error: {e}")
                self.connected = False
                if self.socket:
                    self.socket.close()
                    self.socket = None
            time.sleep(0.01)

    def connect_socket(self):
        """建立连接"""
        try:
            print(f"[Socket] Connecting to {self.host}:{self.port} ...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.settimeout(0.1)
            self.socket.connect((self.host, self.port))
            self.connected = True
            print(f"[Socket] Connected to {self.host}:{self.port}")
        except Exception as e:
            print(f"[Socket] Connection failed: {e}")
            self.connected = False
            if self.socket:
                self.socket.close()
                self.socket = None

    def start_socket_thread(self):
        """启动后台线程"""
        self.socket_thread = threading.Thread(target=self.socket_loop, daemon=True)
        self.socket_thread.start()

    def cleanup(self):
        """释放资源"""
        print("[Cleanup] Closing socket...")
        self.stop_flag = True
        if self.socket:
            try:
                self.socket.close()
            except:
                pass

    # 每帧更新
    def update_robot_pose(self):
        """每帧更新机器人姿态"""
        with self.lock:
            if self.latest_positions is None:
                return
            positions = self.latest_positions.copy()

        # 更新左机械臂关节角度（前6维）
        for joint_path, idx in self.left_joint_mapping.items():
            if idx < len(positions):
                self.set_joint_drive_target(joint_path, positions[idx])

        # 更新右机械臂关节角度（8-13维）
        for joint_path, idx in self.right_joint_mapping.items():
            if idx < len(positions):
                self.set_joint_drive_target(joint_path, positions[idx])

        # 更新左夹爪关节（第7维）
        if len(positions) >= 7:
            left_gripper_value = positions[6]
            self.set_gripper_position(left_gripper_value, self.left_gripper_joints, "left")

        # 更新右夹爪关节（第14维）
        if len(positions) >= 14:
            right_gripper_value = positions[13]
            self.set_gripper_position(right_gripper_value, self.right_gripper_joints, "right")
        
        self.set_torso_joints_to_zero()

    def set_joint_drive_target(self, joint_path, position_rad):
        joint_prim = self.stage.GetPrimAtPath(joint_path)
        if not joint_prim:
            return
        position_deg = np.degrees(position_rad)
        for attr in [
            "state:angular:physics:position",
            "physics:state:angular:position",
            "state:angular:position",
        ]:
            state_attr = joint_prim.GetAttribute(attr)
            if state_attr:
                state_attr.Set(position_deg)
                return

    def set_gripper_position(self, x, gripper_joints, side="left"):
        """设置夹爪位置
        x: 夹爪数值
        gripper_joints: 夹爪关节路径列表
        side: "left" 或 "right" 用于区分左右夹爪
        """
        if len(gripper_joints) < 2:
            print(f"[{side.capitalize()} Gripper] Warning: Need 2 gripper joints, found {len(gripper_joints)}")
            return

        # 计算目标位置
        target_pos_joint1 = x / 100.0 * 0.05
        target_pos_joint2 = -x / 100.0 * 0.05
        target_velocity = 0.5

        # 获取夹爪链接和物体对象
        if side == "left":
            finger1_path = "/World/mmp_revB_invconfig_upright_a1x/left_gripper_finger_link1"
            finger2_path = "/World/mmp_revB_invconfig_upright_a1x/left_gripper_finger_link2"
            duck_path = "/World/obj_duck"  # 鸭子对象
            duck_01_path = "/World/obj_duck_01"  # 杯子对象
        else:
            finger1_path = "/World/mmp_revB_invconfig_upright_a1x/right_gripper_finger_link1"
            finger2_path = "/World/mmp_revB_invconfig_upright_a1x/right_gripper_finger_link2"
            duck_path = "/World/obj_duck"  # 同一个鸭子对象
            duck_01_path = "/World/obj_duck_01"  # 同一个杯子对象

        finger1 = self.stage.GetPrimAtPath(finger1_path)
        finger2 = self.stage.GetPrimAtPath(finger2_path)
        duck_prim = self.stage.GetPrimAtPath(duck_path)
        duck_01_prim = self.stage.GetPrimAtPath(duck_01_path)
        
        if not finger1 or not finger2:
            print(f"[{side.capitalize()} Gripper] Warning: Finger links not found")
            return

        # 计算夹爪中心位置
        pos1 = Gf.Vec3f(*UsdGeom.Xformable(finger1).ComputeLocalToWorldTransform(0).ExtractTranslation())
        pos2 = Gf.Vec3f(*UsdGeom.Xformable(finger2).ComputeLocalToWorldTransform(0).ExtractTranslation())
        gripper_center = (pos1 + pos2) * 0.5

        # 距离判断（检查鸭子和杯子）
        grab_distance = 0.1
        duck_distance = float('inf')
        duck_01_distance = float('inf')
        
        if duck_prim and duck_prim.IsValid():
            duck_world_matrix = UsdGeom.Xformable(duck_prim).ComputeLocalToWorldTransform(0)
            duck_pos = duck_world_matrix.ExtractTranslation()
            duck_distance = (Gf.Vec3f(*duck_pos) - gripper_center).GetLength()
            
        if duck_01_prim and duck_01_prim.IsValid():
            duck_01_world_matrix = UsdGeom.Xformable(duck_01_prim).ComputeLocalToWorldTransform(0)
            duck_01_pos = duck_01_world_matrix.ExtractTranslation()
            duck_01_distance = (Gf.Vec3f(*duck_01_pos) - gripper_center).GetLength()

        # 设置第一个夹爪关节
        joint1_prim = self.stage.GetPrimAtPath(gripper_joints[0])
        if joint1_prim:
            # 设置目标位置
            pos_attr = joint1_prim.GetAttribute("drive:linear:physics:targetPosition")
            if pos_attr:
                pos_attr.Set(target_pos_joint1)
            # 设置目标速度
            vel_attr = joint1_prim.GetAttribute("drive:linear:physics:targetVelocity")
            if vel_attr:
                # if x <= 1 and (duck_distance < grab_distance or duck_01_distance < grab_distance):
                #     pos_attr.Set(0.03)
                #     vel_attr.Set(0.0)
                # else:
                if target_pos_joint1 > 0:
                    vel_attr.Set(target_velocity)
                else:
                    vel_attr.Set(0-target_velocity)

            # 设置力和刚度参数
            for attr_name, value in [("drive:linear:physics:maxForce", 60),
                                   ("drive:linear:physics:damping", 40),
                                   ("drive:linear:physics:stiffness", 500)]:
                attr = joint1_prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(value)

        # 设置第二个夹爪关节
        joint2_prim = self.stage.GetPrimAtPath(gripper_joints[1])
        if joint2_prim:
            # 设置目标位置
            pos_attr = joint2_prim.GetAttribute("drive:linear:physics:targetPosition")
            if pos_attr:
                pos_attr.Set(target_pos_joint2)
            # 设置目标速度
            vel_attr = joint2_prim.GetAttribute("drive:linear:physics:targetVelocity")
            if vel_attr:
                # if x <= 1 and (duck_distance < grab_distance or duck_01_distance < grab_distance):
                #     pos_attr.Set(-0.03)
                #     vel_attr.Set(0.0)
                # else:
                if target_pos_joint2 < 0:
                    vel_attr.Set(0-target_velocity)
                else:
                    vel_attr.Set(target_velocity)

            # 设置力和刚度参数
            for attr_name, value in [("drive:linear:physics:maxForce", 60),
                                   ("drive:linear:physics:damping", 40),
                                   ("drive:linear:physics:stiffness", 500)]:
                attr = joint2_prim.GetAttribute(attr_name)
                if attr:
                    attr.Set(value)

        # # 鸭子抓取逻辑（固定在两夹爪正中间）
        # if duck_prim and duck_prim.IsValid():
        #     if x <= 1 and duck_distance < grab_distance:
        #         # 对齐夹爪和鸭子（固定在正中间）
        #         finger_world_matrix = self.get_finger_mid_transform(finger1, finger2)
        #         duck_xform = UsdGeom.Xformable(duck_prim)
        #         duck_xform.MakeMatrixXform().Set(finger_world_matrix)
        #         self.duck_grabbed[side] = True
        #         #print(f"[{side.capitalize()} Gripper] Duck grabbed, distance={duck_distance:.3f}")
        #     else:
        #         if self.duck_grabbed[side]:
        #             self.duck_grabbed[side] = False
        #         #print(f"[{side.capitalize()} Gripper] Duck released, distance={duck_distance:.3f}")

        # # 杯子抓取逻辑（保持在第一次夹上时的相对位置）
        # if duck_01_prim and duck_01_prim.IsValid():
        #     if x <= 1 and duck_01_distance < grab_distance:
        #         finger_world_matrix = self.get_finger_mid_transform(finger1, finger2)
        #         duck_01_xform = UsdGeom.Xformable(duck_01_prim)
        #         duck_01_xform.MakeMatrixXform().Set(finger_world_matrix)
        #         self.duck_01_grabbed[side] = True
        #         print(f"[{side.capitalize()} Gripper] Duck_01 grabbed, distance={duck_01_distance:.3f}")
        #     else:
        #         if self.duck_01_grabbed[side]:
        #             self.duck_01_grabbed[side] = False
        #         print(f"[{side.capitalize()} Gripper] Duck_01 released, distance={duck_01_distance:.3f}")

    def get_finger_mid_transform(self, finger1, finger2, time_code=Usd.TimeCode.Default()):
        """
        计算两个手指中间的变换矩阵（用于鸭子）
        """
        finger1_xform = UsdGeom.Xformable(finger1)
        finger2_xform = UsdGeom.Xformable(finger2)
        
        matrix1 = finger1_xform.ComputeLocalToWorldTransform(time_code)
        matrix2 = finger2_xform.ComputeLocalToWorldTransform(time_code)
        
        translation1 = matrix1.ExtractTranslation()
        translation2 = matrix2.ExtractTranslation()
        
        translation_mid = (translation1 + translation2) / 2.0
        rotation_mid = matrix1.ExtractRotation()
        
        transform_mid = Gf.Matrix4d().SetRotate(rotation_mid) * Gf.Matrix4d().SetTranslate(translation_mid)
        
        return transform_mid

    def set_torso_joints_to_zero(self):
        """设置所有躯干关节位置为固定值"""
        torso_joint_paths = [
            "/World/mmp_revB_invconfig_upright_a1x/joints/torso_joint1",
            "/World/mmp_revB_invconfig_upright_a1x/joints/torso_joint2", 
            "/World/mmp_revB_invconfig_upright_a1x/joints/torso_joint3"
        ]

        torso_position = [-0.7325999736785889, 1.7037999629974365, 0.9828000068664551]
        count = 0
        
        for joint_path in torso_joint_paths:
            joint_prim = self.stage.GetPrimAtPath(joint_path)
            if joint_prim and joint_prim.IsValid():
                position_deg = np.degrees(torso_position[count])
                state_attr = joint_prim.GetAttribute("state:angular:physics:position")
                if state_attr:
                    state_attr.Set(position_deg)
            else:
                print(f"[Torso] Warning: Joint not found at {joint_path}")
            count = count + 1

_controller = IsaacArmController()

def on_update(e: carb.events.IEvent):
    global _controller
    if _controller:
        _controller.update_robot_pose()

# 获取更新事件流并创建订阅
update_stream = omni.kit.app.get_app_interface().get_update_event_stream()
update_subscription = update_stream.create_subscription_to_pop(on_update, name="arm_controller_update")
print("Dual arm controller started with event subscription.")