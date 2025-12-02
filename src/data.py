import numpy as np
from omni.replicator.core import AnnotatorRegistry, BackendDispatch, Writer, WriterRegistry

class MyCustomWriter(Writer):
    def __init__(
        self,
        output_dir,
        rgb=True,
        normals=False,
    ):
        self.version = "0.0.1"
        self.backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self.annotators = []
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        if normals:
            self.annotators.append(AnnotatorRegistry.get_annotator("normals"))
        self._frame_id = 0
        
        # 定义要记录的关节属性路径 - 左右臂共14维数据
        self.joint_paths = [
            # 左臂6个关节
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint1.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint2.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint3.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint4.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint5.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_arm_joint6.state:angular:physics:position",
            # 左夹爪
            "/World/mmp_revB_invconfig_upright_a1x/joints/left_gripper_finger_joint1.drive:linear:physics:targetPosition",
            # 右臂6个关节
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint1.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint2.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint3.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint4.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint5.state:angular:physics:position",
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_arm_joint6.state:angular:physics:position",
            # 右夹爪
            "/World/mmp_revB_invconfig_upright_a1x/joints/right_gripper_finger_joint1.drive:linear:physics:targetPosition"
        ]

    def get_joint_data(self):
        """获取左右臂关节数据并返回为14维数组"""
        import omni.usd
        joint_values = []
        stage = omni.usd.get_context().get_stage()
        
        for joint_path in self.joint_paths:
            try:
                attr = stage.GetAttributeAtPath(joint_path)
                if attr and attr.Get() is not None:
                    joint_values.append(float(attr.Get()))
                else:
                    print(f"Warning: Unable to read joint data from {joint_path}")
                    joint_values.append(0.0)
            except Exception as e:
                print(f"Error reading joint {joint_path}: {e}")
                joint_values.append(0.0)
        
        return np.array(joint_values, dtype=np.float32)

    def write_joint_data_to_txt(self, joint_data, frame_id):
        """将关节数据写入TXT文件"""
        try:
            
            filename = f"joint_data_{frame_id:04d}.txt"
            
           
            np.savetxt(
                f"{self.backend.output_dir}/{filename}",
                joint_data.reshape(1, -1),  # 将一维数组重塑为二维（1行14列）
                fmt='%.6f',
                delimiter=',',
                header='Joint positions: left_arm_joint1, left_arm_joint2, left_arm_joint3, left_arm_joint4, left_arm_joint5, left_arm_joint6, left_gripper_finger_joint1, right_arm_joint1, right_arm_joint2, right_arm_joint3, right_arm_joint4, right_arm_joint5, right_arm_joint6, right_gripper_finger_joint1',  # 文件头
                comments='# ' 
            )
            print(f"[{frame_id}] Writing joint data to {self.backend.output_dir}/{filename}")
            
        except Exception as e:
            print(f"Error writing joint data to file: {e}")

    def write(self, data: dict):
        
        joint_data = self.get_joint_data()
        
        self.write_joint_data_to_txt(joint_data, self._frame_id)
        
        # 原有的RGB和法线数据保存逻辑
        for annotator in data.keys():
            # If there are multiple render products the data will be stored in subfolders
            annotator_split = annotator.split("-")
            render_product_path = ""
            multi_render_prod = 0
            if len(annotator_split) > 1:
                multi_render_prod = 1
                render_product_name = annotator_split[-1]
                render_product_path = f"{render_product_name}/"

            # rgb
            if annotator.startswith("rgb"):
                if multi_render_prod:
                    render_product_path += "rgb/"
                filename = f"{render_product_path}rgb_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename} ..")
                self.backend.write_image(filename, data[annotator])

            # normals
            if annotator.startswith("normals"):
                if multi_render_prod:
                    render_product_path += "normals/"
                filename = f"{render_product_path}normals_{self._frame_id}.png"
                print(f"[{self._frame_id}] Writing {self.backend.output_dir}/{filename} ..")
                colored_data = ((data[annotator] * 0.5 + 0.5) * 255).astype(np.uint8)
                self.backend.write_image(filename, colored_data)

        self._frame_id += 1

    def on_final_frame(self):
        self._frame_id = 0

WriterRegistry.register(MyCustomWriter)