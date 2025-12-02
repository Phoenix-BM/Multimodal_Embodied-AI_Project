#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert custom HDF5 robot dataset (e.g. /data3/pi0.5/10-10/*.hdf5)
to LeRobot dataset format, compatible with OpenPi / Pi0.5 / SmolVLA fine-tuning.

Action = Δstate (state[t+1] - state[t])
"""

import h5py
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# ---------------------- 配置区 ----------------------
DATA_DIR = Path("/data3/pi0.5/10-10")   # 你的原始 hdf5 数据目录
REPO_NAME = "CRRaphael/aloha10-10"      # 输出数据集名，可改
FPS = 10                                # 采样频率
# ----------------------------------------------------


def convert_one_episode(h5_path, dataset):
    with h5py.File(h5_path, "r") as f:
        if "action" not in f:
            print(f"[ERROR] {h5_path.name} 缺少 'action'，跳过。")
            return

        # 保留全部14维（左右臂）
        positions = np.array(f["action"], dtype=np.float32)  # (N,14)
        if positions.shape[1] < 14:
            print(f"[ERROR] {h5_path.name} action维度异常: {positions.shape}")
            return

        states = positions
        actions = np.diff(states, axis=0, prepend=states[0:1])

        # 图像读取
        try:
            cam_front = np.array(f["observations/images/cam_high"], dtype=np.uint8)
            cam_left = np.array(f["observations/images/cam_left_wrist"], dtype=np.uint8)
            cam_right = np.array(f["observations/images/cam_right_wrist"], dtype=np.uint8)
        except KeyError as e:
            print(f"[ERROR] {h5_path.name} 缺少图像键: {e}")
            return

        num_frames = min(len(cam_front), len(states))
        task_text = "Put the cubes into the plate"

        for i in range(num_frames):
            dataset.add_frame(
                {
                    "image": cam_front[i],
                    "left_image": cam_left[i],
                    "right_image": cam_right[i],
                    "state": states[i],
                    "action": actions[i],
                    "task": task_text,
                }
            )
        dataset.save_episode()
        print(f"[OK] {h5_path.name} 转换完成 ({num_frames} 帧)")

def main():
    data_dir = DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {data_dir}")

    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"删除已有数据集目录: {output_path}")
        shutil.rmtree(output_path)

    # 创建 LeRobot 数据集结构
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="Aloha",
        fps=FPS,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "left_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "right_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    h5_files = sorted(data_dir.glob("*.hdf5"))
    print(f"在 {data_dir} 中找到 {len(h5_files)} 个 HDF5 文件。")

    for h5_file in tqdm(h5_files, desc="转换中"):
        convert_one_episode(h5_file, dataset)

    print(f"\n数据集转换完成！保存路径：{output_path}")
    print(f"可以用以下方式加载：\nfrom lerobot.common.datasets.lerobot_dataset import LeRobotDataset\nLeRobotDataset.load('{REPO_NAME}')")

    pub_name = REPO_NAME
    print(f"正在推送数据集到 Hugging Face Hub: {pub_name}")
    dataset.push_to_hub(
        tags=["robot", "galaxea", "rl", "imitation-learning"],
        private=True,  # 设为 True 表示私有，False 表示公开
        push_videos=True,
        license="apache-2.0",
    )
    print(f"数据集已推送到: https://huggingface.co/datasets/{pub_name}")



if __name__ == "__main__":
    main()
