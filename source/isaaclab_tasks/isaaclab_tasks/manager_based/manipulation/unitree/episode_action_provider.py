import json
from typing import List, Optional
import numpy as np
import time

import torch

from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv

class EpisodeActionProvider:

    def __init__(self, env: ManagerBasedRLEnv):
        self.env = env
        self.robot_data: ArticulationData = self.env.scene["robot"].data

        self.all_joint_names = self.robot_data.joint_names
        self.joint_to_index = {name: i for i, name in enumerate(self.all_joint_names)}

        self.left_arm_joint = [        
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint"]
        self.right_arm_joint = [        
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint"]
        self.left_arm_joint_indices = [self.joint_to_index[name] for name in self.left_arm_joint]
        self.right_arm_joint_indices = [self.joint_to_index[name] for name in self.right_arm_joint]

        self.left_hand_joint = [
            "L_pinky_proximal_joint",
            "L_ring_proximal_joint",
            "L_middle_proximal_joint",
            "L_index_proximal_joint",
            "L_thumb_proximal_pitch_joint",
            "L_thumb_proximal_yaw_joint",
        ]
        self.right_hand_joint = [
            "R_pinky_proximal_joint",
            "R_ring_proximal_joint",
            "R_middle_proximal_joint",
            "R_index_proximal_joint",
            "R_thumb_proximal_pitch_joint",
            "R_thumb_proximal_yaw_joint",
        ]
        self.left_hand_joint_indices = [self.joint_to_index[name] for name in self.left_hand_joint]
        self.right_hand_joint_indices = [self.joint_to_index[name] for name in self.right_hand_joint]

        self.episode_data = []

    def get_sim_robot_joints_data(self) -> dict[str, float]:
        env_index = 0
        return {name: self.robot_data.joint_pos[env_index].detach().cpu().numpy()[i] for i, name in enumerate(self.all_joint_names)}

    def load_episode_data(self, json_file: str):
        right_arm_actions, right_hand_actions = self.load_robot_right_side_data_from_json(json_file)
        self.episode_data = []
        for i in range(len(right_arm_actions)):
            self.episode_data.append({
                "right_arm_actions": right_arm_actions[i],
                "right_hand_actions": right_hand_actions[i],
            })

    def get_episode_len(self) -> int:
        return len(self.episode_data)

    def has_next_step(self, step_idx: int):
        return step_idx < len(self.episode_data)

    def get_action_joints(self, step_idx: int) -> tuple[dict[str, float], dict[str, float]]:
        # if step_idx >= len(self.episode_data):
        #     raise IndexError("Step index out of range for episode data.")
        step_idx %= self.get_episode_len()

        right_arm_action = self.episode_data[step_idx]["right_arm_actions"]
        right_hand_action = self.episode_data[step_idx]["right_hand_actions"]

        joints = {}
        right_arm_action_dict = {name: right_arm_action[i] for i, name in enumerate(self.right_arm_joint) }
        right_hand_action_dict = {name: right_hand_action[i] for i, name in enumerate(self.right_hand_joint) }

        joints.update(right_arm_action_dict)
        joints.update(right_hand_action_dict)

        return right_arm_action_dict, right_hand_action_dict

    def load_robot_right_side_data_from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            content = json.load(f)

        info = content.get("info", {})
        text = content.get("text", {})
        data = content.get("data", [])


        if not data:
            raise ValueError("data is None")

        robot_action = []
        hand_action = []
        sim_state_json_list=[]
        sim_state_list=[]
        sim_task_name_list=[]

        right_arm_actions = []
        right_hand_actions = []

        for item in data:
            action = item.get("actions", {})
            if not action:
                raise ValueError("data not have action")

            left_arm = action.get("left_arm", {})
            right_arm = action.get("right_arm", {})
            left_arm_action = np.array(left_arm.get("qpos", []))
            right_arm_action = np.array(right_arm.get("qpos", []))
            left_right_arm = np.concatenate([left_arm_action, right_arm_action])

            right_arm_actions.append(right_arm_action)

            left_hand = action.get("left_ee", {})
            right_hand = action.get("right_ee", {})
            left_hand_action = np.array(left_hand.get("qpos", []))
            right_hand_action = np.array(right_hand.get("qpos", []))
            left_right_hand = np.concatenate([right_hand_action, left_hand_action])

            right_hand_actions.append(right_hand_action)

            robot_action.append(left_right_arm)
            hand_action.append(left_right_hand)
            sim_state_json = item.get("sim_state", "{}")
            if not sim_state_json:
                raise ValueError("sim_state is None")
            sim_state_json_list.append(sim_state_json)
            # sim_state = parse_nested_sim_state(sim_state_json)
            sim_state_raw = sim_state_json.get("init_state","{}")
            task_name = sim_state_json.get("task_name","")
            if task_name=="":
                raise ValueError("task_name is None")
            # 如果 sim_state 是 JSON 字符串则解析
            if not sim_state_raw:
                raise ValueError("sim_state_raw is None")
            if isinstance(sim_state_raw, str):
                sim_state_dict = json.loads(sim_state_raw)
            else:
                sim_state_dict = sim_state_raw
            sim_state = convert_nested_lists_to_tensor(sim_state_dict)
            sim_state_list.append(sim_state)
            sim_task_name_list.append(task_name)

        return right_arm_actions, right_hand_actions
        #return robot_action, hand_action, sim_state_list,sim_task_name_list,sim_state_json_list


    def get_motion_state(self, motion_ids: torch.Tensor, motion_times: torch.Tensor):
        """
        Podobně jako motion_lib.get_motion_state() v DreamControl.
        Vstupy:
            motion_ids:   (num_envs,) tensor – identifikátor motionu (u tebe ignorujeme, použijeme jednu epizodu)
            motion_times: (num_envs,) tensor – aktuální čas každého envu (v sekundách)
        Výstup:
            dict s kloubovými úhly pro každý env ve tvaru:
                {
                    "dof_pos": tensor[num_envs, num_joints],
                }
        """
        num_envs = motion_times.shape[0]
        device = motion_times.device
        episode_len = self.get_episode_len()
        if episode_len == 0:
            raise RuntimeError("Episode data nejsou načtena!")

        # převod času na index v záznamu (lineárně)
        step_dt = getattr(self.env, "step_dt", 1.0)
        indices = (motion_times / step_dt).long() % episode_len  # každý env má vlastní posun

        joint_count = len(self.all_joint_names)
        dof_pos = torch.zeros((num_envs, joint_count), device=device, dtype=torch.float32)

        for env_i in range(num_envs):
            idx = int(indices[env_i].item())
            # vezmi klouby z teleop dat
            right_arm_dict, right_hand_dict = self.get_action_joints(idx)
            ref_joint_dict = {**right_arm_dict, **right_hand_dict}

            for j_name, j_value in ref_joint_dict.items():
                if j_name in self.joint_to_index:
                    j_idx = self.joint_to_index[j_name]
                    dof_pos[env_i, j_idx] = float(j_value)

        return {"dof_pos": dof_pos}

    def get_reference_joint_positions(self, motion_times: torch.Tensor):
        """Zjednodušený wrapper: vrátí tensor kloubových úhlů podle časů všech envů."""
        return self.get_motion_state(torch.zeros_like(motion_times), motion_times)["dof_pos"]


def convert_nested_lists_to_tensor(obj):
    if isinstance(obj, dict):
        return {k: convert_nested_lists_to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # list[list[number]]
        if all(isinstance(item, list) and all(isinstance(x, (int, float)) for x in item) for item in obj):
            return torch.tensor(obj, dtype=torch.float32)
        else:
            return [convert_nested_lists_to_tensor(item) for item in obj]
    else:
        return obj
