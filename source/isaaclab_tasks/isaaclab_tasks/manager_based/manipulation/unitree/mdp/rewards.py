# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""

    object: RigidObject = env.scene[object_cfg.name]
    res =  torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)

    return res


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_target_distance(
    env: ManagerBasedRLEnv,
    std: float = 0.3,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_square_cfg: SceneEntityCfg = SceneEntityCfg("target_square"),
    minimal_height: float = 0.04,
) -> torch.Tensor:
    """
    Reward function that encourages the object to move closer to the target square
    using tanh kernel — active only when the object is lifted.
    """
    # Extract entities
    object: RigidObject = env.scene[object_cfg.name]
    target_square: RigidObject = env.scene[target_square_cfg.name]

    # Get world positions
    object_pos_w = object.data.root_pos_w           # (num_envs, 3)
    target_pos_w = target_square.data.root_pos_w    # (num_envs, 3)

    # Compute Euclidean distance between object and target
    dist = torch.norm(object_pos_w - target_pos_w, dim=1)  # (num_envs,)

    # Check if object is lifted
    lifted_res = object_is_lifted(env, minimal_height, object_cfg=object_cfg)  # (num_envs,)

    # Compute reward only if lifted
    reward = 1 - torch.tanh(dist / std)
    reward = reward * lifted_res.float()  # zero reward if not lifted

    return reward

def is_object_placed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_square_cfg: SceneEntityCfg = SceneEntityCfg("target_square"),
    xy_tol: float = 0.03,
    z_tol: float = 0.03,
    vel_tol: float = 0.05,
) -> torch.Tensor:
    """
    Returns a tensor (num_envs,) with 1.0 if the object is placed within the target square
    and 0.0 otherwise.

    Conditions:
    - XY position of object center within xy_tol of target center
    - Z distance within z_tol
    - Object velocity below vel_tol (object is stable)
    """
    # Extract world positions and velocities
    object: RigidObject = env.scene[object_cfg.name]
    target_square: RigidObject = env.scene[target_square_cfg.name]

    obj_pos = object.data.root_pos_w        # (num_envs, 3)
    obj_vel = object.data.root_vel_w        # (num_envs, 3)
    tgt_pos = target_square.data.root_pos_w # (num_envs, 3)

    # Compute distances
    xy_dist = torch.norm(obj_pos[:, :2] - tgt_pos[:, :2], dim=-1)  # (num_envs,)
    z_dist = torch.abs(obj_pos[:, 2] - tgt_pos[:, 2])              # (num_envs,)
    vel_mag = torch.norm(obj_vel, dim=-1)                          # (num_envs,)

    # Convert to float masks
    xy_ok = (xy_dist < xy_tol).float()
    z_ok = (z_dist < z_tol).float()
    still = (vel_mag < vel_tol).float()

    # Combine (1.0 if all three true)
    is_success = xy_ok * z_ok * still

    return is_success


def distance_hand_object(env, env_ids=None, asset_cfg=None, palm_link_name="right_hand_base_link"):
    """Euklidovská vzdálenost mezi dlaní (base link ruky) a kostkou."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # najdeme index base linku ruky
    try:
        link_id = env.scene["robot"].data.body_names.index(palm_link_name)
    except ValueError:
        # fallback – pokud se název změní, vezmeme první prst jako approx
        print(f"[WARN] palm_link_name '{palm_link_name}' not found, falling back to R_index_proximal")
        link_id = env.scene["robot"].data.body_names.index("R_index_proximal")

    palm_pos = env.scene["robot"].data.body_pos_w[env_ids, link_id, :]  # (N,3)
    obj_pos = env.scene[asset_cfg.name].data.body_pos_w[env_ids, 0, :]  # (N,3)

        # vypočítej vzdálenost
    distances = torch.norm(palm_pos - obj_pos, dim=-1)  # (N,)

    # --- DEBUG výpis ---
    # každých pár kroků (ne každou ms), aby to nebylo zahlcené
    if getattr(env, "common_step_counter", 0) % 10 == 0:
        dist_cpu = distances.detach().cpu().numpy()
        print("\n[DEBUG] Distance hand ↔ object per env:")
        for i, d in zip(env_ids.tolist(), dist_cpu):
            print(f"  Env {i:02d}: {d:.4f} m")

    return distances
