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

    # enable reward if lifted at least once in the episode
    # print("Lifted res:", res)
    if False:
        print("Lifted weight set")
        reward_term = env.reward_manager.get_term_cfg("move_to_target")
        reward_term.weight = 10.0

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
    """Reward function that encourages the object to move closer to the target square using tanh kernel."""
    # Extract entities
    object: RigidObject = env.scene[object_cfg.name]
    target_square: RigidObject = env.scene[target_square_cfg.name]

    # Get world positions
    object_pos_w = object.data.root_pos_w           # (num_envs, 3)
    target_pos_w = target_square.data.root_pos_w    # (num_envs, 3)

    # Compute Euclidean distance between object and target
    dist = torch.norm(object_pos_w - target_pos_w, dim=1)  # (num_envs,)

    lifted_res = object_is_lifted(env, minimal_height, object_cfg=object_cfg)
    # Reward only if the object is lifted
    dist = dist * lifted_res

    print("Distance:", dist)

    # Convert distance to a reward in range (0, 1)
    return 1 - torch.tanh(dist / std)


def is_object_placed(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    target_square_cfg: SceneEntityCfg = SceneEntityCfg("target_square"),
    xy_tol: float = 0.03,
    z_tol: float = 0.03,
    vel_tol: float = 0.05,
    env_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Check if the object is successfully placed within the target square.

    Success is defined as:
    - Object center within (xy_tol, z_tol) of target center.
    - Object velocity below vel_tol (object is stable).
    """
    # Select environment IDs (default: all envs)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Extract world positions and velocities
    object: RigidObject = env.scene[object_cfg.name]
    target_square: RigidObject = env.scene[target_square_cfg.name]

    obj_pos = object.data.root_pos_w[env_ids]     # (num_envs, 3)
    obj_vel = object.data.root_vel_w[env_ids]     # (num_envs, 3)
    tgt_pos = target_square.data.root_pos_w[env_ids]  # (num_envs, 3)

    # Check position tolerance
    xy_ok = torch.norm(obj_pos[:, :2] - tgt_pos[:, :2], dim=-1) < xy_tol
    z_ok = torch.abs(obj_pos[:, 2] - tgt_pos[:, 2]) < z_tol

    # Check if object is nearly stationary
    still = torch.norm(obj_vel, dim=-1) < vel_tol

    # Combine all success conditions
    is_success = (xy_ok & z_ok & still).bool()

    return is_success