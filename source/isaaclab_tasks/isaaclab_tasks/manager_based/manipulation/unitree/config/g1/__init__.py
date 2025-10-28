# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import os

from . import custom_manager_env
from . import agents

gym.register(
    id="Isaac-Unitree-G1-PickPlace-v0",
    entry_point=f"{__name__}.custom_manager_env:CustomManagerBasedRLEnv",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_pickplace_env_cfg:UnitreeG1PickPlaceEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg_head_cam.yaml",
    },
    disable_env_checker=True,
)