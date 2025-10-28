# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import tempfile
import torch

import carb
import isaaclab.controllers.utils as ControllerUtils
import isaaclab.envs.mdp as base_mdp
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
#from isaaclab.controllers.pink_ik import NullSpacePostureTask, PinkIKControllerCfg
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import ManusViveCfg, OpenXRDeviceCfg, XrCfg
from isaaclab.devices.openxr.retargeters.humanoid.fourier.gr1t2_retargeter import GR1T2RetargeterCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
#from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from ...lib.camera_configs import CameraBaseCfg
from .base_scene_pickplace_redblock import TableRedBlockSceneCfg
from ...lib.robot_configs import G1RobotPresets
from ... import mdp

marker_cfg = FRAME_MARKER_CFG.copy()
marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
marker_cfg.prim_path = "/Visuals/FrameTransformer"



##
# Scene definition
##
@configclass
class PickPlaceSceneCfg(TableRedBlockSceneCfg):
    """Scene with table, red block and G1 robot."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=(-4.2, -3.7, 0.76),
        init_rot=(0.7071, 0, 0, -0.7071),
        #self_collisions=True,
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/head_link",
        debug_vis=True,
        visualizer_cfg=marker_cfg, # enable frame marker visualization
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/right_hand_base_link",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.0, -0.05, 0.0],
                ),
            ),
        ],
    )

    # Add target area (green square) on the table
    # target = SceneEntityCfg(
    #     "target_square",
    # )

    head_camera = CameraBaseCfg.get_camera_config(
        width=128,
        height=128,
    )

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=GroundPlaneCfg(),
    )

    # Lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        # joint_names=JointNamesOrder,
        # preserve_order=True,
        joint_names=[
            "right_shoulder.*",
            "right_elbow.*",
            "right_wrist.*",
            "R_.*",
        ],
        scale=0.5,
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group with state values."""

        head_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("head_camera"), "data_type": "rgb"},
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance hand → cube
    reach_cube = RewTerm(
        func=mdp.distance_hand_object,
        weight=-3.0,
        params={"asset_cfg": SceneEntityCfg("object")},
    )


    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # TODO calc best middle distance
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.845}, weight=15.0)

    # place = RewTerm(
    #     func=mdp.is_object_placed,
    #     weight=40.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "target_square_cfg": SceneEntityCfg("target_square")},
    # )

    # move_to_target = RewTerm(
    #     func=mdp.object_target_distance,
    #     weight=20.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "target_square_cfg": SceneEntityCfg("target_square"), "minimal_height": 0.05},
    # )



    # object_goal_tracking = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=16.0,
    # )

    # object_goal_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_distance,
    #     params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
    #     weight=5.0,
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("object")}
    )

    #success = DoneTerm(func=mdp.task_done_pick_place)


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": [-0.01, 0.01],
                "y": [-0.01, 0.01],
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class UnitreeG1PickPlaceEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Unitree G1 Pick and Place environment."""

    # Scene settings
    scene: PickPlaceSceneCfg = PickPlaceSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    terminations: TerminationsCfg = TerminationsCfg()
    events = EventCfg()

    rewards = RewardsCfg()

    # Unused managers
    commands = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 6
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1 / 120  # 120Hz
        self.sim.render_interval = 2



        self.decimation = 2
        self.episode_length_s = 5 # def 5
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation
