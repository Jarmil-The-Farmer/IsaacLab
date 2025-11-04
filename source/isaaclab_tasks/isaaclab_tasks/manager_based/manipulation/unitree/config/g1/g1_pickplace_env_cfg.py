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
from isaaclab_tasks.manager_based.manipulation.unitree.episode_action_provider import EpisodeActionProvider

from ...lib.camera_configs import CameraBaseCfg
from .base_scene_pickplace_redblock import TableRedBlockSceneCfg
from ...lib.robot_configs import G1RobotPresets
from ... import mdp

#from .custom_manager_env import CustomManagerBasedRLEnv

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

    
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
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


def compare_target_joints_zeros(
    #env: CustomManagerBasedRLEnv,
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    #print(env.episode_action_provider.all_joint_names)
    return torch.zeros(env.num_envs, device=env.device)

def compare_target_joints_arm(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_idx: int = 0,
    sigma: float = 0.3
) -> torch.Tensor:
    """
    Reward za podobnost kloubových úhlů robota a teleopu.
    - env: běžící MDP env (ManagerBasedRLEnv)
    - robot_cfg: definice entit (defaultně "robot")
    - step_idx: index kroku, který chceme porovnat s teleop daty
    - sigma: šířka Gaussovské tolerance (větší = měkčí penalizace)
    """

    # 1️⃣ získej cílové klouby z teleopu
    target_joints_dict = env.episode_action_provider.get_action_joints(step_idx)[0]  # jen dict[str, float]
    target_joint_names = list(target_joints_dict.keys())
    target_joint_values = torch.tensor(
        [target_joints_dict[j] for j in target_joint_names],
        device=env.device,
        dtype=torch.float32
    )

    # 2️⃣ aktuální klouby ze všech envů
    robot = env.scene[robot_cfg.name]
    joint_names = list(robot.data.joint_names)
    joint_pos = robot.data.joint_pos  # shape: (num_envs, num_joints)

    # 3️⃣ vytvoř indexy kloubů, které chceme porovnávat
    idxs = [joint_names.index(j) for j in target_joint_names if j in joint_names]
    if len(idxs) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    current = joint_pos[:, idxs]                   # shape (num_envs, N)
    target = target_joint_values.unsqueeze(0)      # shape (1, N)

    # 4️⃣ rozdíl a podobnost
    diff = current - target
    mse = torch.mean(diff ** 2, dim=1)             # průměrná chyba na jeden robot-env

    # 5️⃣ převod na [0–1] (vyšší = větší podobnost)
    # Použijeme Gaussovskou penalizaci -> exp(-(error² / (2σ²)))
    reward = torch.exp(-mse / (2 * sigma ** 2))

    return reward

def compare_target_joints_hand(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    step_idx: int = 0,
    sigma: float = 0.3
) -> torch.Tensor:
    """
    Reward za podobnost kloubových úhlů robota a teleopu.
    - env: běžící MDP env (ManagerBasedRLEnv)
    - robot_cfg: definice entit (defaultně "robot")
    - step_idx: index kroku, který chceme porovnat s teleop daty
    - sigma: šířka Gaussovské tolerance (větší = měkčí penalizace)
    """

    # 1️⃣ získej cílové klouby z teleopu
    target_joints_dict = env.episode_action_provider.get_action_joints(step_idx)[0]  # jen dict[str, float]
    target_joint_names = list(target_joints_dict.keys())
    target_joint_values = torch.tensor(
        [target_joints_dict[j] for j in target_joint_names],
        device=env.device,
        dtype=torch.float32
    )

    # 2️⃣ aktuální klouby ze všech envů
    robot = env.scene[robot_cfg.name]
    joint_names = list(robot.data.joint_names)
    joint_pos = robot.data.joint_pos  # shape: (num_envs, num_joints)

    # 3️⃣ vytvoř indexy kloubů, které chceme porovnávat
    idxs = [joint_names.index(j) for j in target_joint_names if j in joint_names]
    if len(idxs) == 0:
        return torch.zeros(env.num_envs, device=env.device)

    current = joint_pos[:, idxs]                   # shape (num_envs, N)
    target = target_joint_values.unsqueeze(0)      # shape (1, N)

    # 4️⃣ rozdíl a podobnost
    diff = current - target
    mse = torch.mean(diff ** 2, dim=1)             # průměrná chyba na jeden robot-env

    # 5️⃣ převod na [0–1] (vyšší = větší podobnost)
    # Použijeme Gaussovskou penalizaci -> exp(-(error² / (2σ²)))
    reward = torch.exp(-mse / (2 * sigma ** 2))

    return reward



def joint_deviation_ref_l1_teleop(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    joint_mask=None
) -> torch.Tensor:
    """
    Penalizace odchylky kloubových úhlů robota od teleop reference (L1 norma).
    Funguje analogicky k DreamControl joint_deviation_ref_l1.
    """
    asset = env.scene[asset_cfg.name]
    motion_times = env.episode_length_buf * env.step_dt + getattr(env, "start_motion_times", 0.0)
    motion_times = motion_times.to(device=env.device, dtype=torch.float32)

    motion_res = env.episode_action_provider.get_motion_state(torch.zeros_like(motion_times), motion_times)
    ref_joint_pos = motion_res["dof_pos"]  # [num_envs, num_joints]
    cur_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    diff = (cur_joint_pos - ref_joint_pos) * (1.0 if joint_mask is None else torch.tensor(joint_mask, device=env.device).unsqueeze(0))
    return torch.sum(torch.abs(diff), dim=1)


def get_keypts_from_robot(joint_angles: torch.Tensor, joint_names: list[str], pk2_robot) -> torch.Tensor:
    """
    joint_angles: [num_envs, num_joints]
    joint_names:  list of joint names length num_joints
    pk2_robot:    kinematický model s metodou forward_kinematics(q_dict)
    Výstup: keypts tensor [num_envs, num_keypts, 3]
    """
    # vytvoř dict {name: tensor[num_envs]}
    q_dict = {name: joint_angles[:, i] for i, name in enumerate(joint_names)}

    # FK – vrátí dict link_name -> TransformBatch (s metodou get_matrix())
    tf_dict = pk2_robot.forward_kinematics(q_dict)

    num_envs = joint_angles.shape[0]
    num_keypts = len(tf_dict)
    keypts = torch.zeros((num_envs, num_keypts, 3), device=joint_angles.device, dtype=torch.float32)

    cntr = 0
    for name, tf in tf_dict.items():
        mat = tf.get_matrix()  # očekává se [num_envs, 4,4]
        t = mat[:, :3, -1]     # translace
        keypts[:, cntr, :] = t
        cntr += 1

    return keypts

def keypts_deviation_ref_l2_teleop(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    keypts_mask: list[float] | None = None
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    motion_times = env.episode_length_buf * env.step_dt + getattr(env, "start_motion_times", 0.0)
    motion_times = motion_times.to(device=env.device, dtype=torch.float32)

    # referenční klouby z teleopu
    motion_res = env.episode_action_provider.get_motion_state(torch.zeros_like(motion_times), motion_times)
    ref_joint_pos = motion_res["dof_pos"]  # [num_envs, num_joints]

    # aktuální klouby robota
    cur_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]  # [num_envs, num_joints]

    # keypts pro aktuální robot
    keypts_robot = get_keypts_from_robot(cur_joint_pos, env.joint_names, env.pk2_robot)

    # keypts pro referenční trajektorii
    keypts_ref = get_keypts_from_robot(ref_joint_pos, env.joint_names, env.pk2_robot)

    # pokud je maska – použij
    if keypts_mask is not None:
        mask = torch.tensor(keypts_mask, device=env.device).unsqueeze(0).unsqueeze(-1)  # [1, num_keypts,1]
        diff = (keypts_ref - keypts_robot) * mask
    else:
        diff = (keypts_ref - keypts_robot)

    # L2 norma přes osy, poté součet přes keypt dim
    err = torch.norm(diff, dim=2)  # [num_envs, num_keypts]
    return torch.sum(err, dim=1)  # [num_envs]

def position_tracking_error_teleop(env, asset_cfg=SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    motion_times = env.episode_length_buf * env.step_dt + getattr(env, "start_motion_times", 0.0)
    motion_times = motion_times.to(device=env.device, dtype=torch.float32)
    motion_res = env.episode_action_provider.get_motion_state(torch.zeros_like(motion_times), motion_times)

    ref_root_pos = motion_res.get("root_pos", torch.zeros_like(asset.data.root_pos_w))
    cur_root_pos = asset.data.root_pos_w - env.scene.env_origins
    return torch.norm(ref_root_pos - cur_root_pos, dim=1)


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Distance hand → cube
    # reach_cube = RewTerm(
    #     func=mdp.distance_hand_object,
    #     weight=-3.0,
    #     params={"asset_cfg": SceneEntityCfg("object")},
    # )

    # compare_target_joints_arm = RewTerm(func=compare_target_joints_arm, weight=1.0)
    # compare_target_joints_hand = RewTerm(func=compare_target_joints_hand, weight=2.0)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)
    alive_reward = RewTerm(func=mdp.is_alive, weight=1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"])})
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7,params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_.*", ".*_knee_joint"])})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)

            
    joint_deviation_ref = RewTerm(
            func=joint_deviation_ref_l1_teleop,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")})
    
    keypts_deviation_ref = RewTerm(
            func=keypts_deviation_ref_l2_teleop,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot")})
        
        
    position_tracking_error = RewTerm(
            func=position_tracking_error_teleop,
            weight=-0.2,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )


    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=5.0)

    # TODO calc best middle distance
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.845}, weight=10.0)

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

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # object_dropping = DoneTerm(
    #     func=mdp.root_height_below_minimum, params={"minimum_height": 0.6, "asset_cfg": SceneEntityCfg("object")}
    # )

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
        #self.sim.render_interval = 2

        self.decimation = 4
        self.episode_length_s = 10.0 # def 5
        # simulation settings
        self.sim.dt = 0.005  # 100Hz
        #self.sim.render_interval = self.decimation

        self.episode_action_provider: EpisodeActionProvider
