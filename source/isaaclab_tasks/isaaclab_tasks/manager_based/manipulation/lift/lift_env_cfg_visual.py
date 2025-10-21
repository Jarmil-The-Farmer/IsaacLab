# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.camera.tiled_camera_cfg import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    hand_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/camera_hand",
        #offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5)),
        width=128,
        height=128,
        data_types=["rgb"], # Říkáme, že chceme RGB data
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            #horizontal_aperture=20.955,
            #clipping_range=(0.1, 20.0),
            horizontal_aperture=53.7,
            clipping_range=(0.01, 1.0e5),
        ),
        offset=TiledCameraCfg.OffsetCfg(
                pos=(0.05, 0, 0.05), rot=(0.70441603, -0.06162842, -0.06162842, 0.70441603), convention="ros"
        ),
    )

    top_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/top_camera",
        #offset=CameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5)),
        width=128,
        height=128,
        data_types=["rgb"], # Říkáme, že chceme RGB data
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            #horizontal_aperture=20.955,
            #clipping_range=(0.1, 20.0),
            #horizontal_aperture=53.7,
            clipping_range=(0.01, 2), # clip camera at 2 meters - so does not see other envs
        ),
        # 
        offset=TiledCameraCfg.OffsetCfg(
            # rot form sim: x=0 y=45 z=90 # (click on orient icon in isaac sim to see quat values from degrees)
            pos=(1.2, 0, 0.7), rot=(0.65328, 0.2706, 0.2706, 0.65328), convention="opengl",

        ),
    )

    target_square = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetSquare",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.6, 0.0, 0.0],  # pozice na stole
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 0.15, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0)),
        ),
    )



##
# MDP settings
##

def debug_print_on_reset(env, env_ids):
    pass
    # seber celé pozorování v okamžiku resetu
    # obs_dict = env.observation_manager.compute()
    # try:
    #     print("---- DEBUG OBS (after reset) ----")
    #     for k, v in obs_dict.items():
    #         print(f"{k:>10}: {tuple(v.shape)}  dtype={v.dtype}")
    #     print("---------------------------------")
    # except Exception as e:
    #     print(f"[debug_print_obs_on_reset] failed: {e}")


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=False, # enable visualization of the command target
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        hand_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("hand_camera"), "data_type": "rgb"},
        )

        top_camera = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("top_camera"), "data_type": "rgb"},
        )


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
            #self.concatenate_dim = 2

    # groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )
    
    debug_obs_on_reset = EventTerm(func=debug_print_on_reset, mode="reset")


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)

    # TODO calc best middle distance
    lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.08}, weight=15.0)

    # place = RewTerm(
    #     func=mdp.is_object_placed,
    #     weight=40.0,
    #     params={"object_cfg": SceneEntityCfg("object"), "target_square_cfg": SceneEntityCfg("target_square")},
    # )

    move_to_target = RewTerm(
        func=mdp.object_target_distance,
        weight=20.0,
        params={"object_cfg": SceneEntityCfg("object"), "target_square_cfg": SceneEntityCfg("target_square"), "minimal_height": 0.08},
    )



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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

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
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 20000}
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 20000}
    )


##
# Environment configuration
##

import torch
@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=128, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 6 # def 5
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
