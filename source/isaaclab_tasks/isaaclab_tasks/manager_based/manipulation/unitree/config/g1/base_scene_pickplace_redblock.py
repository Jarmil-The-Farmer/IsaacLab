# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0     
"""
public base scene configuration module
provides reusable scene element configurations, such as tables, objects, ground, lights, etc.
"""
import isaaclab.sim as sim_utils
from isaaclab.assets import  AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import os
project_root = "/home/jarmil/unitree_sim_isaaclab2"
@configclass
class TableRedBlockSceneCfg(InteractiveSceneCfg): # inherit from the interactive scene configuration class
    """object table scene configuration class
    defines a complete scene containing robot, object, table, etc.
    """
      # 1. room wall configuration - simplified configuration to avoid rigid body property conflicts
    # room_walls = AssetBaseCfg(
    #     prim_path="/World/envs/env_.*/Room",
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=[0.0, 0.0, 0],  # room center point
    #         rot=[1.0, 0.0, 0.0, 0.0]
    #     ),
    #     spawn=UsdFileCfg(
    #         usd_path=f"{project_root}/assets/objects/small_warehouse_digital_twin/small_warehouse_digital_twin.usd",
    #     ),
    # )


    # 1. table configuration
    packing_table = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable",    # table in the scene
        init_state=AssetBaseCfg.InitialStateCfg(pos=[-4.3,-4.2,-0.2],   # initial position [x, y, z]
                                                rot=[1.0, 0.0, 0.0, 0.0]), # initial rotation [x, y, z, w]
        spawn=UsdFileCfg(
            usd_path=f"{project_root}/assets/objects/table_with_yellowbox.usd",    # table model file
            #rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),    # set to kinematic object
            
        ),
    )
    # Object
    # 2. object configuration
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.25, -4.03, 0.84],
            rot=[1, 0, 0, 0]
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.03,           # ↑ o trochu víc tlumení → míň ujíždí z úchopu
                angular_damping=0.03,          # ↑ snáz “sedne” mezi prsty
                solver_position_iteration_count=24,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.40),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.0015,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0), metallic=0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="min",    # s nízkotřecím stolem stále klouže
                restitution_combine_mode="min",
                static_friction=2.8,            # ↑ pevnější grip
                dynamic_friction=2.2,           # ↑ menší skluz v ruce
                restitution=0.0,
            ),
        ),
    )

    target_square = RigidObjectCfg(
        prim_path="/World/envs/env_.*/TargetSquare",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-4.195, -3.945, 0.795],  # pozice na stole
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(0.08, 0.08, 0.001),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
        ),
    )


    # Ground plane
    # 3. ground configuration
    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",    # ground in the scene
    #     spawn=GroundPlaneCfg( ),    # ground configuration
    # )

    # Lights
    # 4. light configuration
    light = AssetBaseCfg(
        prim_path="/World/light",   # light in the scene
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), # light color (white)
                                     intensity=3000.0),    # light intensity
    )



    # world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera",
    #                                                 pos_offset=(-4.1, -4.9, 1.8),
    #                                                 rot_offset=( -0.3173,0.94833, 0.0, 0.0))

    wolrd_camera = None

