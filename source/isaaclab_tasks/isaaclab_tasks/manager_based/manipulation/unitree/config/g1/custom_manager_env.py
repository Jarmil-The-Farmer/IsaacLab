import torch
from isaaclab.assets.articulation.articulation_data import ArticulationData
from isaaclab.envs import ManagerBasedRLEnv
# from .joint_pos_env_cfg_visual import FrankaCubeLiftEnvCfg
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg
from .g1_pickplace_env_cfg import UnitreeG1PickPlaceEnvCfg
from ...episode_action_provider import EpisodeActionProvider
from ...rerun_visualizer import RerunLogger
import pytorch_kinematics as pk2


class CustomManagerBasedRLEnv(ManagerBasedRLEnv):

    def __init__(self, cfg: UnitreeG1PickPlaceEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.rerun_logger = RerunLogger()

        self.episode_action_provider = EpisodeActionProvider(self)
        cfg.episode_action_provider = self.episode_action_provider

        json_data_file = "/home/jarmil/datasets/test_new3/data/episode_0000/data.json"
        self.episode_action_provider.load_episode_data(json_data_file)

        urdf_path = "/home/jarmil/DreamControl/Training/HumanoidVerse/humanoidverse/data/robots/g1/g1_29dof.urdf"
        self.pk2_robot = pk2.build_chain_from_urdf(open(urdf_path).read())

        self.joint_names = ['left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint', 'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint', 'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint', 'waist_yaw_joint', 'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint', 'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint', 'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint']


    def step(self, action):
        step_idx = self.common_step_counter

        if step_idx != 0 and step_idx % self.episode_action_provider.get_episode_len() == 0:
            print("Episode data does not have next step, resetting")
            self.reset()


        # --- zavolej standardní krok ---
        obs, rew, terminated, truncated, info = super().step(action)
        #print("obs", obs)
        # info looks like:
        # {'log': {'Episode_Reward/reaching_object': tensor(0.0829, device='cuda:0'), 'Episode_Reward/lifting_object': tensor(0.1852, device='cuda:0'), 'Episode_Reward/action_rate': tensor(-0.0016, device='cuda:0'), 'Episode_Reward/joint_vel': tensor(-0.0024, device='cuda:0'), 'Curriculum/action_rate': -0.0001, 'Curriculum/joint_vel': -0.0001, 'Metrics/object_pose/position_error': 0.25700417160987854, 'Metrics/object_pose/orientation_error': 3.1088671684265137, 'Episode_Termination/time_out': 1.0, 'Episode_Termination/object_dropping': 0.0}}
        # pretty print info every 200 steps

        if 'log' in info:
            log_info = info['log']
        else:
            log_info = {}

        if step_idx % 200 == 0:
            print(f"[Step {step_idx}] Info:")
            for key, value in log_info.items():
                print(f"  {key}: {value}")

            print(f"\n[Step {step_idx}] Reward breakdown:")
            for env_i in range(self.num_envs):
                terms = self.reward_manager.get_active_iterable_terms(env_i)
                total = sum(val[0] for _, val in terms)
                print(f"  Env {env_i:02d} | total = {total:+.3f}")
                for name, val in terms:
                    print(f"     {name:<20}: {val[0]:+.4f}")
            print("------------------------------------------------------")


        rewards_env0 = self.reward_manager.get_active_iterable_terms(0)
        log_info_env0 = {name: val[0] for name, val in rewards_env0}

        self.rerun_logger.log_data({
            'step': step_idx,
            'rewards': {
                key.replace("Episode_Reward/", ""): value.item() if hasattr(value, 'item') else value
                for key, value in log_info.items() if key.startswith("Episode_Reward/")
            },
            'rewards_env0': log_info_env0
        })

        # obs is obs {'policy': tensor([[[[ 0.1502,  0.1567,  0.0595],
        #   [ 0.1463,  0.1528,  0.0595],
        #   [ 0.1463,  0.1528,  0.0556],

        
        # # Předpokládáme, že ve step() máš proměnnou obs
        # rgb_tensor = obs["policy"][0]  # vezmeme env 0 → shape (H, W, 3)

        # # Převod z 0–1 → 0–255 a na numpy uint8
        # rgb_image = (rgb_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # # Zalogi do Rerun (např. pod jménem "env0/rgb")
        # self.rerun_logger.log_image(rgb_image)

        head_camera_tensor = obs["policy"]["head_camera"][0]  # vezmeme env 0 → shape (H, W, 3)
        head_camera_image = (head_camera_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        self.rerun_logger.log_image("head_camera", head_camera_image)

        
        # top_camera_tensor = obs["policy"]["top_camera"][0]  # vezmeme env 0 → shape (H, W, 3)
        # top_camera_image = (top_camera_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        # self.rerun_logger.log_image2("top_camera", top_camera_image)

        # Log joint positions

        #print("current_right_arm_action:", current_right_arm_action)

        all_robot_joints = self.episode_action_provider.get_sim_robot_joints_data()

        robot_right_arm_joints = {name: all_robot_joints[name] for i, name in enumerate(self.episode_action_provider.right_arm_joint)} 
        robot_right_hand_joints = {name: all_robot_joints[name] for i, name in enumerate(self.episode_action_provider.right_hand_joint)}

        self.rerun_logger.log_right_arm("robot", robot_right_arm_joints)
        self.rerun_logger.log_right_hand("robot", robot_right_hand_joints)

        #self.rerun_logger.log_joints("arm", joint_positions_dict)

        right_arm_joints, right_hand_joints = self.episode_action_provider.get_action_joints(step_idx)

        self.rerun_logger.log_right_arm("target", right_arm_joints)
        self.rerun_logger.log_right_hand("target", right_hand_joints)



        return obs, rew, terminated, truncated, info