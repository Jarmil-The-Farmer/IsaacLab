import torch
from isaaclab.envs import ManagerBasedRLEnv
from .joint_pos_env_cfg_visual import FrankaCubeLiftEnvCfg
from .rerun_visualizer import RerunLogger

class CustomManagerBasedRLEnv(ManagerBasedRLEnv):
    cfg_cls = FrankaCubeLiftEnvCfg

    rerun_logger: RerunLogger

    def __init__(self, cfg: FrankaCubeLiftEnvCfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.rerun_logger = RerunLogger()


    def step(self, action):
        step_idx = self.common_step_counter

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

        
        # Předpokládáme, že ve step() máš proměnnou obs
        rgb_tensor = obs["policy"][0]  # vezmeme env 0 → shape (H, W, 3)

        # Převod z 0–1 → 0–255 a na numpy uint8
        rgb_image = (rgb_tensor.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

        # Zalogi do Rerun (např. pod jménem "env0/rgb")
        self.rerun_logger.log_image(rgb_image)


        return obs, rew, terminated, truncated, info