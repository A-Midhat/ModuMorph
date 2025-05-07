import argparse
import os
import sys
import numpy as np
import torch

import robosuite
from robosuite.controllers import load_controller_config

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg, dump_cfg, get_list_cfg

from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu

# --- Padding Calculation Functions ---

def calculate_max_flat_obs_dim():
    """Calculates MAX_FLAT_OBS_DIM and updates global cfg."""
    max_dim_needed = 0
    print("[Config] Calculating Robosuite MAX_FLAT_OBS_DIM...")

    morphologies = cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES', [])
    env_names = cfg.ROBOSUITE.get('ENV_NAMES', [])
    controllers_cfg = cfg.ROBOSUITE.get('CONTROLLERS', [])

    # Check lists using global cfg
    if not morphologies or not env_names or not controllers_cfg or \
       len(morphologies) != len(env_names) or len(morphologies) != len(controllers_cfg):
        print("Warning: TRAINING_MORPHOLOGIES/ENV_NAMES/CONTROLLERS missing/mismatched. Cannot calculate MAX_FLAT_OBS_DIM.")
        # Modify global cfg directly
        cfg.MODEL.MAX_FLAT_OBS_DIM = cfg.MODEL.get('MAX_FLAT_OBS_DIM', 100)
        return

    common_env_args = {
        "has_renderer": False, "has_offscreen_renderer": False,
        "ignore_done": True, "use_camera_obs": False,
        "control_freq": 20, "horizon": 10
    }

    for i, morph_spec in enumerate(morphologies):
        env_name = env_names[i]
        robots = get_list_cfg(morph_spec)
        controllers_spec = controllers_cfg[i]
        controllers_list = get_list_cfg(controllers_spec)

        if len(controllers_list) == 1 and len(robots) > 1: controllers_list *= len(robots)
        elif len(controllers_list) != len(robots): continue # Skip

        controller_configs = []
        valid_controllers = True
        for c_name in controllers_list:
            try: controller_configs.append(load_controller_config(default_controller=c_name))
            except: valid_controllers = False; break
        if not valid_controllers: continue # Skip

        temp_env_instance = None
        try:
            temp_env_instance = robosuite.make(env_name=env_name, robots=robots, controller_configs=controller_configs, **common_env_args)
            num_r = len(temp_env_instance.robots)
            keys_instance = [f"robot{r}_proprio-state" for r in range(num_r)] + ["object-state"]
            obs_spec = temp_env_instance.observation_spec()
            current_flat_dim = sum( np.prod(obs_spec[k].shape) for k in keys_instance if k in obs_spec and hasattr(obs_spec[k], 'shape') )
            max_dim_needed = max(max_dim_needed, current_flat_dim)
        except Exception as e: print(f"Warn flat dim calc morph {i}: {e}")
        finally:
            if temp_env_instance: 
                try: 
                    temp_env_instance.close()
                except:
                    pass

    if max_dim_needed == 0:
         print("Warning: Could not determine max flat dim. Using default 100.")
         cfg.MODEL.MAX_FLAT_OBS_DIM = 100 # Modify global cfg
    else:
         cfg.MODEL.MAX_FLAT_OBS_DIM = max_dim_needed # Modify global cfg
    print(f"[Config] Set MAX_FLAT_OBS_DIM = {cfg.MODEL.MAX_FLAT_OBS_DIM}")


def calculate_robosuite_padding():
    """Calculates Robosuite MAX_LIMBS/JOINTS and updates global cfg."""
    max_limbs_needed = 0; max_joints_needed = 0
    print("[Config] Calculating Robosuite MAX_LIMBS/JOINTS padding...")
    # Access global cfg directly
    morphologies=cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES',[]); env_names=cfg.ROBOSUITE.get('ENV_NAMES',[])
    if not morphologies or not env_names or len(morphologies)!=len(env_names): print("Warning: Morph/Env list mismatch."); return
    try: controller_config=load_controller_config(default_controller="JOINT_VELOCITY")
    except: print("Warn: Cannot load default controller for padding calc."); return
    common_args={"has_renderer":False,"has_offscreen_renderer":False,"ignore_done":True,"use_camera_obs":False,"control_freq":20,"horizon":10}

    for i,morph in enumerate(morphologies):
        env_name=env_names[i]; robots=get_list_cfg(morph); temp_env=None
        try:
            temp_env=robosuite.make(env_name=env_name,robots=robots,controller_configs=[controller_config]*len(robots),**common_args)
            limbs=0; joints=0
            for r in temp_env.robots:
                n_arm=r.dof-r.gripper.dof; n_grip=cfg.ROBOSUITE.get('GRIPPER_DIM',r.gripper.dof); limbs+=1+n_arm+1; joints+=n_arm+n_grip # Node/joint counting logic
            max_limbs_needed=max(max_limbs_needed,limbs); max_joints_needed=max(max_joints_needed,joints)
        except Exception as e: print(f"Warn pad calc morph {i}: {e}")
        finally:
            if temp_env: 
                try: 
                    temp_env.close()
                except: 
                    pass

    pad=1
    # Modify global cfg directly
    if max_limbs_needed == 0: cfg.MODEL.MAX_LIMBS = cfg.MODEL.get('MAX_LIMBS', 11) # Fallback
    else: cfg.MODEL.MAX_LIMBS = max_limbs_needed + pad
    if max_joints_needed == 0: cfg.MODEL.MAX_JOINTS = cfg.MODEL.get('MAX_JOINTS', 10) # Fallback
    else: cfg.MODEL.MAX_JOINTS = max_joints_needed + pad
    print(f"[Config] Set Robosuite MAX_LIMBS={cfg.MODEL.MAX_LIMBS}, MAX_JOINTS={cfg.MODEL.MAX_JOINTS}")


def calculate_unimal_modular_padding():
    """Calculates Unimal/Modular MAX_LIMBS/JOINTS and updates global cfg."""
    # Access global cfg directly
    if cfg.ENV_NAME == "Unimal-v0":
        # ... (rest of the Unimal padding logic using global cfg) ...
        num_joints, num_limbs = [], []
        walkers = cfg.ENV.get('WALKERS', [])
        if not walkers: return
        walker_dir = cfg.ENV.get('WALKER_DIR')
        if not walker_dir or not os.path.isdir(walker_dir): return
        for agent in walkers:
            metadata_path = os.path.join(walker_dir, "metadata", f"{agent}.json")
            if os.path.exists(metadata_path):
                metadata = fu.load_json(metadata_path)
                num_joints.append(metadata.get("dof", 0))
                num_limbs.append(metadata.get("num_limbs", 0) + 1)
        if num_joints and num_limbs:
            padding = 1; cfg.MODEL.MAX_JOINTS = max(num_joints)+padding; cfg.MODEL.MAX_LIMBS = max(num_limbs)+padding

    elif cfg.ENV_NAME == 'Modular-v0':
        # ... (rest of the Modular padding logic using global cfg) ...
        walker_dir = cfg.ENV.get('WALKER_DIR', "")
        if 'hopper' in walker_dir: cfg.MODEL.MAX_LIMBS=5; cfg.MODEL.MAX_JOINTS=5
        elif 'walker' in walker_dir: cfg.MODEL.MAX_LIMBS=7; cfg.MODEL.MAX_JOINTS=7
        elif 'humanoid' in walker_dir: cfg.MODEL.MAX_LIMBS=9; cfg.MODEL.MAX_JOINTS=9
        else: cfg.MODEL.MAX_LIMBS=9; cfg.MODEL.MAX_JOINTS=9


def calculate_max_iters():
    """Calculates MAX_ITERS/EARLY_EXIT and updates global cfg."""
    # Access global cfg directly
    try:
        cfg.PPO.MAX_ITERS = int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
        cfg.PPO.EARLY_EXIT_MAX_ITERS = int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    except ZeroDivisionError:
        print("Warning: Division by zero in MAX_ITERS calc.")
        cfg.PPO.MAX_ITERS = -1; cfg.PPO.EARLY_EXIT_MAX_ITERS = -1


def maybe_infer_morphologies():
    """Populates TRAINING_MORPHOLOGIES, ENV_NAMES, CONTROLLERS if empty, using global cfg."""
    # Access global cfg directly
    if cfg.ENV_NAME == "Robosuite-v0":
        # ... (rest of the Robosuite inference logic using global cfg) ...
        if not cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES'): cfg.ROBOSUITE.TRAINING_MORPHOLOGIES=get_list_cfg(cfg.ROBOSUITE.get('ROBOTS',["Panda"])); print(f"Infer Morph: {cfg.ROBOSUITE.TRAINING_MORPHOLOGIES}")
        num_morphs=len(cfg.ROBOSUITE.TRAINING_MORPHOLOGIES)
        if not cfg.ROBOSUITE.get('ENV_NAMES') or len(cfg.ROBOSUITE.ENV_NAMES)!=num_morphs:
            cfg.ROBOSUITE.ENV_NAMES=[]; default_single="Lift"; default_two="TwoArmLift"
            for m in cfg.ROBOSUITE.TRAINING_MORPHOLOGIES: cfg.ROBOSUITE.ENV_NAMES.append(default_two if isinstance(m,(list,tuple)) else default_single)
            print(f"Infer EnvNames: {cfg.ROBOSUITE.ENV_NAMES}")
        if not cfg.ROBOSUITE.get('CONTROLLERS') or len(cfg.ROBOSUITE.CONTROLLERS)!=num_morphs:
            default_controller="JOINT_VELOCITY"; cfg.ROBOSUITE.CONTROLLERS=[]
            for m in cfg.ROBOSUITE.TRAINING_MORPHOLOGIES: cfg.ROBOSUITE.CONTROLLERS.append([default_controller]*len(get_list_cfg(m)) if isinstance(m,(list,tuple)) else default_controller)
            print(f"Infer Controllers: {cfg.ROBOSUITE.CONTROLLERS}")

    elif cfg.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        # ... (rest of the Unimal/Modular inference logic using global cfg) ...
        if not cfg.ENV.get('WALKERS'):
            walker_dir=cfg.ENV.get('WALKER_DIR')
            if walker_dir and os.path.isdir(os.path.join(walker_dir,"xml")):
                cfg.ENV.WALKERS=[f.split('.')[0] for f in os.listdir(os.path.join(walker_dir,"xml")) if f.endswith('.xml')]


def set_cfg_options():
    """Sets calculated/inferred config options using global cfg."""
    calculate_max_iters()
    maybe_infer_morphologies() # Infer lists first

    # Calculate padding based on the environment type
    if cfg.ENV_NAME == "Robosuite-v0":
        calculate_robosuite_padding() # Calculates MAX_LIMBS/JOINTS
        calculate_max_flat_obs_dim() # Calculates MAX_FLAT_OBS_DIM
    elif cfg.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        calculate_unimal_modular_padding()


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train RL agent")
    parser.add_argument("--cfg", dest="cfg_file", help="Config file", required=True, type=str)
    # Corrected help text typo
    parser.add_argument(
        "opts",
        help="See metamorph/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    """Sets up and runs the PPO training process."""
    su.set_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.set_num_threads(1)

    PPOTrainer = PPO()
    PPOTrainer.train()

    PPOTrainer.save_rewards()
    PPOTrainer.save_model(-1)


# --- Main Execution ---
def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    # Optionally re-add the --no_context_in_state logic here if needed:
    # if args.no_context_in_state:
    #     obs_type = [...] # Define the shorter list
    #     ob_opts = ["MODEL.PROPRIOCEPTIVE_OBS_TYPES", obs_type]
    #     cfg.merge_from_list(ob_opts)

    # *** Call setup function which uses global cfg ***
    set_cfg_options()

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    # Save the final config *after* calculations
    dump_cfg()

    print(f"Environment: {cfg.ENV_NAME}")
    if cfg.ENV_NAME == "Robosuite-v0":
        print(f"Morphologies: {cfg.ROBOSUITE.TRAINING_MORPHOLOGIES}")
    elif cfg.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        print(f"Walkers: {cfg.ENV.WALKERS}")

    ppo_train() 

    print(f"Training run completed: {cfg.OUT_DIR}")


if __name__ == "__main__":
    main()