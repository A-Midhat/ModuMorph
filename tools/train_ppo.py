# tools/train_ppo.py

import argparse
import os
import sys
import numpy as np # Add numpy import
import torch

# --- Imports for Robosuite padding calculation ---
import robosuite
from robosuite.controllers import load_controller_config
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper # Needed for metadata structure
# --- End Robosuite imports ---

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg, dump_cfg, get_list_cfg # Import get_list_cfg
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu

# --- Padding Calculation Function (Moved Here) ---
def calculate_max_flat_obs_dim(config):
    """Calculates MAX_FLAT_OBS_DIM and updates config."""
    max_dim_needed = 0
    print("[Config] Calculating Robosuite MAX_FLAT_OBS_DIM...")

    morphologies = config.ROBOSUITE.get('TRAINING_MORPHOLOGIES', [])
    env_names = config.ROBOSUITE.get('ENV_NAMES', [])
    controllers_cfg = config.ROBOSUITE.get('CONTROLLERS', []) # This is list of lists/strings

    if not morphologies or not env_names or not controllers_cfg or \
       len(morphologies) != len(env_names) or len(morphologies) != len(controllers_cfg):
        print("Warning: TRAINING_MORPHOLOGIES/ENV_NAMES/CONTROLLERS missing/mismatched. Cannot calculate MAX_FLAT_OBS_DIM.")
        config.MODEL.MAX_FLAT_OBS_DIM = config.MODEL.get('MAX_FLAT_OBS_DIM', 100) # Use existing or default
        return

    common_env_args = { # Minimal args for inspection
        "has_renderer": False, "has_offscreen_renderer": False,
        "ignore_done": True, "use_camera_obs": False,
        "control_freq": 20, "horizon": 10
    }

    for i, morph_spec in enumerate(morphologies):
        env_name = env_names[i]
        robots = get_list_cfg(morph_spec)
        controllers_spec = controllers_cfg[i]
        controllers_list = get_list_cfg(controllers_spec) # Ensure list for multi-arm

        # Ensure controller list matches robot list for this morph
        if len(controllers_list) == 1 and len(robots) > 1:
            controllers_list = controllers_list * len(robots)
        elif len(controllers_list) != len(robots):
            print(f"Warning: Controller/Robot mismatch for morph {i}. Skipping dim calc.")
            continue

        controller_configs = []
        valid_controllers = True
        for c_name in controllers_list:
            try: controller_configs.append(load_controller_config(default_controller=c_name))
            except: valid_controllers = False; break
        if not valid_controllers:
            print(f"Warning: Invalid controller name in morph {i}. Skipping dim calc.")
            continue

        temp_env_instance = None
        try:
            temp_env_instance = robosuite.make(
                env_name=env_name, robots=robots,
                controller_configs=controller_configs, **common_env_args
            )
            # --- Determine keys to flatten for *this* instance ---
            num_r = len(temp_env_instance.robots)
            keys_instance = [f"robot{r}_proprio-state" for r in range(num_r)] + ["object-state"]
            # --- Calculate flat dim for *this* instance ---
            obs_spec = temp_env_instance.observation_spec()
            current_flat_dim = sum(
                np.prod(obs_spec[k].shape)
                for k in keys_instance if k in obs_spec and hasattr(obs_spec[k], 'shape')
            )
            max_dim_needed = max(max_dim_needed, current_flat_dim)
            # print(f"  Morph {i} flat dim: {current_flat_dim}") # Optional debug

        except Exception as e:
            print(f"Warning: Failed to instantiate morph {i} for flat dim calc: {e}")
        finally:
            if temp_env_instance: 
                try: 
                    temp_env_instance.close()
                except: 
                    pass

    if max_dim_needed == 0:
         print("Warning: Could not determine max flat dim. Using default 100.")
         config.MODEL.MAX_FLAT_OBS_DIM = 100
    else:
         config.MODEL.MAX_FLAT_OBS_DIM = max_dim_needed
    print(f"[Config] Set MAX_FLAT_OBS_DIM = {config.MODEL.MAX_FLAT_OBS_DIM}")

# --- Existing functions ---
def calculate_unimal_modular_padding(config):
    # Renamed original function slightly
    if config.ENV_NAME == "Unimal-v0":
        num_joints, num_limbs = [], []
        walkers = config.ENV.get('WALKERS', []) # Use get for safety
        if not walkers:
            print("Warning: No Unimal walkers found for padding calculation.")
            return # Use defaults if no walkers

        walker_dir = config.ENV.get('WALKER_DIR')
        if not walker_dir or not os.path.isdir(walker_dir):
             print(f"Warning: Invalid WALKER_DIR '{walker_dir}'. Cannot calculate Unimal padding.")
             return

        for agent in walkers:
            metadata_path = os.path.join(walker_dir, "metadata", f"{agent}.json")
            if os.path.exists(metadata_path):
                metadata = fu.load_json(metadata_path)
                num_joints.append(metadata.get("dof", 0))
                # Assuming num_limbs in metadata is correct (doesn't include torso)
                num_limbs.append(metadata.get("num_limbs", 0) + 1) # +1 for torso
            else:
                print(f"Warning: Metadata not found for Unimal agent: {agent}")

        if num_joints and num_limbs:
            padding_needed = 1
            config.MODEL.MAX_JOINTS = max(num_joints) + padding_needed
            config.MODEL.MAX_LIMBS = max(num_limbs) + padding_needed
            # print(f"[Config] Calculated Unimal padding: MAX_JOINTS={config.MODEL.MAX_JOINTS}, MAX_LIMBS={config.MODEL.MAX_LIMBS}")
        else:
             print("Warning: Could not calculate Unimal padding due to missing data.")


    elif config.ENV_NAME == 'Modular-v0':
        # Keep hardcoded logic based on eval agent dir if needed, or implement dynamic calc
        walker_dir = config.ENV.get('WALKER_DIR', "")
        if not walker_dir: print("Warning: WALKER_DIR not set for Modular padding."); return

        if 'hopper' in walker_dir: config.MODEL.MAX_LIMBS = 5; config.MODEL.MAX_JOINTS = 5
        elif 'walker' in walker_dir: config.MODEL.MAX_LIMBS = 7; config.MODEL.MAX_JOINTS = 7
        elif 'humanoid' in walker_dir: config.MODEL.MAX_LIMBS = 9; config.MODEL.MAX_JOINTS = 9
        # Add more cases or a dynamic calculation if necessary
        else: config.MODEL.MAX_LIMBS = 9; config.MODEL.MAX_JOINTS = 9 # Default fallback
        # print(f"[Config] Set Modular padding: MAX_LIMBS={config.MODEL.MAX_LIMBS}, MAX_JOINTS={config.MODEL.MAX_JOINTS}")


def calculate_max_iters(config): # Pass config
    # Iter here refers to 1 cycle of experience collection and policy update.
    try:
        config.PPO.MAX_ITERS = (
            int(config.PPO.MAX_STATE_ACTION_PAIRS) // config.PPO.TIMESTEPS // config.PPO.NUM_ENVS
        )
        config.PPO.EARLY_EXIT_MAX_ITERS = (
            int(config.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // config.PPO.TIMESTEPS // config.PPO.NUM_ENVS
        )
        # print("Max Iterations: ", config.PPO.MAX_ITERS)
    except ZeroDivisionError:
        print("Warning: Could not calculate MAX_ITERS due to zero division (check TIMESTEPS/NUM_ENVS). Setting to -1.")
        config.PPO.MAX_ITERS = -1
        config.PPO.EARLY_EXIT_MAX_ITERS = -1


def maybe_infer_morphologies(config): # Pass config, renamed function
    """Populates TRAINING_MORPHOLOGIES, ENV_NAMES, CONTROLLERS if empty."""
    if config.ENV_NAME == "Robosuite-v0":
        # Only populate if the main list is empty
        if not config.ROBOSUITE.get('TRAINING_MORPHOLOGIES'):
            # Default or infer based on ROBOTS if that old key exists
            default_robots = config.ROBOSUITE.get('ROBOTS', ["Panda"]) # Fallback
            config.ROBOSUITE.TRAINING_MORPHOLOGIES = get_list_cfg(default_robots)
            print(f"Inferred TRAINING_MORPHOLOGIES: {config.ROBOSUITE.TRAINING_MORPHOLOGIES}")

        # Auto-populate ENV_NAMES and CONTROLLERS if empty or mismatched
        num_morphs = len(config.ROBOSUITE.TRAINING_MORPHOLOGIES)
        if not config.ROBOSUITE.get('ENV_NAMES') or len(config.ROBOSUITE.ENV_NAMES) != num_morphs:
             # Infer based on morphology structure (simple heuristic)
             config.ROBOSUITE.ENV_NAMES = []
             for morph in config.ROBOSUITE.TRAINING_MORPHOLOGIES:
                  if isinstance(morph, (list, tuple)): # Assume TwoArm task
                      # Use a default TwoArm task name or require it to be set
                      default_two_arm = "TwoArmLift" # Or fetch from config if defined
                      config.ROBOSUITE.ENV_NAMES.append(default_two_arm)
                  else: # Assume single arm task
                      # Use a default single arm task name or require it
                      default_single_arm = "Lift" # Or fetch from config
                      config.ROBOSUITE.ENV_NAMES.append(default_single_arm)
             print(f"Inferred ENV_NAMES: {config.ROBOSUITE.ENV_NAMES}")

        if not config.ROBOSUITE.get('CONTROLLERS') or len(config.ROBOSUITE.CONTROLLERS) != num_morphs:
            # Default to JOINT_VELOCITY, create lists for TwoArm if needed
            default_controller = "JOINT_VELOCITY"
            config.ROBOSUITE.CONTROLLERS = []
            for morph in config.ROBOSUITE.TRAINING_MORPHOLOGIES:
                 if isinstance(morph, (list, tuple)):
                      # Use OSC_POSE for TwoArm by default? Or JOINT_VELOCITY? Let's stick to simpler one first.
                      config.ROBOSUITE.CONTROLLERS.append([default_controller] * len(morph))
                 else:
                      config.ROBOSUITE.CONTROLLERS.append(default_controller)
            print(f"Inferred CONTROLLERS: {config.ROBOSUITE.CONTROLLERS}")

    elif config.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        # Existing logic for walkers
        if not config.ENV.get('WALKERS'): # Use get for safety
            walker_dir = config.ENV.get('WALKER_DIR')
            if walker_dir and os.path.isdir(os.path.join(walker_dir, "xml")):
                config.ENV.WALKERS = [
                    xml_file.split(".")[0]
                    for xml_file in os.listdir(os.path.join(walker_dir, "xml"))
                    if xml_file.endswith('.xml') # Be more specific
                ]
                # print(f"Inferred Walkers: {config.ENV.WALKERS}")
            # else: print("Warning: Could not infer walkers.")

        # Modular env registration (keep if needed)
        # if config.ENV_NAME == 'Modular-v0':
        #     register_modular_envs(config) # Pass config if needed

def set_cfg_options(config): # Pass config
    """Sets calculated/inferred config options."""
    calculate_max_iters(config)
    # Determine morphologies first (Unimal/Modular walkers or Robosuite TRAINING_MORPHOLOGIES)
    maybe_infer_morphologies(config)
    # Calculate padding based on the environment type and inferred morphologies
    if config.ENV_NAME == "Robosuite-v0":
        calculate_robosuite_padding(config)
    elif config.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        calculate_unimal_modular_padding(config)
    # else: no padding calculation for other env types


# --- Main Execution ---
def main():
    args = parse_args()
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    set_cfg_options(cfg) 

    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    dump_cfg() # Save the potentially modified config

    # --- Rest of training setup ---
    su.set_seed(cfg.RNG_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.set_num_threads(1)

    # Initialize PPO Trainer (it will use the updated cfg)
    PPOTrainer = PPO()
    PPOTrainer.train()

    # hparams = get_hparams() # Keep if needed for logging
    PPOTrainer.save_rewards() # Removed hparams argument for simplicity
    PPOTrainer.save_model(-1)
    # cleanup_tensorboard() # Keep if needed

if __name__ == "__main__":
    # Define or import parse_args, PPO, dump_cfg, etc.
    main()