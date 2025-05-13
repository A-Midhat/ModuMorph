import argparse
import os
import sys

import torch

from gym.envs.registration import register

import robosuite 
from robosuite.controllers import load_controller_config

from metamorph.algos.ppo.ppo import PPO
from metamorph.config import cfg
from metamorph.config import dump_cfg
from metamorph.config import get_list_cfg

from metamorph.utils import robosuite_utils as ru 
from metamorph.utils import file as fu
from metamorph.utils import sample as su
from metamorph.utils import sweep as swu


def set_cfg_options():
    calculate_max_iters()
    maybe_infer_morphs()
    if cfg.ENV_NAME == "Robosuite-v0":
        calculate_max_limbs_joints_robosuite() 
    else:
        # modular/unimal  
        calculate_max_limbs_joints()
    
def calculate_max_limbs_joints_robosuite():
    """
    Calculate max limbs and joints for robosuite, and updates cfg
    """
   
    morphs = cfg.ROBOSUITE.TRAINING_MORPHOLOGIES # can't be empty because of the maybn_infer_morphs()
    env_names = cfg.ROBOSUITE.ENV_NAMES
    controllers = cfg.ROBOSUITE.CONTROLLERS 
    # VERY BASIC SOLUTION TO FIND THE MAX LIMBS AND JOINTS
    # we will need to instantiate simple envs for this 
    # Could use sim but no need since this would take 1s to calc 
    common_args = {
        "has_renderer": False, "has_offscreen_renderer": False, 
        "ignore_done": True, "use_camera_obs": False, 
        "control_freq": 20, "horizon": 10
        }
    max_l, max_j = ru.max_limbs_joints(morphs,  env_names, controllers, common_args)
    # HARD CODED: 
    # limbs calc (base + joints + gripper) i.e. Panda becomes 9 instead of just 7 
    # robosuite actions depends on the controller , i.e. osc_pose 6 + gripper dim, joint_velocisity 7 + gripper
    # so for TwoArmEnv we have 18 and the real action dim is 16 so we substract 2
    # for SingleArmEnv we have 9 and the real action dim is 8 so we substract 1
    # else we would use node cetnric and it already has act masking built with in (robosuite_wrappers.py)
    
    if len(morphs) == 1 and isinstance(morphs[0], (list, tuple)):
        # no need for padding for SR
        # also we remove the base from the strucure 
        # TODO: Add this back for mobile bases
        cfg.MODEL.MAX_LIMBS = max_l - 2
        cfg.MODEL.MAX_JOINTS = max_j  - 2
    elif len(morphs) == 1 and not isinstance(morphs[0], (list, tuple)):
        cfg.MODEL.MAX_LIMBS = max_l - 1
        cfg.MODEL.MAX_JOINTS = max_j - 1
    else: 
        cfg.MODEL.MAX_LIMBS = max_l + 1 
        cfg.MODEL.MAX_JOINTS = max_j + 1 

    print(f"[Config] Set MAX_LIMBS={cfg.MODEL.MAX_LIMBS}, MAX_JOINTS={cfg.MODEL.MAX_JOINTS}")

def calculate_max_limbs_joints():
    if cfg.ENV_NAME == "Unimal-v0":

        num_joints, num_limbs = [], []

        metadata_paths = []
        for agent in cfg.ENV.WALKERS:
            metadata_paths.append(os.path.join(
                cfg.ENV.WALKER_DIR, "metadata", "{}.json".format(agent)
            ))

        for metadata_path in metadata_paths:
            metadata = fu.load_json(metadata_path)
            num_joints.append(metadata["dof"])
            num_limbs.append(metadata["num_limbs"] + 1)

        # Add extra 1 for max_joints; needed for adding edge padding
        cfg.MODEL.MAX_JOINTS = max(num_joints) + 1
        cfg.MODEL.MAX_LIMBS = max(num_limbs) + 1
        cfg.MODEL.MAX_JOINTS = 16
        cfg.MODEL.MAX_LIMBS = 12
        print (cfg.MODEL.MAX_JOINTS, cfg.MODEL.MAX_LIMBS)
    
    elif cfg.ENV_NAME == 'Modular-v0':
        # hardcode this part
        if 'hopper' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 5
            cfg.MODEL.MAX_JOINTS = 5
        if 'walker' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 7
            cfg.MODEL.MAX_JOINTS = 7
        if 'humanoid' in cfg.ENV.WALKER_DIR:
            cfg.MODEL.MAX_LIMBS = 9
            cfg.MODEL.MAX_JOINTS = 9
        if 'all' in cfg.ENV.WALKER_DIR:
            if cfg.MODEL.MLP.CONSISTENT_PADDING:
                cfg.MODEL.MAX_LIMBS = 19
                cfg.MODEL.MAX_JOINTS = 19
            else:
                cfg.MODEL.MAX_LIMBS = 9
                cfg.MODEL.MAX_JOINTS = 9

def calculate_max_iters():
    # Iter here refers to 1 cycle of experience collection and policy update.
    cfg.PPO.MAX_ITERS = (
        int(cfg.PPO.MAX_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )
    cfg.PPO.EARLY_EXIT_MAX_ITERS = (
        int(cfg.PPO.EARLY_EXIT_STATE_ACTION_PAIRS) // cfg.PPO.TIMESTEPS // cfg.PPO.NUM_ENVS
    )


def maybe_infer_morphs():
    """"
    Populates TRAINING_MORPHOLOGIES/ENV_NAMES/CONTROLLERS for robosuite,
    or WALKERS for unimal/modular
    """
    if cfg.ENV_NAME == "Robosuite-v0":
        # TODO: add logic to know if im using SR or MR
        if cfg.ROBOSUITE.get("TASK_TYPE", "SR"): # SINGLE ROBOT
            if not cfg.ROBOSUITE.get("TRAINING_MORPHOLOGIES"):
                cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = ["Panda"] # default 
            else:
                return # already set
        else: # MR 
            if not cfg.ROBOSUITE.get("TRAINING_MORPHOLOGIES"):
                cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = ["Panda", "Jaco", "Kinova3"]  
            else:
                return

        if not cfg.ROBOSUITE.get("ENV_NAMES") or len(cfg.ROBOSUITE.ENV_NAMES) != len(cfg.ROBOSUITE.TRAINING_MORPHOLOGIES):
            # reset for the sake of simplicity
            cfg.ROBOSUITE.ENV_NAMES = []
            singleArm_env = "Lift"
            twoArm_env = "TwoArmLift"

            for morph in cfg.ROBOSUITE.TRAINING_MORPHOLOGIES:
                cfg.ROBOSUITE.ENV_NAMES.append(twoArm_env if isinstance(morph, (list, tuple)) else singleArm_env) # checks for [["R1", "R2"]] or ["R1"]
        
        if not cfg.ROBOSUITE.get("CONTROLLERS"):
            default_controller = "JOINT_VELOCITY"
            cfg.ROBOSUITE.CONTROLLERS = [] 
            for controller in cfg.ROBOSUITE.TRAINING_MORPHOLOGIES:
                controller_ = get_list_cfg(controller)
                cfg.ROBOSUITE.CONTROLLERS.append([default_controller]*len(controller_) if isinstance(controller_, (list, tuple)) else default_controller)

        print(f"Infer Robosuite morph(s):\t{cfg.ROBOSUITE.TRAINING_MORPHOLOGIES}\nTask(s):\t{cfg.ROBOSUITE.ENV_NAMES}\nController(s):\t{cfg.ROBOSUITE.CONTROLLERS}")

    elif cfg.ENV_NAME in ["Unimal-v0", "Modular-v0"]:
        # only infer if did not specify any
        if not cfg.ENV.get("WALKERS"):
            walker_dir = cfg.ENV.get("WALKER_DIR", "")
            if walker_dir and os.path.isdir(os.path.join(walker_dir, "xml")):
                cfg.ENV.WALKERS = [
                    f[:-4] for f in os.listdir(os.path.join(walker_dir, "xml"))
                    if f.endswith(".xml")
                ]
        # for Modular, also register gym ids
        if cfg.ENV_NAME=="Modular-v0" and cfg.ENV.WALKERS:
            register_modular_envs()
        else:
            return # already set

    else:
        # no inference for other ENV_NAMEs
        return

def register_modular_envs():
    """register the MuJoCo envs with Gym and return the per-agent observation size and max action value (for modular policy training)"""
    # register each env
    for agent in cfg.ENV.WALKERS:
        # create a copy of modular environment for custom xml model
        # if not os.path.exists(os.path.join(ENV_DIR, "{}.py".format(env_name))):
        #     # create a duplicate of gym environment file for each env (necessary for avoiding bug in gym)
        #     copyfile(
        #         BASE_MODULAR_ENV_PATH, "{}.py".format(os.path.join(ENV_DIR, env_name))
        #     )
        xml = os.path.join(cfg.ENV.WALKER_DIR, 'xml', agent + '.xml')
        params = {"xml": os.path.abspath(xml)}
        try:
            register(
                id=f"{agent}-v0",
                max_episode_steps=1000,
                entry_point=f"modular.{agent}:make_env",
                kwargs=params,
            )
        except:
            continue
    #     env = wrappers.IdentityWrapper(gym.make(f"{agent}-v0"))
    #     # the following is the same for each env
    #     agent_obs_size = env.agent_obs_size
    #     max_action = env.max_action
    # return agent_obs_size, max_action


def get_hparams():
    hparam_path = os.path.join(cfg.OUT_DIR, "hparam.json")
    # For local sweep return
    if not os.path.exists(hparam_path):
        return {}

    hparams = {}
    varying_args = fu.load_json(hparam_path)
    flatten_cfg = swu.flatten(cfg)

    for k in varying_args:
        hparams[k] = flatten_cfg[k]

    return hparams


def cleanup_tensorboard():
    tb_dir = os.path.join(cfg.OUT_DIR, "tensorboard")

    # Assume there is only one sub_dir and break when it's found
    for content in os.listdir(tb_dir):
        content = os.path.join(tb_dir, content)
        if os.path.isdir(content):
            break

    # Return if no dir found
    if not os.path.isdir(content):
        return

    # Move all the event files from sub_dir to tb_idr
    for event_file in os.listdir(content):
        src = os.path.join(content, event_file)
        dst = os.path.join(tb_dir, event_file)
        fu.move_file(src, dst)

    # Delete the sub_dir
    os.rmdir(content)


def parse_args():
    """Parses the arguments."""
    parser = argparse.ArgumentParser(description="Train a RL agent")
    parser.add_argument(
        "--cfg", dest="cfg_file", help="Config file", required=True, type=str
    )
    parser.add_argument(
        "--no_context_in_state", action="store_true"
    )
    parser.add_argument(
        "opts",
        help="See morphology/core/config.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def ppo_train():
    su.set_seed(cfg.RNG_SEED)
    # Configure the CUDNN backend
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC

    torch.set_num_threads(1)
    PPOTrainer = PPO()
    PPOTrainer.train()
    hparams = get_hparams()
    PPOTrainer.save_rewards(hparams=hparams)
    PPOTrainer.save_model(-1)
    # if not using wandb 
    if PPOTrainer.logger_backend == "tensorboard":
        cleanup_tensorboard()


def main():
    # Parse cmd line args
    args = parse_args()

    # Load config options
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    # TODO: make robosuite wrappers extracts those obs from  env.sim 
    if args.no_context_in_state:
        obs_type = [
            "body_xpos", "body_xvelp", "body_xvelr", "body_xquat", # limb
            "qpos", "qvel", # joint
        ]
        ob_opts = ["MODEL.PROPRIOCEPTIVE_OBS_TYPES", obs_type]
        cfg.merge_from_list(ob_opts)

    # Set cfg options which are inferred
    set_cfg_options()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)

    # Save the config
    dump_cfg()
    ppo_train()


if __name__ == "__main__":
    main()
