import argparse
import json
import os
import pickle
import sys 

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import gym 
from gym import spaces



from metamorph.config import cfg, load_cfg

import metamorph.envs
from metamorph.algos.ppo.envs import (make_env, get_ob_rms,
                                      get_vec_normalize, set_ob_rms)

from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
from metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
from metamorph.envs.vec_env.vec_normalize import VecNormalize
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.algos.ppo.model import ActorCritic, Agent

from metamorph.utils import file as fu
from metamorph.utils import sweep as swu

from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper
import robosuite # For controller config loading
from robosuite.controllers import load_controller_config


def set_cfg_options():
    """Sets inferred config options relevant for evaluation."""
    maybe_infer_agents() # Ensure AGENTS list is populated based on args
    calculate_max_limbs_joints() # Crucial for model init based on test agents

def calculate_max_limbs_joints():
    """
    Calculates the maximum number of limbs and joints required for padding
    based on the agents specified in the config (for evaluation).
    """
    agents = cfg.ENV.get('AGENTS', []) # Use the potentially overridden list
    if not agents:
         print("Warning: cfg.ENV.AGENTS is empty in calculate_max_limbs_joints.")
         if not hasattr(cfg.MODEL, 'MAX_LIMBS') or not cfg.MODEL.get('MAX_LIMBS'): cfg.MODEL.MAX_LIMBS = 8
         if not hasattr(cfg.MODEL, 'MAX_JOINTS') or not cfg.MODEL.get('MAX_JOINTS'): cfg.MODEL.MAX_JOINTS = 8
         print(f"Using fallback/default MAX_LIMBS={cfg.MODEL.MAX_LIMBS}, MAX_JOINTS={cfg.MODEL.MAX_JOINTS}")
         return

    num_joints, num_limbs = [], []
    padding_needed = 1 

    if cfg.ENV_NAME == "Unimal-v0":
        
        agent_dir = cfg.ENV.get('WALKER_DIR', None)
        if not agent_dir: raise ValueError("cfg.ENV.WALKER_DIR not set for Unimal/Modular eval.")
        metadata_paths = [os.path.join(agent_dir, "metadata", f"{agent}.json") for agent in agents]
        for metadata_path in metadata_paths:
            if not os.path.exists(metadata_path): print(f"Warning: Metadata missing {metadata_path}"); continue
            metadata = fu.load_json(metadata_path)
            num_joints.append(metadata.get("dof", 0))
            num_limbs.append(metadata.get("num_limbs", 0) + 1) # +1 for torso/root
        max_limbs_calculated = max(num_limbs) + padding_needed if num_limbs else 12
        max_joints_calculated = max(num_joints) + padding_needed if num_joints else 16
        cfg.MODEL.MAX_JOINTS = max_joints_calculated
        cfg.MODEL.MAX_LIMBS = max_limbs_calculated
        print(f"[Unimal/Modular Eval] Calculated MAX_JOINTS={cfg.MODEL.MAX_JOINTS}, MAX_LIMBS={cfg.MODEL.MAX_LIMBS}")

    elif cfg.ENV_NAME == 'Modular-v0':
        # Keep hardcoded logic based on eval agent dir
        agent_dir = cfg.ENV.get('WALKER_DIR', "")
        if 'hopper' in agent_dir: cfg.MODEL.MAX_LIMBS = 5; cfg.MODEL.MAX_JOINTS = 5
        elif 'walker' in agent_dir: cfg.MODEL.MAX_LIMBS = 7; cfg.MODEL.MAX_JOINTS = 7
        elif 'humanoid' in agent_dir: cfg.MODEL.MAX_LIMBS = 9; cfg.MODEL.MAX_JOINTS = 9
        elif 'all' in agent_dir: cfg.MODEL.MAX_LIMBS = 19; cfg.MODEL.MAX_JOINTS = 19 # Assume consistent padding for 'all'
        else: cfg.MODEL.MAX_LIMBS = 9; cfg.MODEL.MAX_JOINTS = 9 # Default
        print(f"[Modular Eval] Set MAX_LIMBS={cfg.MODEL.MAX_LIMBS}, MAX_JOINTS={cfg.MODEL.MAX_JOINTS}")

    elif cfg.ENV_NAME == 'Robosuite-v0':
        print("[RoboSuite Eval] Dynamically calculating MAX_LIMBS/JOINTS for eval agents...")
        controller_config = load_controller_config(default_controller="JOINT_VELOCITY") # Use simple controller
        for robot_name in agents:
            env = None
            try:
                # Instantiate minimal wrapper - use minimal config from loaded cfg
                minimal_robosuite_cfg = cfg.ROBOSUITE.clone()
                minimal_robosuite_cfg.defrost()
                minimal_robosuite_cfg.ENV_ARGS = {'has_renderer': False, 'has_offscreen_renderer': True, 'horizon': 500} # Minimal
                minimal_robosuite_cfg.freeze()
                env = RobosuiteEnvWrapper(
                    robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
                    robot_name=robot_name,
                    controller_name="JOINT_VELOCITY", # Use simple controller name
                    robosuite_cfg=minimal_robosuite_cfg # Pass minimal cfg node
                )
                metadata = env.metadata.get('robot_metadata', {})
                if not metadata: continue
                # --- ADAPT BASED ON FINAL NODE/JOINT DEFINITION in wrappers ---
                # This needs to match how your node-centric wrappers count things
                # Example: Assuming link-based nodes, and joints = arm+gripper DoF
                num_limbs_current = metadata.get('num_links', metadata.get('num_nodes', 0)) # Use 'num_nodes' as fallback
                num_joints_current = metadata.get('num_arm_joints', 0) + metadata.get('num_gripper_joints', 0)
                # --- End adaptation section ---
                if num_limbs_current > 0: num_limbs.append(num_limbs_current)
                if num_joints_current > 0: num_joints.append(num_joints_current)
            except Exception as e: print(f"Warning: Failed to inspect robot {robot_name}: {e}")
            finally:
                if env: env.close()

        if not num_limbs or not num_joints:
             print("Error: Could not determine dimensions for RoboSuite eval robots.")
             cfg.MODEL.MAX_LIMBS = cfg.MODEL.get('MAX_LIMBS', 8)
             cfg.MODEL.MAX_JOINTS = cfg.MODEL.get('MAX_JOINTS', 8)
        else:
             cfg.MODEL.MAX_LIMBS = max(num_limbs) + padding_needed
             cfg.MODEL.MAX_JOINTS = max(num_joints) + padding_needed
        print(f"[RoboSuite Eval] Calculated MAX_LIMBS={cfg.MODEL.MAX_LIMBS}, MAX_JOINTS={cfg.MODEL.MAX_JOINTS}")

    else:
        print(f"Warning: Unknown ENV_NAME '{cfg.ENV_NAME}'. Keeping configured MAX_LIMBS/JOINTS.")
        if not hasattr(cfg.MODEL, 'MAX_LIMBS') or not cfg.MODEL.get('MAX_LIMBS'): cfg.MODEL.MAX_LIMBS = 8
        if not hasattr(cfg.MODEL, 'MAX_JOINTS') or not cfg.MODEL.get('MAX_JOINTS'): cfg.MODEL.MAX_JOINTS = 8


def maybe_infer_agents():
    """Sets cfg.ENV.AGENTS based on evaluation args if not already set."""
    # This is mostly redundant here as main_stats sets cfg.ENV.AGENTS first,
    # but acts as a safeguard if called independently.
    if not cfg.ENV.get('AGENTS'):
        print("Warning: cfg.ENV.AGENTS not set before maybe_infer_agents in eval script.")

# --- End copied functions ---


# --- Evaluation Function ---
def evaluate_rollouts(policy, env, num_episodes_target):
    """Runs rollouts until a target number of episodes are completed."""
    num_envs = env.num_envs
    all_episode_returns = []
    episode_lengths = []
    episode_successes = []
    obs = env.reset()
    current_returns = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs)
    episodes_collected = 0
    pbar = tqdm(total=num_episodes_target, desc="Evaluating Rollouts", leave=False)
    while episodes_collected < num_episodes_target:
        with torch.no_grad(): _, action, _, _, _ = policy.act(obs, compute_val=False)
        try: obs, reward, step_dones, infos = env.step(action)
        except Exception as e: print(f"\nStep Error: {e}. Aborting agent."); break
        current_returns += reward.cpu().numpy().flatten(); current_lengths += 1
        for i in range(num_envs):
            if step_dones[i]:
                if episodes_collected < num_episodes_target:
                    all_episode_returns.append(current_returns[i])
                    episode_lengths.append(current_lengths[i])
                    success = infos[i].get('success', False) or infos[i].get('episode', {}).get('success', False)
                    episode_successes.append(success)
                    episodes_collected += 1; pbar.update(1)
                current_returns[i] = 0; current_lengths[i] = 0
        if episodes_collected >= num_episodes_target: break
    pbar.close()
    if episodes_collected < num_episodes_target: print(f"\nWarning: Collected only {episodes_collected}/{num_episodes_target} episodes.")
    results = {"all_returns": all_episode_returns, "mean_return": np.mean(all_episode_returns) if all_episode_returns else 0, "std_return": np.std(all_episode_returns) if all_episode_returns else 0, "median_return": np.median(all_episode_returns) if all_episode_returns else 0, "min_return": np.min(all_episode_returns) if all_episode_returns else 0, "max_return": np.max(all_episode_returns) if all_episode_returns else 0, "mean_length": np.mean(episode_lengths) if episode_lengths else 0, "success_rate": np.mean(episode_successes) if episode_successes else 0, "num_episodes": len(all_episode_returns)}
    return results

def plot_histogram(returns, save_path, title="Episode Return Distribution", bins=30):
    """Generates and saves a histogram of episode returns."""
    if not returns: print("No return data to plot histogram."); return
    try:
        plt.figure(figsize=(10, 6)); plt.hist(returns, bins=bins, edgecolor='black', alpha=0.7)
        mean_ret = np.mean(returns); std_ret = np.std(returns)
        plt.axvline(mean_ret, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_ret:.2f}')
        plt.title(f"{title}\n(N={len(returns)}, Mean={mean_ret:.2f}, Std={std_ret:.2f})")
        plt.xlabel("Episode Return"); plt.ylabel("Frequency"); plt.legend(); plt.grid(axis='y', alpha=0.5)
        plt.tight_layout(); plt.savefig(save_path); plt.close()
        print(f"Histogram saved to {save_path}")
    except Exception as e: print(f"Error plotting histogram: {e}")

def main_stats():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy")
    parser.add_argument("--policy_dir", required=True, help="Directory containing config.yaml and model checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file name. Auto-detects if None.")
    parser.add_argument("--test_agent_dir", default=None, help="Directory for test agents (UniMal/Modular)")
    parser.add_argument("--test_robots", nargs='+', default=None, help="List of robot names for Robosuite evaluation")
    parser.add_argument("--num_episodes", type=int, default=50, help="Total number of evaluation episodes per agent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (policy mean)")
    parser.add_argument("--out_file", default="eval_results.json", help="Output JSON file name")
    parser.add_argument("--hist_file", default="eval_return_histogram.png", help="Output histogram file name")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments for evaluation rollouts")
    parser.add_argument("--record_video", action="store_true", help="Enable video recording (first env instance)")
    parser.add_argument("--video_length", type=int, default=500, help="Max length of recorded videos in steps")
    parser.add_argument("--video_dir_suffix", default="_videos", help="Suffix for video directory (created inside policy_dir)")
    args = parser.parse_args()

    config_path = os.path.join(args.policy_dir, "config.yaml")
    if not os.path.exists(config_path): raise FileNotFoundError(f"Config file not found: {config_path}")
    cfg.merge_from_file(config_path)

    cfg.defrost()
    # Use evaluation settings for num_envs, seed, deterministic only for stat rollouts
    # Video recording will use num_envs=1 temporarily if needed
    cfg.RNG_SEED = args.seed
    cfg.DETERMINISTIC = args.deterministic 
    cfg.TERMINATE_ON_FALL = False
    cfg.LOGGING.USE_WANDB = False

    cfg.ROBOSUITE.ENV_ARGS.has_offscreen_renderer = True
    cfg.ROBOSUITE.ENV_ARGS.has_renderer = False 

    agents_to_evaluate = []
    if cfg.ENV_NAME == "Robosuite-v0":
        agents_to_evaluate = args.test_robots if args.test_robots else cfg.ROBOSUITE.get('ROBOTS', [])
        if not agents_to_evaluate: raise ValueError("No RoboSuite robots specified for evaluation.")
        cfg.ROBOSUITE.ROBOTS = agents_to_evaluate
    else:
        if not args.test_agent_dir: raise ValueError("--test_agent_dir required for non-Robosuite.")
        cfg.ENV.WALKER_DIR = args.test_agent_dir
        try:
            xml_dir = os.path.join(args.test_agent_dir, 'xml')
            agents_to_evaluate = [f.split('.')[0] for f in os.listdir(xml_dir) if f.endswith('.xml')]
        except FileNotFoundError: raise FileNotFoundError(f"Cannot find 'xml' subdirectory in {args.test_agent_dir}")
        if not agents_to_evaluate: raise ValueError(f"No agent XML files found in {xml_dir}")
    cfg.ENV.AGENTS = agents_to_evaluate # Set the definitive list
    print(f"Evaluating {cfg.ENV_NAME} with agents: {agents_to_evaluate}")
    set_cfg_options()
    cfg.freeze()

    print("Loading policy and normalization stats...")
    if args.checkpoint: checkpoint_filename = args.checkpoint
    else: 
        potential_names=[f"{cfg.ENV_NAME}.pt", "checkpoint_-1.pt"]
        checkpoint_filename=None
        for name in potential_names: 
            if os.path.exists(os.path.join(args.policy_dir, name)): 
                checkpoint_filename = name
            break
        if checkpoint_filename is None:
             raise FileNotFoundError(f"Checkpoint not found in {args.policy_dir}. Tried: {potential_names}")
    checkpoint_path = os.path.join(args.policy_dir, checkpoint_filename)
    try:
        loaded_data = torch.load(checkpoint_path, map_location=torch.device(cfg.DEVICE)) # Ensure device
        if isinstance(loaded_data, (list, tuple)) and len(loaded_data) == 2:
             policy_obj, saved_ob_rms = loaded_data
             if hasattr(policy_obj, 'state_dict'): policy_state_dict = policy_obj.state_dict()
             else: raise TypeError("Loaded object is not a model or state dict.")
             print("Loaded model object and ob_rms. Extracted state_dict.")
        elif isinstance(loaded_data, dict):
             policy_state_dict = loaded_data; saved_ob_rms = None
             print("Warning: Loaded state dict only, ob_rms missing.")
        else: raise TypeError(f"Unexpected data type in checkpoint: {type(loaded_data)}")
        if not isinstance(policy_state_dict, dict): raise TypeError("Extracted policy data is not state dict.")
    except Exception as e: raise RuntimeError(f"Could not load checkpoint '{checkpoint_path}': {e}") from e


    print("Initializing ActorCritic...")
    first_agent = agents_to_evaluate[0]
    temp_env_factory = make_env(cfg.ENV_NAME, cfg.RNG_SEED, 0,
                                        xml_file=first_agent if cfg.ENV_NAME != "Robosuite-v0" else None,
                                        robot_name=first_agent if cfg.ENV_NAME == "Robosuite-v0" else None)
    # Use DummyVecEnv with one instance to get spaces
    init_envs_instance = DummyVecEnv([temp_env_factory])

    try: actor_critic = ActorCritic(init_envs_instance.observation_space, init_envs_instance.action_space)
    except Exception as e: print("\nERROR initializing ActorCritic!"); print("Obs Space:", init_envs_instance.observation_space); print("Act Space:", init_envs_instance.action_space); init_envs_instance.close(); raise e
    init_envs_instance.close()

    actor_critic.load_state_dict(policy_state_dict)
    actor_critic.to(cfg.DEVICE); actor_critic.eval(); policy = Agent(actor_critic)
    print("Policy initialized.")

    print(f"\nStarting evaluation...")
    all_agent_results = {}
    all_returns_for_hist = []

    for agent_name in agents_to_evaluate:
        print(f"\n--- Evaluating: {agent_name} ---")
        current_xml_file = agent_name if cfg.ENV_NAME != "Robosuite-v0" else None
        current_robot_name = agent_name if cfg.ENV_NAME == "Robosuite-v0" else None

        num_eval_envs = 1 if args.record_video else args.num_envs

        env_fns = [
            make_env(cfg.ENV_NAME, args.seed, i, xml_file=current_xml_file, robot_name=current_robot_name)
            for i in range(num_eval_envs) # Use potentially adjusted num_envs
        ]

        vecenv_class = SubprocVecEnv if cfg.VECENV.TYPE == "SubprocVecEnv" and not args.record_video else DummyVecEnv
        agent_envs_base = vecenv_class(env_fns)

        eval_env = agent_envs_base # Start with base
        if args.record_video:
             if not isinstance(agent_envs_base, DummyVecEnv) or agent_envs_base.num_envs != 1:
                  print("Warning: Recreating env as DummyVecEnv(N=1) for video recording.")
                  agent_envs_base.close() # Close original VecEnv
                  eval_env = DummyVecEnv([env_fns[0]]) # Create new DummyVecEnv(N=1)
             else:
                  eval_env = agent_envs_base # Already suitable

             agent_video_dir = os.path.join(args.policy_dir, agent_name + args.video_dir_suffix)
             os.makedirs(agent_video_dir, exist_ok=True)
             video_prefix = f"{agent_name}_seed{args.seed}"
             if args.deterministic: video_prefix += "_deterministic"

             eval_env = VecVideoRecorder(
                 eval_env, agent_video_dir,
                 record_video_trigger=lambda x: x < 1, # Record frame 0 (first step)
                 video_length=args.video_length, file_prefix=video_prefix
             )
             print(f"Video recording enabled for {agent_name} to {agent_video_dir}")

        # if saved_ob_rms:
        #     eval_env = VecNormalize(eval_env, training=False, ob=True, ret=False)
        #     set_ob_rms(eval_env, saved_ob_rms)
        # vec_norm_eval = get_vec_normalize(eval_env) # Get ref after potential wrapping
        # if vec_norm_eval:
        #      vec_norm_eval.eval() # Ensure eval mode
        # Determine keys to normalize FROM THE LOADED CONFIG
        keys_to_normalize = cfg.MODEL.get('OBS_TO_NORM')
        apply_norm = isinstance(keys_to_normalize, list) and len(keys_to_normalize) > 0

        if apply_norm:
            # print(f"Applying VecNormalize with keys: {keys_to_normalize}")
            # Pass the keys from config to the wrapper
            eval_env = VecNormalize(eval_env, training=False, ob=True, ret=False,
                                    obs_to_norm=keys_to_normalize) 
            if saved_ob_rms:
                 set_ob_rms(eval_env, saved_ob_rms) # Set stats AFTER wrapping
            eval_env.eval() # Ensure it doesn't update stats
        # else: print("Skipping VecNormalize.")

        eval_env = VecPyTorch(eval_env, cfg.DEVICE)
        # Rollouts
        num_episodes_this_agent = args.num_episodes # Target total episodes for this agent
        results = evaluate_rollouts(policy, eval_env, num_episodes_this_agent)
        all_agent_results[agent_name] = results
        all_returns_for_hist.extend(results["all_returns"])
        print(f"  Result: Mean Return={results['mean_return']:.2f} +/- {results['std_return']:.2f}, Success Rate={results['success_rate']:.2%}, N={results['num_episodes']}")

        eval_env.close()
        if args.record_video and agent_video_dir:
             print(f"Video saving finalized for {agent_name}.")

    print("\n--- Overall Evaluation Summary ---")
    overall_returns = [res['mean_return'] for res in all_agent_results.values() if res['num_episodes'] > 0]
    overall_success = [res['success_rate'] for res in all_agent_results.values() if res['num_episodes'] > 0]
    total_episodes_eval = sum(res['num_episodes'] for res in all_agent_results.values())
    print(f"Total Episodes Evaluated: {total_episodes_eval}")
    print(f"Average Mean Return Across Agents: {np.mean(overall_returns):.2f}" if overall_returns else "N/A")
    print(f"Average Success Rate Across Agents: {np.mean(overall_success):.2%}" if overall_success else "N/A")


    output_path = os.path.join(args.policy_dir, args.out_file)
    try:
        with open(output_path, 'w') as f:
             def json_converter(obj):
                  if isinstance(obj, np.ndarray): return obj.tolist()
                  if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
                  elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
                  elif isinstance(obj, (np.bool_)): return bool(obj)
                  elif isinstance(obj, (np.void)): return None # Handle void/structured arrays if they appear
                  return obj # Let default handle others, will raise TypeError if needed
             json.dump(all_agent_results, f, default=json_converter, indent=4)
        print(f"Detailed results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # Plot and save histogram
    hist_path = os.path.join(args.policy_dir, args.hist_file)
    policy_name = os.path.basename(args.policy_dir.rstrip('/'))
    plot_histogram(all_returns_for_hist, hist_path, title=f"Episode Return Distribution ({policy_name}) - All Agents")

if __name__ == "__main__":
    # Define or import set_cfg_options and its dependencies here
    # For simplicity, assuming they are now defined within this file
    main_stats()