import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt 
import numpy as np
import torch
from tqdm import tqdm 

from metamorph.algos.ppo.envs import (get_ob_rms, get_vec_normalize,
                                      make_vec_envs, set_ob_rms)
from metamorph.algos.ppo.model import ActorCritic, Agent
from metamorph.config import cfg, load_cfg, set_cfg_options


def evaluate_rollouts(policy, env, num_episodes_target):
    """
    Runs rollouts until a target number of episodes are completed across all envs.

    Args:
        policy: The policy agent (metamorph.algos.ppo.model.Agent).
        env: The vectorized environment instance.
        num_episodes_target: The total number of episodes to collect across all parallel envs.

    Returns:
        A dictionary containing evaluation statistics, including 'all_returns'.
    """
    num_envs = env.num_envs
    all_episode_returns = []
    episode_lengths = []
    episode_successes = [] 
    obs = env.reset()
    current_returns = np.zeros(num_envs)
    current_lengths = np.zeros(num_envs)
    episodes_collected = 0

    pbar = tqdm(total=num_episodes_target, desc="Evaluating Rollouts")

    while episodes_collected < num_episodes_target:
        # Get action from policy
        with torch.no_grad():
            # cfg.DETERMINISTIC controls deterministic vs stochastic actions
            _, action, _, _, _ = policy.act(obs, compute_val=False)

        # Step environment
        try:
            obs, reward, step_dones, infos = env.step(action)
        except Exception as e:
             print(f"\nError during env.step(): {e}")
             # Decide how to handle step errors during eval: skip or fail?
             # For now, let's break and report what we have
             print("Aborting evaluation for this agent due to step error.")
             break # Exit the while loop

        current_returns += reward.cpu().numpy().flatten() # Assuming reward is (N, 1)
        current_lengths += 1

        # Handle dones
        for i in range(num_envs):
            # Important: Check the actual 'done' signal from the step, NOT just the info dict
            if step_dones[i]:
                if episodes_collected < num_episodes_target:
                    all_episode_returns.append(current_returns[i])
                    episode_lengths.append(current_lengths[i])
                    # Check for success key robustly
                    success = infos[i].get('success', False) # From RobosuiteEnvWrapper
                    if not success and "episode" in infos[i]: # Fallback check in episode dict
                        success = infos[i]["episode"].get('success', False)
                    episode_successes.append(success)

                    episodes_collected += 1
                    pbar.update(1)

                # Reset tracking for the finished episode in this env
                current_returns[i] = 0
                current_lengths[i] = 0
                # VecEnv handles the internal env reset automatically when step_dones[i] is True

        # Safety break in case loop condition is tricky
        if episodes_collected >= num_episodes_target:
             break

    pbar.close()
    if episodes_collected < num_episodes_target:
        print(f"\nWarning: Collected only {episodes_collected}/{num_episodes_target} episodes.")


    results = {
        "all_returns": all_episode_returns, # Return the full list for histogram
        "mean_return": np.mean(all_episode_returns) if all_episode_returns else 0,
        "std_return": np.std(all_episode_returns) if all_episode_returns else 0,
        "median_return": np.median(all_episode_returns) if all_episode_returns else 0,
        "min_return": np.min(all_episode_returns) if all_episode_returns else 0,
        "max_return": np.max(all_episode_returns) if all_episode_returns else 0,
        "mean_length": np.mean(episode_lengths) if episode_lengths else 0,
        "success_rate": np.mean(episode_successes) if episode_successes else 0,
        "num_episodes": len(all_episode_returns),
    }
    return results

# --- Histogram Plotting Function ---

def plot_histogram(returns, save_path, title="Episode Return Distribution", bins=30):
    """Generates and saves a histogram of returns."""
    if not returns:
        print("No return data to plot histogram.")
        return
    try:
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=bins, edgecolor='black', alpha=0.7)
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        plt.axvline(mean_ret, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_ret:.2f}')
        plt.title(f"{title}\n(N={len(returns)}, Mean={mean_ret:.2f}, Std={std_ret:.2f})")
        plt.xlabel("Episode Return")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() # Close the plot to free memory
        print(f"Histogram saved to {save_path}")
    except Exception as e:
        print(f"Error plotting histogram: {e}")

# --- Main Evaluation Logic ---

def main_stats():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL policy")
    parser.add_argument("--policy_dir", required=True, help="Directory containing config.yaml and model checkpoint")
    parser.add_argument("--checkpoint", default=None, help="Name of the model checkpoint file (e.g., Unimal-v0.pt or checkpoint_1000.pt). If None, tries common names.")
    parser.add_argument("--test_agent_dir", default=None, help="Directory for test agents (e.g., unimals_100/test), required if not Robosuite")
    parser.add_argument("--test_robots", nargs='+', default=None, help="List of robot names for Robosuite evaluation (overrides cfg)")
    parser.add_argument("--num_episodes", type=int, default=50, help="Total number of episodes to collect across all envs/agents")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions (policy mean)")
    parser.add_argument("--out_file", default="eval_results.json", help="Output file for results (JSON), saved in policy_dir")
    parser.add_argument("--hist_file", default="eval_return_histogram.png", help="Output file for histogram plot, saved in policy_dir")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments for evaluation")
    args = parser.parse_args()

    # --- 1. Load Config ---
    config_path = os.path.join(args.policy_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    cfg.merge_from_file(config_path)

    # --- 2. Override Config for Evaluation ---
    cfg.defrost()
    cfg.PPO.NUM_ENVS = args.num_envs # Use eval num_envs
    cfg.RNG_SEED = args.seed
    cfg.DETERMINISTIC = args.deterministic
    cfg.TERMINATE_ON_FALL = False # Typically off for eval
    cfg.LOGGING.USE_WANDB = False # Disable logging during eval

    # Determine agents/robots to evaluate
    agents_to_evaluate = []
    if cfg.ENV_NAME == "Robosuite-v0":
        agents_to_evaluate = args.test_robots if args.test_robots else cfg.ROBOSUITE.ROBOTS
        if not agents_to_evaluate: # Fallback if test_robots and config list are empty
             raise ValueError("No RoboSuite robots specified for evaluation (--test_robots or in config).")
        cfg.ROBOSUITE.ROBOTS = agents_to_evaluate # Ensure config matches eval set
        cfg.ENV.AGENTS = agents_to_evaluate # Keep consistent
        print(f"Evaluating RoboSuite with robots: {agents_to_evaluate}")
    else: # UniMorph / Modular
        if not args.test_agent_dir:
            raise ValueError("--test_agent_dir is required for non-Robosuite environments")
        cfg.ENV.WALKER_DIR = args.test_agent_dir # Keep original key if needed by legacy code
        try:
            agent_files = [f for f in os.listdir(os.path.join(args.test_agent_dir, 'xml')) if f.endswith('.xml')]
            agents_to_evaluate = [f.split('.')[0] for f in agent_files]
        except FileNotFoundError:
             raise FileNotFoundError(f"Cannot find 'xml' subdirectory in --test_agent_dir: {args.test_agent_dir}")
        if not agents_to_evaluate:
             raise ValueError(f"No agent XML files found in {os.path.join(args.test_agent_dir, 'xml')}")
        cfg.ENV.AGENTS = agents_to_evaluate
        print(f"Evaluating {cfg.ENV_NAME} with agents from {args.test_agent_dir}")

    set_cfg_options() # Recalculate MAX_LIMBS etc. based on the *loaded evaluation config*
    cfg.freeze()

    # --- 3. Setup Shared Policy and Normalization ---
    print("Loading policy and normalization stats...")

    # Find checkpoint file
    if args.checkpoint:
        checkpoint_filename = args.checkpoint
    else: # Auto-detect common names
        potential_names = [f"{cfg.ENV_NAME}.pt", "checkpoint_-1.pt"]
        checkpoint_filename = None
        for name in potential_names:
             if os.path.exists(os.path.join(args.policy_dir, name)):
                  checkpoint_filename = name
                  print(f"Auto-detected checkpoint: {name}")
                  break
        if checkpoint_filename is None:
             raise FileNotFoundError(f"Could not find checkpoint file in {args.policy_dir}. Tried: {potential_names}")

    checkpoint_path = os.path.join(args.policy_dir, checkpoint_filename)

    # Load checkpoint data
    try:
        # Assume checkpoint might be just the state dict or a list/tuple
        loaded_data = torch.load(checkpoint_path, map_location=cfg.DEVICE)
        if isinstance(loaded_data, (list, tuple)) and len(loaded_data) == 2:
            policy_state_dict, saved_ob_rms = loaded_data
            print("Loaded policy state dict and ob_rms from checkpoint.")
        elif isinstance(loaded_data, dict): # Assume it's just the state dict
            policy_state_dict = loaded_data
            saved_ob_rms = None
            print("Warning: Loaded state dict only from checkpoint, ob_rms missing.")
        elif hasattr(loaded_data, 'state_dict'): # Check if it's a model object
             policy_state_dict = loaded_data.state_dict()
             saved_ob_rms = None # Assume no ob_rms saved with full model object
             print("Warning: Loaded full model object, extracting state dict. ob_rms assumed missing.")
        else:
             raise TypeError(f"Unexpected data type loaded from checkpoint: {type(loaded_data)}")

        # Verify policy_state_dict is actually a dict
        if not isinstance(policy_state_dict, dict):
             raise TypeError(f"Extracted policy data is not a state dictionary: {type(policy_state_dict)}")

    except Exception as e:
        raise RuntimeError(f"Could not load or parse policy checkpoint '{checkpoint_path}': {e}") from e


    # --- 4. Create VecEnv ONCE to initialize ActorCritic ---
    print("Initializing ActorCritic with environment spaces...")
    # Temporarily set config for the first agent to get spaces
    global_cfg_backup = cfg.clone()
    cfg.defrost()
    first_agent = agents_to_evaluate[0]
    if cfg.ENV_NAME == "Robosuite-v0": cfg.ROBOSUITE.ROBOTS = [first_agent]
    cfg.ENV.AGENTS = [first_agent]
    cfg.PPO.NUM_ENVS = 1 # Only need one env instance
    # set_cfg_options() # Re-run in case max dims differ for first agent
    cfg.freeze()

    # Create the temporary env
    init_envs = make_vec_envs(training=False, norm_rew=False, seed=args.seed, num_env=1)

    # Restore global cfg immediately
    cfg.clear()
    cfg.update(global_cfg_backup)

    # Initialize Policy using spaces from init_envs
    try:
        actor_critic = ActorCritic(init_envs.observation_space, init_envs.action_space)
    except Exception as e:
         print("\nERROR initializing ActorCritic!")
         print("Observation Space:", init_envs.observation_space)
         print("Action Space:", init_envs.action_space)
         init_envs.close()
         raise e
    init_envs.close() # Close the temporary env

    actor_critic.load_state_dict(policy_state_dict)
    actor_critic.to(cfg.DEVICE)
    actor_critic.eval() # Set to evaluation mode
    policy = Agent(actor_critic)
    print("Policy initialized successfully.")

    # --- 5. Run Evaluation Per Agent ---
    print(f"Starting evaluation for {len(agents_to_evaluate)} agents/robots...")
    all_agent_results = {}
    all_returns_for_hist = [] # Collect all returns across all agents

    for agent_name in agents_to_evaluate:
        print(f"\n--- Evaluating: {agent_name} ---")
        # Create envs specific to this agent for evaluation
        # Temporarily override global cfg
        global_cfg_backup_agent = cfg.clone()
        cfg.defrost()
        if cfg.ENV_NAME == "Robosuite-v0": cfg.ROBOSUITE.ROBOTS = [agent_name]
        cfg.ENV.AGENTS = [agent_name]
        cfg.PPO.NUM_ENVS = args.num_envs # Use eval num_envs
        # set_cfg_options() # May not be needed if MAX dims are consistent across test set
        cfg.freeze()

        # Create the actual VecEnv for this agent
        agent_envs = make_vec_envs(training=False, norm_rew=False, seed=args.seed)

        # Restore global cfg
        cfg.clear()
        cfg.update(global_cfg_backup_agent)

        # Apply normalization stats
        if saved_ob_rms:
            set_ob_rms(agent_envs, saved_ob_rms)
        else:
            # Ensure VecNormalize doesn't update stats during eval even if ob_rms was missing
            vec_norm_agent = get_vec_normalize(agent_envs)
            if vec_norm_agent: vec_norm_agent.eval()

        results = evaluate_rollouts(policy, agent_envs, args.num_episodes)
        all_agent_results[agent_name] = results # Store detailed results per agent
        all_returns_for_hist.extend(results["all_returns"]) # Add returns to histogram list
        print(f"  Result: Mean Return={results['mean_return']:.2f} +/- {results['std_return']:.2f}, Success Rate={results['success_rate']:.2%}, N={results['num_episodes']}")
        agent_envs.close()

    # --- 6. Aggregate, Save Results, and Plot Histogram ---
    print("\n--- Overall Evaluation Summary ---")
    overall_returns = [res['mean_return'] for res in all_agent_results.values() if res['num_episodes'] > 0]
    overall_success = [res['success_rate'] for res in all_agent_results.values() if res['num_episodes'] > 0]
    total_episodes_eval = sum(res['num_episodes'] for res in all_agent_results.values())
    print(f"Total Episodes Evaluated: {total_episodes_eval}")
    print(f"Average Mean Return Across Agents: {np.mean(overall_returns):.2f}" if overall_returns else "N/A")
    print(f"Average Success Rate Across Agents: {np.mean(overall_success):.2%}" if overall_success else "N/A")


    # Save detailed results
    output_path = os.path.join(args.policy_dir, args.out_file)
    try:
        with open(output_path, 'w') as f:
             # Convert numpy types to standard python types for JSON
             json.dump(all_agent_results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else (int(x) if isinstance(x, np.int_) else (float(x) if isinstance(x, np.float_) else x)), indent=4)
        print(f"Detailed results saved to {output_path}")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


    # Plot and save histogram
    hist_path = os.path.join(args.policy_dir, args.hist_file)
    policy_name = os.path.basename(args.policy_dir.rstrip('/')) # Get name from policy dir
    plot_histogram(all_returns_for_hist, hist_path, title=f"Episode Return Distribution ({policy_name}) - All Agents")


if __name__ == "__main__":
    main_stats()