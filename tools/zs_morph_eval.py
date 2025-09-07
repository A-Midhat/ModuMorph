import argparse
import os
import sys
import torch
import numpy as np
from tqdm import tqdm

from metamorph.config import cfg
from metamorph.algos.ppo.envs import make_vec_envs
from metamorph.algos.ppo.envs import set_ob_rms
from metamorph.algos.ppo.model import Agent
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
# output/panda_iiwa_door_lift_1409/Robosuite-v0.pt
"""
Example:
python tools/zs_morph_eval.py \
  --run_dir ./output/MR-MT_newLogic_3TASKS/ \
  --checkpoint Robosuite-v0.pt \
  --morph Panda \
  --task PickPlaceMilk \
  --controller OSC_POSE \
  --episodes 5 \
  --save_video ./test_videos/ \
  --embedding_strategy average_task \
  --debug

Available embedding strategies:
- 'nearest_task': Use embedding of the same task from training (recommended)
- 'average_task': Use average of all embeddings for the same task
- 'random_task': Use a random task embedding from training
- 'zero': Use zero embedding (baseline)
"""

def parse_args():
    """Parses the arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on unseen morphology and save a video.")
    parser.add_argument("--run_dir", required=True, type=str, help="Path to the training output directory")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint filename")
    parser.add_argument("--morph", required=True, type=str, help="UNSEEN robot morphology to test (e.g., Baxter)")
    parser.add_argument("--task", required=True, type=str, help="Task to test (should be from training set)")
    parser.add_argument("--controller", required=True, type=str, help="Controller to use (e.g., OSC_POSE)")
    parser.add_argument("--episodes", default=5, type=int, help="Number of episodes for evaluation")
    parser.add_argument("--save_video", default=None, type=str, help="Directory to save evaluation videos")
    parser.add_argument("--gripper", default=None, type=str, help="Override gripper type")
    parser.add_argument("--embedding_strategy", default="nearest_task", type=str, 
                       choices=["nearest_task", "average_task", "random_task", "zero"],
                       help="Strategy for selecting task embedding for unseen morphology")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def debug_print(message, debug_enabled):
    if debug_enabled:
        print(f"[DEBUG] {message}")

def _to_scalar_float(x):
    """Robust conversion of reward-like objects to python float."""
    try:
        if isinstance(x, torch.Tensor):
            return float(x.cpu().detach().numpy().reshape(-1)[0])
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(x, (_np.ndarray,)):
            return float(_np.asarray(x).reshape(-1)[0])
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        try:
            return float(np.asarray(x).reshape(-1)[0])
        except Exception:
            return 0.0

def get_task_embedding_indices(task_name, original_tasks):
    """Get all indices where the specified task appears in training."""
    indices = [i for i, t in enumerate(original_tasks) if t == task_name]
    return indices

def select_embedding_index(task_name, original_tasks, original_morphs, strategy, task_embed_module, device, debug=False):
    """
    Select embedding index for unseen morphology based on strategy.
    
    Args:
        task_name: Target task name
        original_tasks: List of all training tasks
        original_morphs: List of all training morphologies  
        strategy: Embedding selection strategy
        task_embed_module: Task embedding module from model
        device: Torch device
        debug: Debug flag
    
    Returns:
        tuple: (embedding_index, embedding_description)
    """
    
    # Get unique tasks and find task index
    unique_tasks = list(dict.fromkeys(original_tasks))
    if task_name not in unique_tasks:
        raise ValueError(f"Task '{task_name}' not found in training tasks: {unique_tasks}")
    
    # CRITICAL FIX: Use unique task index, not pair index
    unique_task_idx = unique_tasks.index(task_name)
    task_indices = get_task_embedding_indices(task_name, original_tasks)
    
    debug_print(f"Unique tasks: {unique_tasks}", debug)
    debug_print(f"Task '{task_name}' unique index: {unique_task_idx}", debug)
    debug_print(f"Task '{task_name}' appears at training pair indices: {task_indices}", debug)
    
    # Check model embedding size
    num_embeddings = task_embed_module.num_embeddings if task_embed_module else len(unique_tasks)
    debug_print(f"Model has {num_embeddings} task embeddings", debug)
    
    if unique_task_idx >= num_embeddings:
        raise ValueError(f"Task index {unique_task_idx} >= model embeddings {num_embeddings}")
    
    if strategy == "nearest_task":
        # Use unique task index (correct approach)
        selected_idx = unique_task_idx
        first_pair_morph = original_morphs[task_indices[0]]
        description = f"unique task index {selected_idx} for '{task_name}' (first trained with {first_pair_morph})"
        
    elif strategy == "average_task":
        # For unique task embeddings, this is the same as nearest_task
        selected_idx = unique_task_idx
        description = f"unique task index {selected_idx} for '{task_name}' (same as nearest for unique embeddings)"
                
    elif strategy == "random_task":
        # For unique task embeddings, this is the same as nearest_task
        selected_idx = unique_task_idx
        description = f"unique task index {selected_idx} for '{task_name}' (same as nearest for unique embeddings)"
        
    elif strategy == "zero":
        # Use zero embedding (create a zero tensor)
        selected_idx = 0  # We'll handle this specially
        description = "zero embedding (baseline)"
        
    else:
        raise ValueError(f"Unknown embedding strategy: {strategy}")
    
    debug_print(f"Selected embedding: {description}", debug)
    return selected_idx, description

def create_zero_embedding_tensor(task_embed_module, device):
    """Create a zero tensor matching the task embedding dimensions."""
    if task_embed_module is not None:
        embedding_dim = task_embed_module.embedding_dim
        return torch.zeros(1, embedding_dim, device=device)
    else:
        # Fallback dimensions if no task embedding module
        return torch.zeros(1, 64, device=device)  # Common embedding size

def main():
    args = parse_args()

    # --- 1. Load config ---
    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg.merge_from_file(config_path)
    cfg.freeze()

    # Get original training configuration
    if hasattr(cfg.ROBOSUITE, "TRAINING_MORPOLOGIES"):
        original_morphs = list(cfg.ROBOSUITE.TRAINING_MORPOLOGIES)
    else:
        original_morphs = list(getattr(cfg.ROBOSUITE, "TRAINING_MORPHOLOGIES", []))
    if not original_morphs:
        original_morphs = list(getattr(cfg.ROBOSUITE, "TRAINING_MORPHOLOGIES", []))

    original_tasks = list(cfg.ROBOSUITE.ENV_NAMES)
    original_controllers = list(cfg.ROBOSUITE.CONTROLLERS)

    debug_print(f"Original training morphs: {original_morphs}", args.debug)
    debug_print(f"Original training tasks: {original_tasks}", args.debug)

    # Check if morphology is actually unseen
    if args.morph in original_morphs:
        print(f"⚠️  WARNING: Morphology '{args.morph}' was seen during training!")
        print(f"   Consider using test_eval_best_one.py for seen morphologies.")
        print(f"   Continuing with unseen morphology evaluation...")

    # Check if task was seen during training
    unique_tasks = list(dict.fromkeys(original_tasks))
    if args.task not in unique_tasks:
        print(f"❌ ERROR: Task '{args.task}' was not seen during training!")
        print(f"Available training tasks: {unique_tasks}")
        print(f"For unseen tasks, you need a different evaluation approach.")
        sys.exit(1)

    cfg.defrost()
    
    # --- 2. Setup evaluation config for unseen morphology ---
    cfg.ROBOSUITE.TASK_TYPE = "MR"
    
    # CRITICAL: Create a new configuration that includes the unseen morphology
    # We add the unseen morph-task pair to the configuration 
    eval_morphs = original_morphs + [args.morph]
    eval_tasks = original_tasks + [args.task] 
    eval_controllers = original_controllers + [args.controller]
    
    cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = eval_morphs
    cfg.ROBOSUITE.ENV_NAMES = eval_tasks
    cfg.ROBOSUITE.CONTROLLERS = eval_controllers
    
    # The unseen morphology will be at the last index
    unseen_morph_idx = len(original_morphs)
    
    debug_print(f"Unseen morphology added at index: {unseen_morph_idx}", args.debug)
    debug_print(f"Evaluation will use morph_idx_render = {unseen_morph_idx}", args.debug)

    if args.gripper:
        print(f"Overriding gripper with: {args.gripper}")
        cfg.ROBOSUITE.GRIPPER_TYPES = [args.gripper]

    cfg.PPO.NUM_ENVS = 1
    if args.save_video:
        os.makedirs(args.save_video, exist_ok=True)
        cfg.ROBOSUITE.ENV_ARGS.has_renderer = False
        cfg.ROBOSUITE.ENV_ARGS.has_offscreen_renderer = True
        cfg.ROBOSUITE.ENV_ARGS.use_camera_obs = False

    cfg.freeze()

    # --- 3. Load model ---
    model_path = os.path.join(args.run_dir, args.checkpoint)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found at {model_path}")

    print(f"Loading model from: {model_path}")
    loaded = torch.load(model_path, map_location=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    if isinstance(loaded, tuple) and len(loaded) >= 2:
        actor_critic, ob_rms = loaded[0], loaded[1]
    elif isinstance(loaded, dict) and "actor_critic" in loaded:
        actor_critic = loaded["actor_critic"]
        ob_rms = loaded.get("ob_rms", None)
    else:
        try:
            actor_critic, ob_rms = loaded
        except Exception:
            actor_critic = loaded
            ob_rms = None

    agent = Agent(actor_critic)
    debug_print(f"Model architecture: {actor_critic}", args.debug)

    # --- 4. Setup embedding strategy for unseen morphology ---
    print(f"\n🔍 Unseen Morphology Evaluation Logic:")
    print(f"Target: UNSEEN Morph='{args.morph}', Task='{args.task}'")
    print(f"Embedding Strategy: {args.embedding_strategy}")

    # Get task embedding module
    task_embed_module = None
    try:
        v_net = getattr(actor_critic, "v_net", None)
        if v_net is not None and hasattr(v_net, "task_embed"):
            task_embed_module = v_net.task_embed
    except Exception:
        task_embed_module = None

    device = getattr(cfg, "DEVICE", "cpu")

    # Select embedding based on strategy
    if args.embedding_strategy == "zero":
        # Special handling for zero embedding
        unimal_id_tensor = torch.tensor([0], dtype=torch.long, device=device)  # Will be handled specially in agent
        embedding_description = "zero embedding (baseline)"
        print(f"[Eval] Using zero embedding strategy")
        
        # Modify agent to use zero embedding - this is a bit hacky but necessary
        if task_embed_module is not None:
            # Temporarily set the first embedding to zero for this evaluation
            original_embedding = task_embed_module.weight.data[0].clone()
            task_embed_module.weight.data[0].fill_(0.0)
            print(f"[Eval] Temporarily set embedding[0] to zero")
    else:
        # Use strategy-based selection
        selected_idx, embedding_description = select_embedding_index(
            args.task, original_tasks, original_morphs, args.embedding_strategy, 
            task_embed_module, device, args.debug
        )
        unimal_id_tensor = torch.tensor([selected_idx], dtype=torch.long, device=device)

    print(f"[Eval] Selected: {embedding_description}")
    print(f"[Eval] unimal_id_tensor = {unimal_id_tensor.tolist()}")
    print(f"[Eval] Environment will be created for UNSEEN morph '{args.morph}' at render_idx = {unseen_morph_idx}")

    # Verify embedding can be accessed
    if task_embed_module is not None and selected_idx < task_embed_module.num_embeddings:
        try:
            emb_vec = task_embed_module.weight.data[selected_idx].cpu().numpy()
            print(f"[Eval] Using embedding[{selected_idx}] preview (first 8): {emb_vec.reshape(-1)[:8].tolist()}")
            print(f"[Eval] Embedding L2 norm: {float(np.linalg.norm(emb_vec)):.6f}")
        except Exception as e:
            print(f"[Eval] Could not read embedding vector preview: {e}")

    # --- 5. Evaluation loop ---
    episode_returns = []
    episode_successes = []

    video_kwargs = {"video_dir": args.save_video, "video_prefix": f"UNSEEN_{args.task}_{args.morph}"} if args.save_video else {}

    for i in tqdm(range(args.episodes), desc=f"🤖 Evaluating UNSEEN Morph {args.morph}"):
        envs = make_vec_envs(
            training=False, 
            save_video=bool(args.save_video), 
            morph_idx_render=unseen_morph_idx,  # Use the unseen morphology index
            **video_kwargs
        )
        set_ob_rms(envs, ob_rms)

        debug_print(f"Episode {i+1}: Environment created for unseen morphology", args.debug)

        obs = envs.reset()
        done = [False]
        episode_reward = 0.0
        step_count = 0
        step_rewards = []

        while not done[0]:
            try:
                with torch.no_grad():
                    _, act, _, value, _ = agent.act(obs, unimal_ids=unimal_id_tensor, compute_val=True)
            except RuntimeError as e:
                print(f"\n❌ RuntimeError during agent.act(): {e}")
                print(f"   This might indicate incompatibility between unseen morphology and model architecture")
                try:
                    envs.close()
                except Exception:
                    pass
                raise

            if args.debug and (step_count % 50 == 0):
                try:
                    if isinstance(act, torch.Tensor):
                        a_preview = act.cpu().detach().numpy().reshape(-1)[:6].tolist()
                        debug_print(f"Step {step_count}: Action preview={a_preview}", args.debug)
                    val_f = float(value.item())
                    debug_print(f"Step {step_count}: Value={val_f:.4f}", args.debug)
                except Exception:
                    debug_print(f"Step {step_count}: couldn't print action/value", args.debug)

            obs, reward, done, infos = envs.step(act)

            rew_val = _to_scalar_float(reward)
            episode_reward += rew_val
            step_rewards.append(rew_val)
            step_count += 1

            if args.debug and step_count % 50 == 0:
                debug_print(f"Step {step_count}: Reward: {rew_val:.4f}, Cumulative: {episode_reward:.4f}", args.debug)

        # Close environment and get video path
        video_path = None
        try:
            video_path = envs.close()
        except Exception:
            pass

        # Check for success
        is_success = False
        try:
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info0 = infos[0]
            else:
                info0 = infos
            is_success = info0.get('success', False)
        except Exception:
            is_success = False

        episode_successes.append(bool(is_success))
        episode_returns.append(float(episode_reward))

        print(f"Episode {i+1} completed:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {float(episode_reward):.4f}")
        print(f"  Success: {is_success}")
        if step_rewards:
            print(f"  Avg Step Reward: {np.mean(step_rewards):.4f}")

        # Handle video saving
        if args.save_video and video_path:
            try:
                if is_success:
                    success_str = "SUCCESS"
                    reward_str = int(float(episode_reward))
                    new_video_name = f"UNSEEN_{args.morph}_episode_{i+1}_{success_str}_reward_{reward_str}.mp4"
                    new_video_path = os.path.join(args.save_video, new_video_name)
                    os.rename(video_path, new_video_path)
                    tqdm.write(f"Success! Saved video to {new_video_name}")
                else:
                    try:
                        os.remove(video_path)
                        tqdm.write("Failure. Discarding video.")
                    except Exception:
                        tqdm.write("Failure. Could not remove video.")
            except Exception as e:
                tqdm.write(f"Could not handle video file: {e}")

    # --- 6. Restore original embedding if we modified it ---
    if args.embedding_strategy == "zero" and task_embed_module is not None:
        try:
            task_embed_module.weight.data[0] = original_embedding
            print(f"[Eval] Restored original embedding[0]")
        except Exception as e:
            print(f"[Eval] Could not restore original embedding: {e}")

    # --- 7. Summary ---
    avg_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0
    avg_success_rate = float(np.mean(episode_successes) * 100) if episode_successes else 0.0

    print(f"\n--- UNSEEN MORPHOLOGY Evaluation Statistics ---")
    print(f"Unseen Morphology: {args.morph}")
    print(f"Task: {args.task}")
    print(f"Embedding Strategy: {args.embedding_strategy}")
    print(f"Episodes: {args.episodes}")
    print(f"Avg. Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {avg_success_rate:.1f}%")
    print(f"Individual episode rewards: {episode_returns}")
    print("--------------------------------------------------------\n")

    if args.debug:
        print("\n--- Debug Information ---")
        print(f"Embedding description: {embedding_description}")
        print(f"Original training morphologies: {original_morphs}")
        print(f"Evaluation morphologies: {eval_morphs}")
        print(f"Unseen morphology index: {unseen_morph_idx}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()