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

"""
Example for generalization testing:
python tools/zs_morph_eval.py \
  --run_dir ./artifacts/Robosuite-v0-MR-ST-MR-MT_ALLNODES4_1409-run:v9/ \
  --checkpoint Robosuite-v0.pt \
  --morph UR5e \
  --task Door \
  --controller OSC_POSE \
  --episodes 1 \
  --seed 42 \
  --save_video ./test_generalization/ \
  --debug
"""

def parse_args():
    """Parses the arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on unseen morphologies and save a video.")
    parser.add_argument("--run_dir", required=True, type=str, help="Path to the training output directory")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint filename")
    parser.add_argument("--morph", required=True, type=str, help="Robot morphology to test (can be unseen)")
    parser.add_argument("--task", required=True, type=str, help="Robosuite task to test")
    parser.add_argument("--controller", required=True, type=str, help="Controller to use (e.g., OSC_POSE)")
    parser.add_argument("--episodes", default=5, type=int, help="Number of episodes for evaluation")
    parser.add_argument("--save_video", default=None, type=str, help="Directory to save evaluation videos")
    parser.add_argument("--gripper", default=None, type=str, help="Override gripper type")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--test_all_ids", action="store_true", help="Quick scan for all unimal ids (debug)")


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

def main():
    args = parse_args()
    # --- 1. Load config ---
    config_path = os.path.join(args.run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    cfg.merge_from_file(config_path)
    cfg.freeze()

    # robust retrieval for training morphs (handle potential typo)
    if hasattr(cfg.ROBOSUITE, "TRAINING_MORPOLOGIES"):
        original_morphs = list(cfg.ROBOSUITE.TRAINING_MORPOLOGIES)
    else:
        original_morphs = list(getattr(cfg.ROBOSUITE, "TRAINING_MORPHOLOGIES", []))
    if not original_morphs:
        original_morphs = list(getattr(cfg.ROBOSUITE, "TRAINING_MORPHOLOGIES", []))

    original_tasks = list(cfg.ROBOSUITE.ENV_NAMES)

    debug_print(f"Original training morphs: {original_morphs}", args.debug)
    debug_print(f"Original training tasks: {original_tasks}", args.debug)
    debug_print(f"Original training pairs: {list(zip(original_morphs, original_tasks))}", args.debug)

    cfg.defrost()
    
    # --- 2. Setup evaluation config for GENERALIZATION ---
    cfg.ROBOSUITE.TASK_TYPE = "MR"
    
    # Check if this is generalization (unseen morph) or standard evaluation
    desired_pair = (args.morph, args.task)
    training_pairs = list(zip(original_morphs, original_tasks))
    is_generalization = desired_pair not in training_pairs
    
    if is_generalization:
        print(f"🚀 GENERALIZATION MODE: Testing unseen morphology '{args.morph}' on task '{args.task}'")
        
        # For generalization, we need to:
        # 1. Add the new morph to the config temporarily for environment creation
        # 2. Use task-based embedding indexing (not pair-based)
        
        # Find a training pair with the same task for reference
        task_pairs = [(m, t) for m, t in training_pairs if t == args.task]
        if not task_pairs:
            print(f"❌ ERROR: Task '{args.task}' was not seen during training!")
            print(f"Available training tasks: {list(set(original_tasks))}")
            sys.exit(1)
        
        # Use the first occurrence of this task for render_idx
        reference_pair = task_pairs[0]
        render_idx = training_pairs.index(reference_pair)
        
        print(f"🎯 Using task '{args.task}' (reference pair: {reference_pair} at index {render_idx})")
        print(f"📝 Will create environment with morph '{args.morph}' but use task embedding from training")
        
        # Temporarily extend the training lists to include our test morph
        extended_morphs = original_morphs + [args.morph]
        extended_tasks = original_tasks + [args.task]
        extened_controllers = list(cfg.ROBOSUITE.CONTROLLERS) + [args.controller]
        extended_render_idx = len(original_morphs)  # Index of our new morph
        
        cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = extended_morphs
        cfg.ROBOSUITE.ENV_NAMES = extended_tasks
        cfg.ROBOSUITE.CONTROLLERS = extened_controllers
        
    else:
        print(f"📊 STANDARD MODE: Testing seen morphology-task pair '{args.morph}'-'{args.task}'")
        render_idx = training_pairs.index(desired_pair)
        extended_render_idx = render_idx
        cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = original_morphs[:]
        cfg.ROBOSUITE.ENV_NAMES = original_tasks[:]
        cfg.ROBOSUITE.CONTROLLERS = list(cfg.ROBOSUITE.CONTROLLERS)

    debug_print(f"Evaluation config - Morph: {args.morph}, Task: {args.task}, Controller: {args.controller}", args.debug)
    debug_print(f"Environment will be created at render_idx = {extended_render_idx}", args.debug)
    
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
    debug_print(f"Observation RMS stats available: {ob_rms is not None}", args.debug)

    # --- 4. Determine task embedding index ---
    print(f"\n🔍 Task Embedding Logic:")
    print(f"Target: Morph='{args.morph}', Task='{args.task}', Generalization={is_generalization}")

    unique_tasks = list(dict.fromkeys(original_tasks))
    debug_print(f"Unique training tasks (preserve order): {unique_tasks}", args.debug)

    # Get task embedding module
    task_embed_module = None
    num_task_emb = None
    try:
        v_net = getattr(actor_critic, "v_net", None)
        if v_net is not None and hasattr(v_net, "task_embed"):
            task_embed_module = v_net.task_embed
            num_task_emb = int(task_embed_module.num_embeddings)
    except Exception:
        pass

    if args.task not in unique_tasks:
        print(f"❌ ERROR: Task '{args.task}' not found in training set.")
        print(f"Available unique tasks: {unique_tasks}")
        sys.exit(1)

    task_idx_unique = unique_tasks.index(args.task)
    print(f"[Eval] Task '{args.task}' => unique task index {task_idx_unique}")

    # Use task-based embedding (critical for generalization)
    if task_embed_module is not None:
        print(f"[Eval] Using model.v_net.task_embed (num={num_task_emb})")
        eval_id = int(task_idx_unique)
        mapping_used = "task-embedding (generalization-friendly)"
        try:
            emb_vec = task_embed_module.weight.data[eval_id].cpu().numpy()
            print(f"[Eval] task_embed[{eval_id}] preview (first 8): {emb_vec.reshape(-1)[:8].tolist()}")
            print(f"[Eval] task_embed[{eval_id}] L2 norm: {float(np.linalg.norm(emb_vec)):.6f}")
        except Exception as e:
            print(f"[Eval] Could not read embedding vector preview: {e}")
    else:
        print("[Eval] WARNING: model has no v_net.task_embed. Using fallback approach.")
        if is_generalization:
            print("[Eval] For generalization, using reference pair index from training")
            eval_id = render_idx  # Use reference pair index
        else:
            eval_id = extended_render_idx
        mapping_used = "pair-index (fallback)"

    if num_task_emb is not None and eval_id >= num_task_emb:
        print("❌ ERROR: eval_id >= model.task_embed.num_embeddings.")
        print("This shouldn't happen with task-based indexing. Check your training setup.")
        sys.exit(1)

    device = getattr(cfg, "DEVICE", "cpu")
    unimal_id_tensor = torch.tensor([int(eval_id)], dtype=torch.long, device=device)
    print(f"[Eval] Final unimal_id_tensor = {unimal_id_tensor.tolist()}, mapping_used = {mapping_used}")
    print(f"[Eval] Environment will render morph '{args.morph}' at extended_render_idx = {extended_render_idx}")

    # --- 5. Optional debug runs ---
    if args.debug:
        print("[Eval] Running quick 1-episode sanity check...")
        quick_video_kwargs = {"video_dir": args.save_video, "video_prefix": f"quick_{args.task}_{args.morph}"} if args.save_video else {}
        quick_env = make_vec_envs(training=False, save_video=False, morph_idx_render=extended_render_idx, **quick_video_kwargs)
        set_ob_rms(quick_env, ob_rms)
        obs = quick_env.reset()
        done = [False]
        ep_ret = 0.0
        step = 0
        while not done[0] and step < 500:
            with torch.no_grad():
                _, act, _, _, _ = agent.act(obs, unimal_ids=unimal_id_tensor, compute_val=False)
            obs, reward, done, infos = quick_env.step(act)
            ep_ret += _to_scalar_float(reward)
            step += 1
        try:
            quick_env.close()
        except Exception:
            pass
        print(f"[Eval] Quick sanity-check return: {ep_ret:.3f}")

    if args.test_all_ids:
        print("\n[Eval] Running quick ID scan across all task embeddings...")
        scan_results = {}
        max_scan = len(unique_tasks) if task_embed_module else len(original_tasks)
        for test_id in range(max_scan):
            tid = torch.tensor([test_id], dtype=torch.long, device=device)
            local_video_kwargs = {"video_dir": args.save_video, "video_prefix": f"scan_{test_id}_{args.task}_{args.morph}"} if args.save_video else {}
            local_env = make_vec_envs(training=False, save_video=False, morph_idx_render=extended_render_idx, **local_video_kwargs)
            set_ob_rms(local_env, ob_rms)
            obs = local_env.reset()
            done = [False]
            ep_ret = 0.0
            step = 0
            while not done[0] and step < 400:
                with torch.no_grad():
                    _, act, _, _, _ = agent.act(obs, unimal_ids=tid, compute_val=False)
                obs, reward, done, infos = local_env.step(act)
                ep_ret += _to_scalar_float(reward)
                step += 1
            try:
                local_env.close()
            except Exception:
                pass
            scan_results[test_id] = ep_ret
            print(f"  ID {test_id}: quick-return={ep_ret:.2f}")

    # --- 6. Main evaluation loop ---
    episode_returns = []
    episode_successes = []

    video_kwargs = {"video_dir": args.save_video, "video_prefix": f"{args.task}_{args.morph}"} if args.save_video else {}

    for i in tqdm(range(args.episodes), desc="🤖 Running Evaluation Episodes"):
        envs = make_vec_envs(training=False, save_video=bool(args.save_video), morph_idx_render=extended_render_idx, **video_kwargs)
        set_ob_rms(envs, ob_rms)

        debug_print(f"Episode {i+1}: Environment created", args.debug)

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
                print("\n❌ RuntimeError during agent.act():")
                print(f"   Exception: {e}")
                print("   unimal_id_tensor:", unimal_id_tensor)
                try:
                    envs.close()
                except Exception:
                    pass
                raise

            if args.debug and (step_count % 50 == 0):
                try:
                    if isinstance(act, torch.Tensor):
                        a_sh = tuple(act.shape)
                        a_preview = act.cpu().detach().numpy().reshape(-1)[:8].tolist()
                        debug_print(f"Step {step_count}: Action shape={a_sh}, preview={a_preview}", args.debug)
                    try:
                        val_f = float(value.item())
                        debug_print(f"Step {step_count}: Value={val_f:.4f}", args.debug)
                    except Exception:
                        pass
                except Exception:
                    pass

            obs, reward, done, infos = envs.step(act)

            rew_val = _to_scalar_float(reward)
            episode_reward = float(episode_reward) + rew_val
            step_rewards.append(rew_val)
            step_count += 1

            if args.debug and step_count % 50 == 0:
                debug_print(f"Step {step_count}: Reward: {rew_val:.4f}, Cumulative: {episode_reward:.4f}", args.debug)

        # Close env and get video path
        video_path = None
        try:
            video_path = envs.close()
        except Exception:
            pass

        is_success = False
        try:
            if isinstance(infos, (list, tuple)):
                info0 = infos[0] if len(infos) > 0 else {}
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
        print(f"  Avg Step Reward: {np.mean(step_rewards):.4f}" if step_rewards else "  Avg Step Reward: N/A")

        # Handle video saving
        if args.save_video and video_path:
            try:
                if is_success:
                    success_str = "SUCCESS"
                    reward_str = int(float(episode_reward))
                    morph_label = "GENERALIZATION" if is_generalization else "TRAINED"
                    new_video_name = f"episode_{i+1}_{morph_label}_{success_str}_reward_{reward_str}.mp4"
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

    # --- 7. Final summary ---
    avg_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0
    avg_success_rate = float(np.mean(episode_successes) * 100) if episode_successes else 0.0

    print("\n" + "="*50)
    if is_generalization:
        print(f"🚀 GENERALIZATION RESULTS: {args.morph} on {args.task}")
    else:
        print(f"📊 STANDARD EVALUATION: {args.morph} on {args.task}")
    print("="*50)
    print(f"Episodes:    {args.episodes}")
    print(f"Avg. Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {avg_success_rate:.1f}%")
    print(f"Individual episode rewards: {episode_returns}")
    print("="*50 + "\n")

    if args.debug:
        print("\n--- Debug Information ---")
        print(f"Evaluation mode: {'Generalization' if is_generalization else 'Standard'}")
        print(f"Task embedding index used: {eval_id}")
        print(f"Environment morph index: {extended_render_idx}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()

"""
python tools/zs_morph_eval.py \
  --run_dir ./artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_2008-run:v3/ \
  --checkpoint Robosuite-v0.pt \
  --morph Panda \
  --task Door \
  --controller OSC_POSE \
  --episodes 5 \
  --save_video ./test_videos/ \
  --debug
  """