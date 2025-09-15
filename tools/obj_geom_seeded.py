##################### Generalization to New Custom Envs (basically new objects) ###########
###########################################################################################


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
# test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19
# artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_1409-run:v4
"""
Example for object generalization testing:
python tools/obj_geom_seeded.py \
  --run_dir ./artifacts/Robosuite-v0-MR-ST-MR-MT_ModuMorph_1409-run:v4 \
  --checkpoint checkpoint_400.pt \
  --morph Kinova3 \
  --task LiftScalableCube  \
  --base_task Lift \
  --controller OSC_POSE \
  --episodes 5 \
  --save_video ./test_GLENS_VIDS/ \
  --friction 1.5 0.0005 0.0001 \
  --scale 1.25 \ 
  
  --density 600 
  --debug
"""

"""
python tools/obj_geom_seeded.py \
  --run_dir ./test_artifacts/Robosuite-v0-MR-ST-MR-MT_avg_nodes_1409-run:v19 \
  --checkpoint Robosuite-v0.pt \
  --morph Sawyer \
  --task LiftScalableCube  \
  --base_task Lift \
  --controller OSC_POSE \
  --episodes 10 \
  --save_video ./test_GLENS_VIDS/ \
  --friction 0.0  \
  --damping 0.1 
  --debug
    
"""

def parse_args():
    """Parses the arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent on unseen objects and save a video.")
    parser.add_argument("--run_dir", required=True, type=str, help="Path to the training output directory")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint filename")
    parser.add_argument("--morph", required=True, type=str, help="Robot morphology to test (should be seen during training)")
    parser.add_argument("--task", required=True, type=str, help="New task/object variant to test (e.g., PickPlaceMilk)")
    parser.add_argument("--base_task", required=True, type=str, help="Base task to use embedding from (e.g., PickPlaceCan)")
    parser.add_argument("--controller", required=True, type=str, help="Controller to use (e.g., OSC_POSE)")
    parser.add_argument("--episodes", default=5, type=int, help="Number of episodes for evaluation")
    parser.add_argument("--save_video", default=None, type=str, help="Directory to save evaluation videos")
    parser.add_argument("--gripper", default=None, type=str, help="Override gripper type")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--test_all_ids", action="store_true", help="Quick scan for all unimal ids (debug)")
    parser.add_argument("--seed", default=None, type=int, help="Master seed for reproducibility")
    parser.add_argument("--scale", type=float, default=1.0, help="Uniform scaling factor for the custom object's geometry.")
    parser.add_argument("--friction", default=None, nargs="+", help="Enter modfieid friction values")
    parser.add_argument("--density", default=None, help="Add modified density")
    parser.add_argument("--damping", default=None, type=float, help="Enter modified damping value for Door handle")
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
    seed_to_use = args.seed if args.seed is not None else int(args.run_dir.split("_")[-1].split("-")[0])
    torch.manual_seed(seed_to_use) 
    np.random.seed(seed_to_use)
    print(f"[SEED-LOG] Using master seed: {seed_to_use}")
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
    # Determine which object scale to modify based on the task name
    if args.scale != 1.0:
        print(f"[SCALE-LOG] Applying scale factor {args.scale} for task '{args.task}'")
        if "Ball" in args.task:
            cfg.ROBOSUITE.OBJECTS.SPHERE_SCALE = args.scale
        elif "Cube" in args.task:
            cfg.ROBOSUITE.OBJECTS.CUBE_SCALE = args.scale
        elif "Cylinder" in args.task:
            cfg.ROBOSUITE.OBJECTS.CYLINDER_SCALE = args.scale
        elif "Rectangle" in args.task:
            cfg.ROBOSUITE.OBJECTS.RECT_SCALE = args.scale
        else:
            print(f"[SCALE-LOG] Warning: Could not determine object type from task name '{args.task}'. Scale not applied.")
    if args.friction: 
        if isinstance(args.friction, list):
           print(f"[Friction] Applying friction values: {args.friction}")
           friction_value = args.friction
        else:
           print(f"[Friction] Applying friction value: {args.friction}")
           friction_value = args.friction
        if "Ball" in args.task:
            cfg.ROBOSUITE.OBJECTS.SPHERE_FRICTION = friction_value
        elif "Cube" in args.task:
            cfg.ROBOSUITE.OBJECTS.CUBE_FRICTION = friction_value
        elif "Cylinder" in args.task:
            cfg.ROBOSUITE.OBJECTS.CYLINDER_FRICTION = friction_value
        elif "Rectangle" in args.task:
            cfg.ROBOSUITE.OBJECTS.RECT_FRICTION = friction_value
        elif "DoorScalableHandle" in args.task:
           cfg.ROBOSUITE.OBJECTS.HANDLE_FRICTION = friction_value
        else:
            print(f"[FRICTION-LOG] Warning: Could not determine object type from task name '{args.task}'. Friction not applied.")
    if args.density: 
        print(f"[Density] Applying this density values: {list(args.density)}")
        if "Ball" in args.task:
            cfg.ROBOSUITE.OBJECTS.SPHERE_DENSITY = args.density
        elif "Cube" in args.task:
            cfg.ROBOSUITE.OBJECTS.CUBE_DENSITY = args.density
        elif "Cylinder" in args.task:
            cfg.ROBOSUITE.OBJECTS.CYLINDER_DENSITY = args.density
        elif "Rectangle" in args.task:
            cfg.ROBOSUITE.OBJECTS.RECT_DENSITY = args.density
        else:
            print(f"[DENSITY-LOG] Warning: Could not determine object type from task name '{args.task}'. density not applied.")
    
    if args.damping:
        print(f"[Damping] Applying damping value: {args.damping}")
        if "DoorScalableHandle" in args.task:
            cfg.ROBOSUITE.OBJECTS.HANDLE_DAMPING = args.damping
        else:
           print(f"[DAMPING-LOG] Warning: Could not determine object type from task name '{args.task}'. Damping not applied.")
    # --- 2. Setup evaluation config for OBJECT GENERALIZATION ---
    cfg.ROBOSUITE.TASK_TYPE = "MR"
    
    # Check if this is object generalization (unseen task but seen morph)
    desired_pair = (args.morph, args.task)
    training_pairs = list(zip(original_morphs, original_tasks))
    is_object_generalization = desired_pair not in training_pairs
    
    # Verify the morph was seen during training
    if args.morph not in original_morphs:
        print(f"❌ ERROR: Morphology '{args.morph}' was not seen during training!")
        print(f"Available training morphs: {list(set(original_morphs))}")
        print("For object generalization, use a seen morphology with an unseen task variant.")
        sys.exit(1)
    
    # Verify the base_task was seen during training
    if args.base_task not in original_tasks:
        print(f"❌ ERROR: Base task '{args.base_task}' was not seen during training!")
        print(f"Available training tasks: {list(set(original_tasks))}")
        sys.exit(1)
    
    if is_object_generalization:
        print(f"🎯 OBJECT GENERALIZATION MODE: Testing unseen task '{args.task}' on seen morph '{args.morph}'")
        print(f"📚 Using task embedding from base task '{args.base_task}'")
        
        # Find a training pair with the same morph for reference  
        morph_pairs = [(m, t) for m, t in training_pairs if m == args.morph]
        if not morph_pairs:
            print(f"❌ ERROR: No training pairs found with morphology '{args.morph}'")
            sys.exit(1)
        
        # Use the first occurrence of this morph for render_idx (we'll override the task)
        reference_pair = morph_pairs[0]
        render_idx = training_pairs.index(reference_pair)
        
        print(f"🔧 Using morph '{args.morph}' (reference pair: {reference_pair} at index {render_idx})")
        print(f"📝 Will create environment with task '{args.task}' but use '{args.base_task}' embedding")
        
        # Extend the training lists to include our new task variant
        extended_morphs = original_morphs + [args.morph]
        extended_tasks = original_tasks + [args.task]
        extended_controllers = list(cfg.ROBOSUITE.CONTROLLERS) + [args.controller]
        extended_render_idx = len(original_morphs)  # Index of our new pair
        
        cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = extended_morphs
        cfg.ROBOSUITE.ENV_NAMES = extended_tasks
        cfg.ROBOSUITE.CONTROLLERS = extended_controllers
        
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
    print(f"Target: Morph='{args.morph}', Task='{args.task}', Base_task='{args.base_task}', Object_generalization={is_object_generalization}")

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

    # For object generalization, use the base_task embedding
    task_for_embedding = args.base_task if is_object_generalization else args.task
    
    if task_for_embedding not in unique_tasks:
        print(f"❌ ERROR: Task '{task_for_embedding}' not found in training set.")
        print(f"Available unique tasks: {unique_tasks}")
        sys.exit(1)

    task_idx_unique = unique_tasks.index(task_for_embedding)
    print(f"[Eval] Using embedding from task '{task_for_embedding}' => unique task index {task_idx_unique}")

    # Use task-based embedding (critical for generalization)
    if task_embed_module is not None:
        print(f"[Eval] Using model.v_net.task_embed (num={num_task_emb})")
        eval_id = int(task_idx_unique)
        mapping_used = "task-embedding (object-generalization-friendly)"
        try:
            emb_vec = task_embed_module.weight.data[eval_id].cpu().numpy()
            print(f"[Eval] task_embed[{eval_id}] preview (first 8): {emb_vec.reshape(-1)[:8].tolist()}")
            print(f"[Eval] task_embed[{eval_id}] L2 norm: {float(np.linalg.norm(emb_vec)):.6f}")
        except Exception as e:
            print(f"[Eval] Could not read embedding vector preview: {e}")
    else:
        print("[Eval] WARNING: model has no v_net.task_embed. Using fallback approach.")
        if is_object_generalization:
            # Find a pair with the base_task for fallback
            base_task_pairs = [(m, t) for m, t in training_pairs if t == args.base_task]
            if base_task_pairs:
                fallback_pair = base_task_pairs[0]
                eval_id = training_pairs.index(fallback_pair)
            else:
                eval_id = render_idx
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
    print(f"[Eval] Environment will render: morph='{args.morph}' + task='{args.task}' at extended_render_idx = {extended_render_idx}")

    # --- 5. Optional debug runs ---
    if args.debug:
        print("[Eval] Running quick 1-episode sanity check...")
        quick_video_kwargs = {"video_dir": args.save_video, "video_prefix": f"quick_{args.task}_{args.morph}"} if args.save_video else {}
        quick_env = make_vec_envs(training=False, save_video=False, morph_idx_render=extended_render_idx,seed=seed_to_use, **quick_video_kwargs)
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
            local_env = make_vec_envs(training=False, save_video=False, morph_idx_render=extended_render_idx,seed=seed_to_use, **local_video_kwargs)
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
            print(f"  ID {test_id} ({unique_tasks[test_id] if test_id < len(unique_tasks) else 'unknown'}): quick-return={ep_ret:.2f}")

    # --- 6. Main evaluation loop ---
    episode_returns = []
    episode_successes = []

    video_kwargs = {"video_dir": args.save_video, "video_prefix": f"{args.task}_{args.morph}"} if args.save_video else {}

    for i in tqdm(range(args.episodes), desc="🎯 Running Object Generalization Episodes"):
        envs = make_vec_envs(training=False, save_video=bool(args.save_video), morph_idx_render=extended_render_idx,seed=seed_to_use, **video_kwargs)
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
                    obj_label = "OBJECT_GEN" if is_object_generalization else "TRAINED"
                    new_video_name = f"episode_{i+1}_{obj_label}_{success_str}_reward_{reward_str}.mp4"
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

    print("\n" + "="*60)
    if is_object_generalization:
        print(f"🎯 OBJECT GENERALIZATION RESULTS: {args.morph} on {args.task}")
        print(f"📚 Using {args.base_task} task knowledge")
    else:
        print(f"📊 STANDARD EVALUATION: {args.morph} on {args.task}")
    print("="*60)
    print(f"Episodes:    {args.episodes}")
    print(f"Avg. Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {avg_success_rate:.1f}%")
    print(f"Individual episode rewards: {episode_returns}")
    print("="*60 + "\n")

    if args.debug:
        print("\n--- Debug Information ---")
        print(f"Evaluation mode: {'Object Generalization' if is_object_generalization else 'Standard'}")
        print(f"Task embedding used: {task_for_embedding} (index {eval_id})")
        print(f"Environment pair: {args.morph} + {args.task}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()