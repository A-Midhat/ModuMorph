
####################################
####################################
#### New GEOM SIZES ################
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
Example:
python tools/obj_geom_seeded.py \
  --run_dir ./artifacts/Robosuite-v0-MR-ST-MR-MT_object_only_JNT_3296-run:v3/ \
  --checkpoint checkpoint_300.pt \
  --morph Kinova3 \
  --task Lift \
  --controller JOINT_VELOCITY \
  --episodes 1 \
  --save_video ./test_videos/ \
  --new-geom-size 0.01 0.01 0.01 \
  --seed 1409 \
  --debug
'RethinkGripper', 'PandaGripper', 'JacoThreeFingerGripper', 'JacoThreeFingerDexterousGripper', 
'WipingGripper', 'Robotiq85Gripper', 'Robotiq140Gripper', 'RobotiqThreeFingerGripper', 
'RobotiqThreeFingerDexterousGripper', None
  """


"""
python tools/test_eval_best_one.py \
  --run_dir ./output/MR-MT_newLogic_3TASKS/ \
  --checkpoint checkpoint_200.pt \
  --morph Panda \
  --task Door \
  --controller JOINT_POSITION \
  --episodes 5 \
  --save_video ./test_mt/ \
  --debug
"""

def parse_args():
    """Parses the arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained agent and save a video.")
    parser.add_argument("--run_dir", required=True, type=str, help="Path to the training output directory")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint filename")
    parser.add_argument("--morph", required=True, type=str, help="Robot morphology to test (e.g., Panda)")
    parser.add_argument("--task", required=True, type=str, help="Robosuite task to test (e.g., Lift)")
    parser.add_argument("--controller", required=True, type=str, help="Controller to use (e.g., OSC_POSE)")
    parser.add_argument("--episodes", default=5, type=int, help="Number of episodes for evaluation")
    parser.add_argument("--save_video", default=None, type=str, help="Directory to save evaluation videos")
    parser.add_argument("--gripper", default=None, type=str, help="Override gripper type")
    parser.add_argument("--debug", action="store_true", help="Enable debug prints")
    parser.add_argument("--test_all_ids", action="store_true", help="Quick scan for all unimal ids (debug)")
    parser.add_argument("--seed", default=None, type=int, help="Master seed for reproducibility")
    # --- NEW ARG: test different object geom sizes (e.g., for Door handle or Lift cube)
    parser.add_argument(
        "--new-geom-size",
        type=float,
        nargs="+",
        default=None,
        help="Specify new raw geom_size values to test generalization (e.g., hx hy hz for box or [radius half_height] for cyl)."
    )
    
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
    # --- LOG: show that new-geom-size flag was passed at startup (even if not using --debug) ---
    if args.new_geom_size is not None:
        print(f"[GEOM-LOG] --new-geom-size was passed on CLI: {args.new_geom_size}")
    else:
        print(f"[GEOM-LOG] --new-geom-size NOT passed")
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
    # --- 2. Setup evaluation config (preserve MR lists to match training) ---
    cfg.ROBOSUITE.TASK_TYPE = "MR"
    cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = original_morphs[:]
    cfg.ROBOSUITE.ENV_NAMES = original_tasks[:]
    cfg.ROBOSUITE.CONTROLLERS = list(cfg.ROBOSUITE.CONTROLLERS)

    # CRITICAL FIX: Find the correct render_idx for the desired (morph, task) pair
    desired_pair = (args.morph, args.task)
    training_pairs = list(zip(original_morphs, original_tasks))
    
    try:
        render_idx = training_pairs.index(desired_pair)
        debug_print(f"Found exact training pair {desired_pair} at index {render_idx}", args.debug)
    except ValueError:
        print(f"❌ ERROR: The pair ({args.morph}, {args.task}) was not found in training pairs!")
        print(f"Available training pairs: {training_pairs}")
        print(f"Make sure you're testing on a morphology-task combination that was actually trained.")
        sys.exit(1)

    debug_print(f"Evaluation config - Morph (render_idx={render_idx}): {args.morph}, Task: {args.task}, Controller: {args.controller}", args.debug)
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
    if ob_rms is not None and args.debug:
        debug_print(f"Obs RMS mean: {getattr(ob_rms, 'mean', 'No mean')}", args.debug)
        debug_print(f"Obs RMS var: {getattr(ob_rms, 'var', 'No var')}", args.debug)

    print(f"\n🔍 Foundation Model Evaluation Logic:")
    print(f"Target: Morph='{args.morph}', Task='{args.task}'")
    print(f"Training pair found at index: {render_idx}")

    unique_tasks = list(dict.fromkeys(original_tasks))
    unique_morphs = list(dict.fromkeys(original_morphs))
    debug_print(f"Unique training tasks (preserve order): {unique_tasks}", args.debug)
    debug_print(f"Unique training morphs (preserve order): {unique_morphs}", args.debug)

    # list embedding modules
    emb_modules = []
    for name, module in actor_critic.named_modules():
        if isinstance(module, torch.nn.Embedding):
            emb_modules.append((name, int(module.num_embeddings), int(module.embedding_dim)))
    print("[Eval] Found Embedding modules (name, num_embeddings, emb_dim):")
    for nm, nemb, edim in emb_modules:
        print(f"   {nm:40s}  |  num={nemb:3d}  dim={edim:3d}")

    
    num_task_emb = None
    task_embed_module = None
    try:
        v_net = getattr(actor_critic, "v_net", None)
        if v_net is not None and hasattr(v_net, "task_embed"):
            task_embed_module = v_net.task_embed
            num_task_emb = int(task_embed_module.num_embeddings)
    except Exception:
        num_task_emb = None

    if args.task not in unique_tasks:
        print(f"ERROR: Task '{args.task}' not found in training set (unique tasks).")
        print(f"Available unique tasks: {unique_tasks}")
        sys.exit(1)

    task_idx_unique = unique_tasks.index(args.task)
    print(f"[Eval] unique_tasks (order): {unique_tasks}")
    print(f"[Eval] Task '{args.task}' => unique task index {task_idx_unique}")

    # Force use of task embedding index (no modulo remapping). Abort if mismatch.
    if task_embed_module is not None:
        print(f"[Eval] Using model.v_net.task_embed (num={num_task_emb}). Forcing index = {task_idx_unique} (no modulo).")
        eval_id = int(task_idx_unique)
        mapping_used = "task-embedding (forced unique index)"
        try:
            emb_vec = task_embed_module.weight.data[eval_id].cpu().numpy()
            print(f"[Eval] task_embed[{eval_id}] preview (first 8): {emb_vec.reshape(-1)[:8].tolist()}")
            print(f"[Eval] task_embed[{eval_id}] L2 norm: {float(np.linalg.norm(emb_vec)):.6f}")
        except Exception as e:
            print(f"[Eval] Could not read embedding vector preview: {e}")
    else:
        print("[Eval] WARNING: model has no v_net.task_embed. Falling back to pair-index mapping.")
        eval_id = render_idx  # Use the same index for both environment and embedding
        mapping_used = "pair-index (fallback)"
        print(f"[Eval] pair-index selected: {eval_id} (trained on {original_morphs[eval_id]}, {original_tasks[eval_id]})")

    if num_task_emb is not None and eval_id >= num_task_emb:
        print("ERROR: requested eval_id >= model.task_embed.num_embeddings.")
        sys.exit(1)

    device = getattr(cfg, "DEVICE", "cpu")
    unimal_id_tensor = torch.tensor([int(eval_id)], dtype=torch.long, device=device)
    print(f"[Eval] Final unimal_id_tensor = {unimal_id_tensor.tolist()}, mapping_used = {mapping_used}")
    print(f"[Eval] Environment will be created at render_idx = {render_idx} for pair {training_pairs[render_idx]}")

    if args.debug:
        print("[Eval] Running quick 1-episode sanity check with forced index...")
        quick_video_kwargs = {"video_dir": args.save_video, "video_prefix": f"quick_{args.task}_{args.morph}"} if args.save_video else {}
        quick_env = make_vec_envs(training=False, save_video=False, morph_idx_render=render_idx, seed=seed_to_use, **quick_video_kwargs)
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
        print(f"[Eval] Quick sanity-check return: {ep_ret:.3f} (should reflect correct task behaviour)")

    if args.test_all_ids:
        print("\n[Eval] Running quick ID scan across all pair ids (first occurrence mapping size).")
        scan_results = {}
        max_scan = len(original_tasks)
        for test_id in range(max_scan):
            tid = torch.tensor([test_id], dtype=torch.long, device=device)
            local_video_kwargs = {"video_dir": args.save_video, "video_prefix": f"scan_{test_id}_{args.task}_{args.morph}"} if args.save_video else {}
            local_env = make_vec_envs(training=False, save_video=False, morph_idx_render=render_idx, seed=seed_to_use, **local_video_kwargs)
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
        print("[Eval] ID scan done. Results:", scan_results)

    episode_returns = []
    episode_successes = []

    # build video kwargs for main loop (pass only if saving)
    video_kwargs = {"video_dir": args.save_video, "video_prefix": f"{args.task}_{args.morph}"} if args.save_video else {}

    
    # Minimal verified mapping of env task -> geom name (used when --new-geom-size passed)
    correct_geom_map = {
        "Lift": "cube_g0",
        "Door": "Door_handle",
        "PickPlaceCan": "Can_g0",
        "PickPlaceMilk": "Milk_g0",
        "PickPlaceBread": "Bread_g0",
        "PickPlaceCereal": "Cereal_g0",
    }
    for i in tqdm(range(args.episodes), desc="🤖 Running Evaluation Episodes"):
        envs = make_vec_envs(training=False, save_video=bool(args.save_video), morph_idx_render=render_idx, seed=seed_to_use, **video_kwargs)
        set_ob_rms(envs, ob_rms)
        
        # --- NEW: Object geometry modification (minimal, safe) ---
        # This only runs if --new-geom-size is provided. We try to apply the change,
        # but we do NOT abort the whole evaluation if it fails (we print a warning).
        if args.new_geom_size:
            target_geom_name = correct_geom_map.get(args.task)
            if not target_geom_name:
                print(f"Warning: Task '{args.task}' not in geom_map. Cannot modify object size.")
            else:
                try:
                    # Access underlying MuJoCo model (assumes NUM_ENVS=1)
                    sim_env = envs.venv.envs[0]
                    model = sim_env.sim.model

                    # Attempt to get geom id (this matches the obj_geom_eval.py approach)
                                        # Attempt to get geom id (this matches the obj_geom_eval.py approach)
                    geom_id = model.geom(target_geom_name).id
                    ###############################################################
                    ##################### DEBUGGING ###############################
                    ###############################################################

                    # Add this after getting geom_id
                    geom_type = model.geom_type[geom_id]
                    geom_type_names = {0: 'plane', 1: 'hfield', 2: 'sphere', 3: 'capsule', 
                                    5: 'cylinder', 6: 'box', 7: 'mesh'}
                    print(f"Geom '{target_geom_name}' is type: {geom_type_names.get(geom_type, 'unknown')}")

                    # Also check if it has a mesh
                    if hasattr(model, 'geom_meshid') and model.geom_meshid[geom_id] >= 0:
                        print("This geometry uses a mesh - size scaling might not work as expected")
                    else:
                        print("This is a primitive geometry - size scaling should work fine")
                    ###############################################################
                    ###############################################################
                    ###############################################################
                    # --- Pre-change logging ---
                    original_size = model.geom_size[geom_id].copy()
                    if i == 0:
                        print("\n--- 🦾 Object Geometry Modification (attempt) ---")
                        print(f"Task: '{args.task}', Target Geom: '{target_geom_name}' (ID: {geom_id})")
                        print(f"  - original model.geom_size[{geom_id}]: {original_size.tolist()}")
                        print(f"  - requested --new-geom-size: {args.new_geom_size}")

                    # Apply the new size (only overwrite first len(new_size) components)
                    new_size = np.array(args.new_geom_size, dtype=float)
                    model.geom_size[geom_id][: len(new_size)] = new_size

                    # Ensure MuJoCo recomputes derived quantities
                    try:
                        sim_env.sim.forward()
                    except Exception:
                        # forward() is best-effort; continue even if it fails
                        pass

                    # --- Immediate post-change check ---
                    after_size = model.geom_size[geom_id].copy()
                    if i == 0:
                        print(f"  - after assignment, model.geom_size[{geom_id}]: {after_size.tolist()}")
                    # verify the first components were set correctly
                    try:
                        compare_ok = np.allclose(after_size[: len(new_size)], new_size, atol=1e-6, rtol=1e-6)
                    except Exception:
                        compare_ok = False

                    if compare_ok:
                        print(f"[GEOM-LOG] Immediate verification: PASS (first {len(new_size)} elements match requested values)")
                    else:
                        print(f"[GEOM-LOG] Immediate verification: FAIL - requested {new_size.tolist()} ; got {after_size[:len(new_size)].tolist()}")
                        print("  -> Will continue; a later re-check is performed after env.reset() in case the wrapper re-initialized model.")

                    if i == 0:
                        print("--------------------------------------\n")         
                except Exception as e:
                    print(f"\n❌ Warning: failed to modify object geometry (will continue). Error: {e}\n")
        # --- END: object geometry modification ---
        debug_print(f"Episode {i+1}: Environment created", args.debug)
        debug_print(f"Environment action space: {envs.action_space}", args.debug)
        debug_print(f"Environment observation space: {envs.observation_space}", args.debug)

        obs = envs.reset()
        # --- POST-RESET verification: confirm the change survived env.reset() ---
        if args.new_geom_size:
            try:
                sim_env = envs.venv.envs[0]
                model = sim_env.sim.model
                geom_id = model.geom(target_geom_name).id
                post_reset_size = model.geom_size[geom_id].copy()
                # Print a concise verification line
                print(f"[GEOM-LOG] After env.reset(): model.geom_size[{geom_id}] = {post_reset_size.tolist()}")
                # If mismatch, attempt to reapply and report
                if not np.allclose(post_reset_size[: len(new_size)], new_size, atol=1e-6, rtol=1e-6):
                    print(f"[GEOM-LOG] MISMATCH after reset (expected {new_size.tolist()}). Reapplying assignment and calling sim.forward()...")
                    model.geom_size[geom_id][: len(new_size)] = new_size
                    try:
                        sim_env.sim.forward()
                    except Exception:
                        pass
                    post_fix_size = model.geom_size[geom_id].copy()
                    print(f"[GEOM-LOG] After reapply: model.geom_size[{geom_id}] = {post_fix_size.tolist()}")
                    if np.allclose(post_fix_size[: len(new_size)], new_size, atol=1e-6, rtol=1e-6):
                        print(f"[GEOM-LOG] Reapply verification: PASS")
                    else:
                        print(f"[GEOM-LOG] Reapply verification: FAIL - geometry still does not match requested values.")
                        print("  -> This indicates the environment likely overwrites geom_size on reset; consider modifying the environment factory or the XML before env creation.")
            except Exception as e:
                print(f"[GEOM-LOG] post-reset verification failed with exception: {e}")

        if isinstance(obs, dict):
            debug_print(f"Episode {i+1}: Observation is dict with keys: {list(obs.keys())}", args.debug)
            for key, value in obs.items():
                try:
                    if hasattr(value, 'shape'):
                        debug_print(f"  {key}: shape {value.shape}, mean: {value.mean():.4f}, std: {value.std():.4f}", args.debug)
                except Exception:
                    debug_print(f"  {key}: Couldn't compute stats", args.debug)
        else:
            try:
                debug_print(f"Episode {i+1}: Initial observation shape: {obs.shape}", args.debug)
                debug_print(f"Episode {i+1}: Initial observation stats - mean: {obs.mean():.4f}, std: {obs.std():.4f}", args.debug)
            except Exception:
                debug_print("Could not compute obs stats", args.debug)

        done = [False]
        episode_reward = 0.0
        step_count = 0
        step_rewards = []

        while not done[0]:
            try:
                with torch.no_grad():
                    _, act, _, value, _ = agent.act(obs, unimal_ids=unimal_id_tensor, compute_val=True)
            except RuntimeError as e:
                print("\nRuntimeError during agent.act():")
                print(f"   Exception: {e}")
                print("   unimal_id_tensor:", unimal_id_tensor)
                try:
                    envs.close()
                except Exception:
                    pass
                raise

            # Safe action/value preview only every 50 steps (keeps logs readable)
            try:
                if args.debug and (step_count % 50 == 0):
                    if isinstance(act, torch.Tensor):
                        a_sh = tuple(act.shape)
                        a_preview = act.cpu().detach().numpy().reshape(-1)[:8].tolist()
                        debug_print(f"Step {step_count}: Action shape={a_sh}, preview={a_preview}", args.debug)
                    else:
                        debug_print(f"Step {step_count}: Action type={type(act)}", args.debug)
                    try:
                        val_f = float(value.item())
                        debug_print(f"Step {step_count}: Value={val_f:.4f}", args.debug)
                    except Exception:
                        debug_print(f"Step {step_count}: Value type={type(value)}", args.debug)
            except Exception:
                debug_print(f"Step {step_count}: couldn't print action/value (preview failed)", args.debug)

            obs, reward, done, infos = envs.step(act)

            rew_val = _to_scalar_float(reward)
            episode_reward = float(episode_reward) + rew_val
            step_rewards.append(rew_val)
            step_count += 1

            if args.debug and step_count % 50 == 0:
                debug_print(f"Step {step_count}: Reward: {rew_val:.4f}, Cumulative: {episode_reward:.4f}", args.debug)

        # close envs and retrieve video path if recorder used
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

        if args.debug and step_rewards:
            print(f"  Step rewards distribution: min={min(step_rewards):.4f}, max={max(step_rewards):.4f}")

        if args.save_video and video_path:
            try:
                if is_success:
                    success_str = "SUCCESS"
                    reward_str = int(float(episode_reward))
                    new_video_name = f"episode_{i+1}_{success_str}_reward_{reward_str}.mp4"
                    new_video_path = os.path.join(args.save_video, new_video_name)
                    os.rename(video_path, new_video_path)
                    tqdm.write(f"Success! Saved video to {new_video_name}")
                else:
                    try:
                        os.remove(video_path)
                        tqdm.write("Failure. Discarding video.")
                    except Exception:
                        tqdm.write("Failure. Could not remove video (maybe None or already removed).")
            except Exception as e:
                tqdm.write(f"Could not handle video file: {e}")

    # --- 6. Summary ---
    avg_reward = float(np.mean(episode_returns)) if episode_returns else 0.0
    std_reward = float(np.std(episode_returns)) if episode_returns else 0.0
    avg_success_rate = float(np.mean(episode_successes) * 100) if episode_successes else 0.0

    print("\n--- Evaluation Statistics ---")
    if args.new_geom_size:
        print(f"NOTE: Ran with MODIFIED object size: {args.new_geom_size}")
    print(f"Episodes:    {args.episodes}")
    print(f"Avg. Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {avg_success_rate:.1f}%")
    print(f"Individual episode rewards: {episode_returns}")
    print("-----------------------------\n")

    if args.debug:
        print("\n--- Debug Information ---")
        print(f"Config used for evaluation:")
        print(f"  TASK_TYPE: {cfg.ROBOSUITE.TASK_TYPE}")
        print(f"  TRAINING_MORPHOLOGIES: {cfg.ROBOSUITE.TRAINING_MORPHOLOGIES}")
        print(f"  ENV_NAMES: {cfg.ROBOSUITE.ENV_NAMES}")
        print(f"  CONTROLLERS: {cfg.ROBOSUITE.CONTROLLERS}")
        print(f"  NUM_ENVS: {cfg.PPO.NUM_ENVS}")
        print("---------------------------\n")

if __name__ == "__main__":
    main()


"""
# Lift
--- Evaluation Statistics ---
Episodes:    50
Avg. Reward: 255.36 ± 111.54
Success Rate: 10.0%

# Door
--- Evaluation Statistics ---
Episodes:    50
Avg. Reward: 204.19 ± 16.44
Success Rate: 0.0%
"""