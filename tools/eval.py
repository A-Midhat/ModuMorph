
import os
os.environ.setdefault("MUJOCO_GL", "glfw") 
import argparse
import json
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm

from metamorph.config import cfg, get_list_cfg
from metamorph.algos.ppo.ppo import PPO
from metamorph.algos.ppo.model import Agent
from metamorph.algos.ppo.envs import (
    make_vec_envs,
    get_ob_rms,
    set_ob_rms,
)
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.envs.vec_env.vec_normalize import VecNormalize

from tools.train_ppo import set_cfg_options


def _load_cfg(run_dir: str) -> None:
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    cfg.merge_from_file(cfg_path)



def _prep_common_cfg(model_path: str, *, deterministic: bool, num_envs: int) -> None:
    cfg.PPO.CHECKPOINT_PATH = model_path
    cfg.DETERMINISTIC = deterministic
    cfg.PPO.NUM_ENVS = num_envs
    cfg.OUT_DIR = "./eval"  
    

@torch.no_grad()
def _eval_multi(policy: Agent, env, *, episodes: int) -> Dict[str, Any]:
    """Roll episodes across a vectorised env and aggregate stats."""
    torch.manual_seed(cfg.RNG_SEED)
    R = []  # returns
    L = []    # lengths
    S = []   # success flags

    obs = env.reset()
    cur_r = np.zeros(cfg.PPO.NUM_ENVS)
    cur_l = np.zeros(cfg.PPO.NUM_ENVS, dtype=int)

    done_total = 0
    pbar = tqdm(total=episodes, desc="Evaluating", leave=False)
    while done_total < episodes:
        _, act, _, _, _ = policy.act(obs, compute_val=False)
        obs, rew, done, infos = env.step(act)
        if infos[0].get("render", False):
            env.render()
        cur_r += rew.cpu().numpy().flatten()
        cur_l += 1

        for i, d in enumerate(done):
            if d:
                R.append(cur_r[i])
                L.append(cur_l[i])
                success = infos[i].get("success", False) or infos[i].get(
                    "episode", {}
                ).get("success", False)
                S.append(bool(success))
                cur_r[i] = 0
                cur_l[i] = 0
                done_total += 1
                pbar.update(1)
    pbar.close()

    R_arr, L_arr, S_arr = map(np.asarray, (R, L, S))
    return {
        "mean_return": float(R_arr.mean()),
        "std_return": float(R_arr.std()),
        "min_return": float(R_arr.min()),
        "max_return": float(R_arr.max()),
        "median_return": float(np.median(R_arr)),
        "mean_length": float(L_arr.mean()),
        "success_rate": float(S_arr.mean()),
        "num_episodes": int(episodes),
    }



def evaluate_robosuite(
    *,
    model_path: str,
    run_dir: str,
    morph: str,
    task: str,
    controller: str,
    episodes: int,
    deterministic: bool,
    render: bool,
    video_dir: str | None,
    num_envs: int,
    video_ep: int,
    seed: int
) -> Dict[str, Any]:
    """Run evaluation for one robot / task combo."""

    _load_cfg(run_dir)
    _prep_common_cfg(model_path, deterministic=deterministic, num_envs=num_envs)

    cfg.ENV_NAME = "Robosuite-v0"
    cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = get_list_cfg(morph)
    cfg.ROBOSUITE.ENV_NAMES = get_list_cfg(task)
    cfg.ROBOSUITE.CONTROLLERS = get_list_cfg(controller)

    # viewer / off-screen flags
    if render:
        cfg.PPO.NUM_ENVS = 1
        cfg.ROBOSUITE.ENV_ARGS.update(
            {"has_renderer": True, "has_offscreen_renderer": False, "render_camera": "frontview"}
        )
    elif video_dir is not None:
        cfg.PPO.NUM_ENVS = 1    
        cfg.ROBOSUITE.ENV_ARGS.update(
             {"has_renderer": False, "has_offscreen_renderer": True}
         )

    set_cfg_options()

    ppo = PPO(print_model=False)
    policy = ppo.agent
    policy.ac.eval()

    morph_cfg = [
        {
            "env_name": task,
            "robot_names": [morph],
            "controller_names": [controller],
            "robosuite_args": dict(cfg.ROBOSUITE.ENV_ARGS),
        }
    ]

    envs = make_vec_envs(
        env_id="Robosuite-v0",
        training=False,
        norm_rew=False,
        num_env=cfg.PPO.NUM_ENVS,
        render_policy=render,
        save_video=video_dir is not None,
        video_dir=video_dir,
        morph_cfg=morph_cfg,
        seed=seed, 
        video_prefix=morph,
        # video_prefix=f"{morph_cfg[0]['robot_names'][0]}",
        video_length=cfg.PPO.VIDEO_LENGTH,    
    )

    set_ob_rms(envs, get_ob_rms(ppo.envs))

    stats = _eval_multi(policy, envs, episodes=episodes)
    if video_dir is not None:
        stats["video_path"] = os.path.join(video_dir, f"{morph}_{task}_ep{video_ep}.mp4")
    envs.close()
    return stats




def evaluate_unimal_modular(
    *,
    model_path: str,
    run_dir: str,
    agent_dir: str,
    deterministic: bool,
    terminate_on_fall: bool,
) -> Dict[str, float]:
    _load_cfg(run_dir)
    _prep_common_cfg(model_path, deterministic=deterministic, num_envs=1)

    cfg.ENV.WALKERS = []
    cfg.ENV.WALKER_DIR = agent_dir
    cfg.TERMINATE_ON_FALL = terminate_on_fall
    set_cfg_options()

    ppo = PPO(print_model=False)
    ppo.agent.ac.eval()

    test_xmls = [
        f[:-4] for f in os.listdir(os.path.join(agent_dir, "xml")) if f.endswith(".xml")
    ]
    scores: Dict[str, float] = {}
    for xml in test_xmls:
        env = make_vec_envs(
            xml_file=xml,
            training=False,
            norm_rew=False,
            render_policy=False,
        )
        set_ob_rms(env, get_ob_rms(ppo.envs))
        res = _eval_multi(ppo.agent, env, episodes=1)
        env.close()
        scores[xml] = res["mean_return"]
    return scores



def _default_ckpt(run_dir: str) -> str:
    for cand in ("checkpoint_-1.pt", "Robosuite-v0.pt", "Unimal-v0.pt"):
        path = os.path.join(run_dir, cand)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No obvious checkpoint in run_dir – pass --checkpoint explicitly."
    )


def main() -> None:
    p = argparse.ArgumentParser("Evaluate a MetaMorph / ModuMorph checkpoint")

    # generic
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--deterministic", action="store_true")

    # RoboSuite specific
    p.add_argument("--morph", help="Robot name, e.g. Panda")
    p.add_argument("--task", default="Lift")
    p.add_argument("--controller", default="JOINT_VELOCITY")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--num_envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--render", action="store_true")
    p.add_argument("--save_video", metavar="DIR")
    p.add_argument("--video_ep", type=int, default=0, help="Episode to save video of")

    # Unimal / Modular
    p.add_argument("--agent_dir", default="unimals_100/test")
    p.add_argument("--terminate_on_fall", action="store_true")

    args = p.parse_args()
    run_dir = args.run_dir.rstrip("/")

    ckpt = (
        os.path.join(run_dir, args.checkpoint)
        if args.checkpoint
        else _default_ckpt(run_dir)
    )

    _load_cfg(run_dir)
    if cfg.ENV_NAME == "Robosuite-v0":
        morph = args.morph or (
            cfg.ROBOSUITE.TRAINING_MORPHOLOGIES[0]
            if cfg.ROBOSUITE.TRAINING_MORPHOLOGIES
            else None
        )
        if morph is None:
            raise ValueError("Please supply --morph (robot name).")

        stats = evaluate_robosuite(
            model_path=ckpt,
            run_dir=run_dir,
            morph=morph,
            task=args.task,
            controller=args.controller,
            episodes=args.episodes,
            deterministic=args.deterministic,
            render=args.render,
            video_dir=args.save_video,
            video_ep=args.video_ep,
            num_envs=args.num_envs,
            seed=args.seed,
        )

        print(json.dumps(stats, indent=2))
        os.makedirs("eval", exist_ok=True)
        out_path = os.path.join("eval", f"{morph}_{args.task}_stats.json")
        with open(out_path, "w") as f:
            json.dump(stats, f, indent=2)

    else:  # Unimal / Modular
        res = evaluate_unimal_modular(
            model_path=ckpt,
            run_dir=run_dir,
            agent_dir=args.agent_dir,
            deterministic=args.deterministic,
            terminate_on_fall=args.terminate_on_fall,
        )
        print(json.dumps(res, indent=2))
        os.makedirs("eval", exist_ok=True)
        with open(os.path.join("eval", "unimal_modular_stats.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()


"""
Render one Panda-Lift episode:

    python tools/test_eval.py \
        --run_dir   output/robosuite_sr_transformer_lift_panda/1 \
        --checkpoint checkpoint_-1.pt \
        --morph Panda \
        --episodes 1 \
        --render

Record a 5-episode off-screen roll-out to videos/Panda_Lift_video.mp4:

    python tools/test_eval.py \
        --run_dir   output/robosuite_sr_transformer_lift_panda/1 \
        --checkpoint checkpoint_-1.pt \
        --morph Panda \
        --episodes 5 \
        --save_video videos
        --video_ep 4 \
        --seed 1 \

"""