import time
from collections import defaultdict
from collections import deque

import gym
import torch

import metamorph.envs  # Register envs
from metamorph.config import cfg
from metamorph.envs import CUSTOM_ENVS
from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
from metamorph.envs.vec_env.vec_normalize import VecNormalize
from metamorph.envs.wrappers.multi_env_wrapper import MultiEnvWrapper

from modular.wrappers import ModularObservationPadding, ModularActionPadding

from metamorph.envs.tasks.robosuite_task import make_env_robosuite

def make_env(env_id, seed, rank, xml_file=None, robot_name=None):
    def _thunk():
        if env_id in CUSTOM_ENVS:
            if env_id == 'Unimal-v0':
                env = gym.make(env_id, agent_name=xml_file)
            elif env_id == 'Modular-v0':
                env = gym.make(f"{xml_file}-v0")
            elif env_id == 'Robosuite-v0':
                env = make_env_robosuite(robot_name=robot_name)
        else:
            env = gym.make(env_id)
        # Note this does not change the global seeds. It creates a numpy
        # rng gen for env.
        env.seed(seed + rank)
        # Don't add wrappers above TimeLimit
        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)
        # Store the un-normalized rewards
        env = RecordEpisodeStatistics(env)
        return env

    return _thunk


# Vecenvs take as input list of functions. Dummy wrapper function for multienvs
def env_func_wrapper(env):
    def _thunk():
        return env
    return _thunk


def make_vec_envs(
    xml_file=None,
    robot_name=None, # for robosuite (eval)
    training=True,
    norm_rew=True,
    num_env=None,
    save_video=False,
    render_policy=False,
    seed=None,
):
    if not num_env:
        num_env = cfg.PPO.NUM_ENVS

    device = torch.device(cfg.DEVICE)

    if seed is None:
        seed = cfg.RNG_SEED

    # if len(cfg.ENV.WALKERS) <= 1 or render_policy or save_video:
    agent_list = cfg.ENV.WALKERS # populated from maybe_infer+walkers()
    if len(agent_list) <= 1 or render_policy or save_video:
        # ---Single Agent/Robot---
        agent_identifier = robot_name if robot_name is not None else (agent_list[0] if agent_list else None)
        if agent_identifier is None:
            raise ValueError("[make vec env] No agent identifier provided for single env.")
        
        if cfg.ENV_NAME == 'Robosuite-v0':
            xml_file_arg, robot_name_arg = None, agent_identifier
        else :
            xml_file_arg, robot_name_arg = agent_identifier, None # use xml for unimal/modular
        envs = [
            make_env(cfg.ENV_NAME, seed, idx, xml_file=xml_file_arg, robot_name=robot_name_arg)
            for idx in range(num_env)
        ]
       
    else:
        # Dummy init the actual xml_file will change on each reset
        envs = []
        if cfg.ENV_NAME == 'Unimal-v0':
            if not cfg.ENV.FIX_ENV:
                # randomly sample robots for each process as in MetaMorph
                for idx in range(num_env):
                    #_env = make_env(cfg.ENV_NAME, seed, idx, xml_file=xml_file)()
                    _env = make_env(cfg.ENV_NAME, seed, idx, xml_file=agent_list[0])() # TODO: Check this
                    envs.append(env_func_wrapper(MultiEnvWrapper(_env, idx)))
            else:
                for i, xml in enumerate(cfg.ENV.WALKERS):
                    _env = make_env(cfg.ENV_NAME, seed, 2 * i, xml_file=xml)()
                    envs.append(env_func_wrapper(_env))
                    _env = make_env(cfg.ENV_NAME, seed, 2 * i + 1, xml_file=xml)()
                    envs.append(env_func_wrapper(_env))
                # cfg.PPO.NUM_ENVS = len(envs) # swith this off for now TODO: uncomment
        elif cfg.ENV_NAME == "Robosuite-v0":
            # --- Multi-Agent(robots) ---
            num_robots = len(cfg.ROBOSUITE.ROBOTS) # OR ENV.WALKERS
            if cfg.ENV.FIX_ENV:
                envs = []
                envs_per_robot = num_env // num_robots # TODO: Need to change when scaling
                assert num_env % num_robots == 0, "NUM_ENVS must be divisible by number of robots for FIX_ENV=True"
                
                robot_idx = 0
                for i in range(num_env):
                    current_robot = cfg.ROBOSUITE.ROBOTS[robot_idx]
                    envs.append(make_env(cfg.ENV_NAME, seed, i, robot_name=current_robot)())
                    
                    env_thunk = make_env(cfg.ENV_NAME, seed, i, robot_name=current_robot)
                    envs.append(env_func_wrapper(env_thunk())) # TODO: no need for the env_func_wrapper
                    # no need just append directly 
                    # envs.append(env_thunk)
                    if i % envs_per_robot == envs_per_robot - 1:
                        robot_idx += 1
            else: 
                raise NotImplementedError("FIX_ENV=False not implemented for Robosuite")
        elif cfg.ENV_NAME == 'Modular-v0':
            for xml in cfg.ENV.WALKERS:
                _env = make_env(cfg.ENV_NAME, seed, 0, xml_file=xml)()
                envs.append(env_func_wrapper(_env))
                _env = make_env(cfg.ENV_NAME, seed, 1, xml_file=xml)()
                envs.append(env_func_wrapper(_env))
            cfg.PPO.NUM_ENVS = len(envs)

    if save_video:
        envs = DummyVecEnv([envs[0]])
    elif cfg.VECENV.TYPE == "DummyVecEnv":
        envs = DummyVecEnv(envs)
    elif cfg.VECENV.TYPE == "SubprocVecEnv":
        envs = SubprocVecEnv(envs, in_series=cfg.VECENV.IN_SERIES, context="fork")
    else:
        raise ValueError("VECENV: {} is not supported.".format(cfg.VECENV.TYPE))

    if cfg.MODEL.OBS_TO_NORM == []:
        print ('not use VecNorm')
    else:
        print ('use VecNorm')
    envs = VecNormalize(
        envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew,
        obs_to_norm=cfg.MODEL.OBS_TO_NORM
    )

    envs = VecPyTorch(envs, device)
    return envs


def make_vec_envs_zs():
    device = torch.device(cfg.DEVICE)
    seed = cfg.RNG_SEED
    norm_rew = False
    training = False

    envs = [
        make_env(cfg.ENV_NAME, seed, idx, xml_file=cfg.ENV.WALKERS[0])
        for idx in range(cfg.PPO.NUM_ENVS)
    ]
    if cfg.VECENV.TYPE == "DummyVecEnv":
        envs = DummyVecEnv(envs)
    elif cfg.VECENV.TYPE == "SubprocVecEnv":
        envs = SubprocVecEnv(envs, in_series=cfg.VECENV.IN_SERIES, context="fork")
    else:
        raise ValueError("VECENV: {} is not supported.".format(cfg.VECENV.TYPE))

    envs = VecNormalize(
        envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew,
        obs_to_norm=cfg.MODEL.OBS_TO_NORM
    )
    envs = VecPyTorch(envs, device)
    return envs


# Get a render function
def get_render_func(venv):
    if hasattr(venv, "envs"):
        return venv.envs[0].render
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_env_attr_from_venv(venv, attr_name):
    if hasattr(venv, "envs"):
        return getattr(venv.envs[0], attr_name)
    elif hasattr(venv, "venv"):
        return get_render_func(venv.venv)
    elif hasattr(venv, "env"):
        return get_render_func(venv.env)

    return None


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"):
        return get_vec_normalize(venv.venv)

    return None


def get_ob_rms(venv):
    return getattr(get_vec_normalize(venv), "ob_rms", None)


def set_ob_rms(venv, ob_rms):
    vec_norm = get_vec_normalize(venv)
    vec_norm.ob_rms = ob_rms


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info["timeout"] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = (
            time.time()
        )  # TODO: use perf_counter when gym removes Python 2 support
        self.episode_return = 0.0
        # Stores individual components of the return. For e.g. return might
        # have separate reward for speed and standing.
        self.episode_return_components = defaultdict(int)
        self.episode_length = 0
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        observation = super(RecordEpisodeStatistics, self).reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        return observation

    def step(self, action):
        observation, reward, done, info = super(
            RecordEpisodeStatistics, self
        ).step(action)
        self.episode_return += reward
        self.episode_length += 1

        #----DEBUGGINF PURPOSES---
        #print(f"RecordEpStats step: Done={done}, EpLen={self.episode_length}, Reward={reward}, infoKeys={list(info.keys())}")

        #-------------------------
        for key, value in info.items():
            if "__reward__" in key:
                self.episode_return_components[key] += value

        if done:
            #----DEBUGGINF PURPOSES---
            #print(f"RecordEpStats Done! Return={self.episode_return}, Len={self.episode_length}")
            #-------------------------
            info["episode"] = {
                "r": self.episode_return,
                "l": self.episode_length,
                "t": round(time.time() - self.t0, 6),
            }
            for key, value in self.episode_return_components.items():
                info["episode"][key] = value
                self.episode_return_components[key] = 0

            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return = 0.0
            self.episode_length = 0
        return observation, reward, done, info
