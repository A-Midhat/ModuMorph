import time
from collections import defaultdict
from collections import deque

import gym
import torch

import metamorph.envs  # Register envs
from metamorph.config import cfg, get_list_cfg
from metamorph.envs import CUSTOM_ENVS
from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
from metamorph.envs.vec_env.vec_normalize import VecNormalize
# from metamorph.envs.wrappers.multi_env_wrapper import MultiEnvWrapper

# Robosuite wrappers
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteMLPFlattener
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteNodeCentricObservation
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteNodeCentricAction
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteSampleWrapper
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper
# from modular.wrappers import ModularObservationPadding, ModularActionPadding

# TODO: move it to task.py for better SoC approach
def _build_inner_robosuite_stack(config_dict):
    """
    Builds the Robosuite environment stack for 
    a specific morphology configuration dictionary.

    Args:
        config_dict (dict): Dictionary defining the morphology configuration
                            (e.g., {'env_name': 'Lift', 'robot_names': ['Panda'], ...}).

    Returns:
        gym.Env: The top wrapper of the built stack (e.g., SelectKeysWrapper).
    """
    # Pass all necessary arguments from the config_dict
    base_env = RobosuiteEnvWrapper(
        env_name=config_dict['env_name'],
        robot_names=config_dict['robot_names'],
        controller_names=config_dict['controller_names'],
        horizon=config_dict.get('horizon', cfg.ROBOSUITE.ENV_ARGS.get('horizon', 500)), 
        robosuite_args=config_dict.get('robosuite_args', dict(cfg.ROBOSUITE.ENV_ARGS))
    )
    
    if cfg.MODEL.TYPE == "mlp":
        wrapped_env = RobosuiteMLPFlattener(base_env)
    elif cfg.MODEL.TYPE == "transformer":
        wrapped_env = RobosuiteNodeCentricObservation(base_env)
        wrapped_env = RobosuiteNodeCentricAction(wrapped_env)
    else:
        raise ValueError(f"Unsupported MODEL.TYPE: {cfg.MODEL.TYPE}")

    keys_to_keep = cfg.ENV.get('KEYS_TO_KEEP', list(cfg.MODEL.get("OBS_TYPES", []))) 
    #print("[keys_to_keep]", keys_to_keep)
    # This should be hanled in the ac foraward method.
    # if cfg.MODEL.TYPE == "transformer":
    #     # Only needed by transformer
    #     keys_to_keep.extend(['traversals', 'SWAT_RE'])

    external_keys_cfg = cfg.ROBOSUITE.get("EXTERO_KEYS", [])
    # keys_to_keep.extend(external_keys_cfg)
    keys_to_keep = list(set(keys_to_keep)) 

    avail_keys = wrapped_env.observation_space.spaces.keys()
    #print("[avail_keys]", avail_keys)
    final_keys_to_keep = [k for k in keys_to_keep if k ]
    wrapped_env = SelectKeysWrapper(wrapped_env, keys_to_keep=final_keys_to_keep)


    return wrapped_env


def make_env(env_id, seed, rank, **kwargs):
    """
    **kwargs is used to pass in the morphology cfg and the xml file
    """
    def _thunk():
        if env_id == "Robosuite-v0":
            all_morph_cfg = kwargs['morph_cfg']
            if all_morph_cfg is None:
                raise ValueError("morph_cfg must be provided for Robosuite-v0")
    
            env = RobosuiteSampleWrapper(
                env = None,
                all_morphology_configs=all_morph_cfg,
                inner_stack_builder_fn=_build_inner_robosuite_stack,
                worker_rank=rank,
                num_workers=cfg.PPO.NUM_ENVS,
            )

            # env = RobosuiteSampleWrapper(
            #     env = None, 
            #     all_morphology_configs=all_morph_cfg,
            #     inner_stack_builder_fn=_build_inner_robosuite_stack,
            # )
            # env.seed(seed + rank)

            # return env

        elif env_id in CUSTOM_ENVS[:-1]: # remove Robosuite-v0
            xml_file = kwargs['xml_file']
            if env_id == 'Unimal-v0':
                env = gym.make(env_id, agent_name=xml_file)
            elif env_id == 'Modular-v0':
                env = gym.make(f"{xml_file}-v0")

        else: 
            env = gym.make(env_id)
        
        env.seed(seed + rank)

        if str(env.__class__.__name__).find("TimeLimit") >= 0 or hasattr(env, "_max_episode_steps"):# to account for robosuite
            env = TimeLimitMask(env)
        
        env = RecordEpisodeStatistics(env)
        
        return env

    return _thunk
    


# Vecenvs take as input list of functions. Dummy wrapper function for multienvs
def env_func_wrapper(env):
    def _thunk():
        return env
    return _thunk


# def make_vec_envs(
#     xml_file=None,
#     training=True,
#     norm_rew=True,
#     num_env=None,
#     save_video=False,
#     render_policy=False,
#     seed=None,
# ):
def make_vec_envs(
    env_id=None,
    seed=None,
    # rank=0,
    training=True,
    norm_rew=True,
    num_env=None,
    save_video=False,
    render_policy=False,
    **kwargs
):
    device = torch.device(cfg.DEVICE)
    
    if env_id is None:
        env_id = cfg.ENV_NAME

    if num_env is None:
        num_env = cfg.PPO.NUM_ENVS

    if seed is None:
        seed = cfg.RNG_SEED

    # envs = []
    if env_id == "Robosuite-v0":
        all_morph_cfg = []
        morphologies = cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES', [])
        env_names = cfg.ROBOSUITE.get('ENV_NAMES', [])
        controllers = cfg.ROBOSUITE.get('CONTROLLERS', [])
        robosuite_args = dict(cfg.ROBOSUITE.ENV_ARGS) 

        num_cfgs = len(morphologies)
        if not num_cfgs:
            # won't happen becasue we 'infer' during train_ppo if the list is empty
            raise ValueError("[VecEnv] No morphologies specified in config")
        # Already rasise error in base env wrapper
        # if len(env_names) != num_cfgs or len(controllers) != num_cfgs:
        #     raise ValueError("[VecEnv] Number of morphologies, env_names, and controllers must be the same")

        for i in range(num_cfgs):
            cfg_dict = {
                "env_name": env_names[i],
                "robot_names": get_list_cfg(morphologies[i]),
                "controller_names": get_list_cfg(controllers[i]),
                "robosuite_args": robosuite_args,
            }

            all_morph_cfg.append(cfg_dict) 
        if save_video or render_policy:
            # we pass 'morph_idx_render` as kwargs during eval 
            morph_idx = kwargs.get('morph_idx_render', 0) # default first idx 
            
            # envs = [
            #     lambda rank: make_env(
            #         env_id, 
            #         seed, 
            #         rank, 
            #         all_morph_cfg=all_morph_cfg,
            #         sample_idx_in_seq=0,
            #         **kwargs
            #     )
            # ]
            envs = [
            make_env(
                env_id,
                seed,
                rank,
                morph_cfg=all_morph_cfg,
                **kwargs
                )
            ]
            # Force only one env for video/render
            num_envs_run = 1 
        else: 
            envs = [
            make_env(
                env_id,
                seed,
                rank,
                morph_cfg=all_morph_cfg,
                **kwargs
                )
                for rank in range(num_env)
            ]

    elif env_id in CUSTOM_ENVS[:-1]: # remove Robosuite-v0
         
        if len(cfg.ENV.WALKERS) <= 1 or render_policy or save_video:
            if xml_file is None or len(cfg.ENV.WALKERS) == 1:
                xml_file = cfg.ENV.WALKERS[0]
            envs = [
                make_env(cfg.ENV_NAME, seed, idx, xml_file=xml_file)
                for idx in range(num_env)
            ]
        else:
            # Dummy init the actual xml_file will change on each reset
            xml_file = cfg.ENV.WALKERS[0]
            envs = []
            if cfg.ENV_NAME == 'Unimal-v0':
                if not cfg.ENV.FIX_ENV:
                    # randomly sample robots for each process as in MetaMorph
                    for idx in range(num_env):
                        _env = make_env(cfg.ENV_NAME, seed, idx, xml_file=xml_file)()
                        envs.append(env_func_wrapper(MultiEnvWrapper(_env, idx)))
                else:
                    for i, xml in enumerate(cfg.ENV.WALKERS):
                        _env = make_env(cfg.ENV_NAME, seed, 2 * i, xml_file=xml)()
                        envs.append(env_func_wrapper(_env))
                        _env = make_env(cfg.ENV_NAME, seed, 2 * i + 1, xml_file=xml)()
                        envs.append(env_func_wrapper(_env))
                    cfg.PPO.NUM_ENVS = len(envs)
            elif cfg.ENV_NAME == 'Modular-v0':
                for xml in cfg.ENV.WALKERS:
                    _env = make_env(cfg.ENV_NAME, seed, 0, xml_file=xml)()
                    envs.append(env_func_wrapper(_env))
                    _env = make_env(cfg.ENV_NAME, seed, 1, xml_file=xml)()
                    envs.append(env_func_wrapper(_env))
                cfg.PPO.NUM_ENVS = len(envs)
    num_envs_run = num_env 
    use_dummy_vec_env = save_video or render_policy or num_envs_run == 1 or cfg.VECENV.get("TYPE", "SubprocVecEnv") == "DummyVecEnv"
    if use_dummy_vec_env:
        print(f"[VecEnv] Using DummyVecEnv (n={num_envs_run})")# Should normally be 1
        envs = DummyVecEnv(envs)
    else: 
        print(f"[VecEnv] Using SubprocVecEnv (N={num_envs_run}, InSeries={cfg.VECENV.get('IN_SERIES', 1)})")
        envs = SubprocVecEnv(envs, in_series=cfg.VECENV.get("IN_SERIES", 1), context="fork")
    

    if cfg.MODEL.OBS_TO_NORM == []:
        print ('not use VecNorm')
    else:
        print ('use VecNorm')
        envs = VecNormalize(
            envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew,
            obs_to_norm=cfg.MODEL.OBS_TO_NORM
        )
        print(f"[VecEnv] Applied VecNormalize for keys: {cfg.MODEL.OBS_TO_NORM}, training={training}, norm_rew={norm_rew}")

    # Convert NumPy observations/actions from VecEnv to PyTorch tensors
    envs = VecPyTorch(envs, device)
    
    # Wrap with video recorder if save_video is True
    if save_video:
        # Requires file_prefix and save_dir in kwargs
        video_dir = kwargs.get('video_dir')
        video_prefix = kwargs.get('video_prefix', 'video')
        video_length = kwargs.get('video_length', cfg.PPO.VIDEO_LENGTH)

        if video_dir is None:
             print("Warning: save_video is True but 'video_dir' not provided in kwargs. Skipping video recording.")
        else:
             print(f"[make_vec_envs] Applied VecVideoRecorder. Saving to {video_dir} with prefix {video_prefix}")
             envs = VecVideoRecorder(
                 envs,
                 video_dir,
                 record_video_trigger=lambda x: x == 0, # Record the first episode
                 video_length=video_length,
                 file_prefix=video_prefix,
             )


    return envs # Return the final wrapped VecEnv


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
        for key, value in info.items():
            if "__reward__" in key:
                self.episode_return_components[key] += value

        if done:
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
