import time
from collections import defaultdict
from collections import deque

import gym
import torch

import os # for testing 

import metamorph.envs  # Register envs
from metamorph.config import cfg
from metamorph.envs.tasks.task import make_base_env 
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper

from metamorph.envs import CUSTOM_ENVS
from metamorph.envs.vec_env.dummy_vec_env import DummyVecEnv
from metamorph.envs.vec_env.pytorch_vec_env import VecPyTorch
from metamorph.envs.vec_env.subproc_vec_env import SubprocVecEnv
from metamorph.envs.vec_env.vec_normalize import VecNormalize
# from metamorph.envs.wrappers.multi_env_wrapper import MultiEnvWrapper

# from modular.wrappers import ModularObservationPadding, ModularActionPadding

# Robosuite Wrappers
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteNodeCentricObservation
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteNodeCentricAction
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteMLPFlattener 


# def make_env(env_id, seed, rank, xml_file=None, robot_name=None):
def make_env(env_id, seed, rank, morphology_spec=None, env_name_spec=None, controller_spec=None):
    """
    Creates a single environment instance with appropriate wrappers.
    Handles dynamic creation for Robosuite based on specs.
    Handles Unimal/Modular based on morphology_spec (agent name).
    """
    def _thunk():
        # --- Debugging --- TODO: Remove this
        env = None 
        print(f"[worker {rank}] Creaating env: {env_id}, MorphSpec: {morphology_spec}", flush=True)
        # -----------------
        if env_id in CUSTOM_ENVS:
            if env_id in ["Unimall-v0", "Modular-v0"] and morphology_spec is None:
                raise  ValueError(f"Morphology spec (agent_name/xml_path) must be provided for Unimal/Modular. worker[{rank}]")
            if env_id == 'Unimal-v0':
                xml_file = morphology_spec
                env = gym.make(env_id, agent_name=xml_file)
            elif env_id == 'Modular-v0':
                xml_file = morphology_spec
                env = gym.make(f"{xml_file}-v0")
            elif env_id == 'Robosuite-v0':
                env = make_base_env(
                    env_id=env_id,
                    morphology_spec=morphology_spec,
                    env_name_spec=env_name_spec,
                    controller_spec=controller_spec,
                )
        else:
            try:
                env = gym.make(env_id)
            except: 
                raise ValueError(f"Unknown env_id: {env_id} for worker[{rank}]")
        #TODO: add debugging for env = None 

        # Note this does not change the global seeds. It creates a numpy
        # rng gen for env.
        env.seed(seed + rank)
        # Apply to all env
        # Don't add wrappers above TimeLimit
        if str(env.__class__.__name__).find("TimeLimit") >= 0:
            env = TimeLimitMask(env)
        
        # apply the following wrappers only for robosuite env
        if cfg.ENV_NAME == "Robosuite-v0":
            if cfg.MODEL.TYPE == "mlp":
                print(f"Applying [RobosuiteMLPFlattener] wrapper, worker[{rank}]")
                # keys_to_keep = ["proprioceptive", "obs_padding_mask", "act_padding_mask", "edges", "context"] # we will use only proprioceptive, the others are dummy for MLPModel
                keys_to_keep = cfg.ENV.get('KEYS_TO_KEEP', ["proprioceptive", "obs_padding_mask", "act_padding_mask", "edges", "context"]) # we will use only proprioceptive, the others are dummy for MLPModel
                # TODO:CHECL THE MODEL.OBS_TYPE IF NEEDED  
                env = RobosuiteMLPFlattener(env)
                env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
            elif cfg.MODEL.TYPE == "transformer":
                print(f"Applying [RobosuiteNodeCentricObservation/Action] wrappers, worker[{rank}]")
                # keys_ = ["proprioceptive", "context", "edges", \
                #             "obs_padding_mask", "act_padding_mask", \
                                
                #                 "object-state", 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
                keys_ = cfg.ENV.get('KEYS_TO_KEEP',list(cfg.MODEL.get("OBS_TYPES", [])))
                keys_.extend(cfg.ROBOSUITE.get("EXTERO_KEYS", [])) # aka object-state mostly 

                env = RobosuiteNodeCentricObservation(env)
                env = RobosuiteNodeCentricAction(env)
                final_obs_spcae_keys = env.observation_space.spaces.keys()
                keys_to_keep = [k for k in keys_ if k in final_obs_spcae_keys]
                # cfg.ENV.KEYS_TO_KEEP = keys_to_keep
                env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
        # for extra wrappers TODO: if needed
        # for wrapper_name in cfg.ENV.WRAPPERS:
        #             env = globals()[wrapper_name](env) 
        #         env = RobosuiteNodeCentricObservation(base_env)
        #         env = RobosuiteNodeCentricAction(env)
        # Store the un-normalized rewards
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
#     robot_name=None, # for robosuite (eval)
#     training=True,
#     norm_rew=True,
#     num_env=None,
#     save_video=False,
#     render_policy=False,
#     seed=None,
# ):
def make_vec_envs(
    training=True,
    norm_rew=True,
    num_env=None,
    seed=None,
    save_video=False,    
    render_policy=False, 
    xml_file=None,
    robot_name=None,
):
    if num_env is None: num_env = cfg.PPO.NUM_ENVS
    if seed is None: seed = cfg.RNG_SEED
    device = torch.device(cfg.DEVICE)

    env_fns = []
    env_id_to_use = cfg.ENV_NAME

    if not training or render_policy or save_video:
        # --- Evaluation or Single Morphology Case ---
        print(f"[make_vec_envs] Creating {num_env} workers for single morphology (eval/render/video).")
        morph_spec = None; env_name_spec = None; controller_spec = None; xml_file_spec = None

        if env_id_to_use == "Robosuite-v0":
            if robot_name is None: raise ValueError("robot_name required for Robosuite eval/render.")
            morph_spec = get_list_cfg(robot_name)
            try: # Find corresponding config entry
                morph_list = cfg.ROBOSUITE.TRAINING_MORPHOLOGIES
                config_idx = next(i for i, m in enumerate(morph_list) if get_list_cfg(m) == morph_spec)
                env_name_spec = cfg.ROBOSUITE.ENV_NAMES[config_idx]
                controller_spec = cfg.ROBOSUITE.CONTROLLERS[config_idx]
            except (StopIteration, IndexError, TypeError, AttributeError): 
                print(f"Warning: Eval - Could not find matching config for '{robot_name}'. Using first config entry.")
                morph_spec = cfg.ROBOSUITE.TRAINING_MORPHOLOGIES[0]
                env_name_spec = cfg.ROBOSUITE.ENV_NAMES[0]
                controller_spec = cfg.ROBOSUITE.CONTROLLERS[0]
        elif env_id_to_use in ["Unimal-v0", "Modular-v0"]:
            if xml_file is None: raise ValueError("xml_file required for Unimal/Modular eval/render.")
            xml_file_spec = xml_file 

        # All workers get the same configuration for eval/render
        for i in range(num_env):
            env_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=i,
                                    # Pass only relevant args
                                    xml_file=xml_file_spec, 
                                    robot_name=robot_name,  
                                    morphology_spec=morph_spec, 
                                    env_name_spec=env_name_spec, 
                                    controller_spec=controller_spec))

    else:
        if env_id_to_use == "Robosuite-v0":
            morphologies = cfg.ROBOSUITE.get('TRAINING_MORPHOLOGIES', [])
            env_names = cfg.ROBOSUITE.get('ENV_NAMES', [])
            controllers = cfg.ROBOSUITE.get('CONTROLLERS', [])
            num_configs = len(morphologies)
            if num_configs == 0: raise ValueError("No TRAINING_MORPHOLOGIES defined.")
            if len(env_names)!=num_configs or len(controllers)!=num_configs: raise ValueError("Config list mismatch")

            print(f"[make_vec_envs] Assigning {num_env} Robosuite workers (training), cycling {num_configs} configs.")
            for i in range(num_env):
                config_idx = i % num_configs
                # Create the thunk using make_env, passing the specific config
                env_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=i,
                                        morphology_spec=morphologies[config_idx],
                                        env_name_spec=env_names[config_idx],
                                        controller_spec=controllers[config_idx]))
        elif env_id_to_use in ["Unimal-v0", "Modular-v0"]:
            walkers = cfg.ENV.get('WALKERS', [])
            if not walkers: raise ValueError(f"No WALKERS for {env_id_to_use} train.")
            num_configs = len(walkers)
            print(f"[make_vec_envs] Assigning {num_env} {env_id_to_use} workers (training), cycling {num_configs} walkers.")

           
            if cfg.ENV.get('FIX_ENV', False):
                 print("  Using FIX_ENV for Unimal/Modular.")
                 assigned_fns = []
                 for walker_idx, walker_name in enumerate(walkers):
                      worker_idx_1 = walker_idx * 2; worker_idx_2 = worker_idx_1 + 1
                      if worker_idx_1 < num_env: assigned_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=worker_idx_1, xml_file=walker_name))
                      if worker_idx_2 < num_env: assigned_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=worker_idx_2, xml_file=walker_name))
                 env_fns = assigned_fns[:num_env]
            elif cfg.ENV.get('TASK_SAMPLING') == "uniform_random_strategy":
                 print("  Using MultiEnvWrapper for Unimal/Modular dynamic sampling.")
                 base_env_fns = [make_env(env_id=env_id_to_use, seed=seed, rank=i, xml_file=walkers[0]) for i in range(num_env)]
                 env_fns = [env_func_wrapper(MultiEnvWrapper(base_env_fn(), i)) for i, base_env_fn in enumerate(base_env_fns)]
            else:
                 print("  Using fixed worker assignment (cycling) for Unimal/Modular.")
                 for i in range(num_env):
                     config_idx = i % num_configs
                     
                     env_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=i, xml_file=walkers[config_idx]))
            
        else: 
            print(f"[make_vec_envs] Assigning {num_env} standard Gym workers (training) for {env_id_to_use}.")
            for i in range(num_env): env_fns.append(make_env(env_id=env_id_to_use, seed=seed, rank=i))


    vecenv_type = cfg.VECENV.get("TYPE", "SubprocVecEnv")
    use_dummy = save_video or render_policy or num_env == 1 or vecenv_type == "DummyVecEnv"

    if use_dummy:
        print(f"[make_vec_envs] Using DummyVecEnv (N={1 if save_video or render_policy else num_env}). Reason: {'Video/Render/N=1' if save_video or render_policy or num_env==1 else 'Config'}")
        envs = DummyVecEnv([env_fns[0]] if save_video or render_policy else env_fns)
    elif vecenv_type == "SubprocVecEnv":
        in_series = cfg.VECENV.get('IN_SERIES', 1); in_series = max(1, in_series)
        if num_env % in_series != 0: in_series = 1
        envs = SubprocVecEnv(env_fns, in_series=in_series, context="fork")
    else: raise ValueError(f"Unsupported VECENV.TYPE: {vecenv_type}")


    obs_to_norm_keys = cfg.MODEL.get('OBS_TO_NORM', [])
    # if obs_to_norm_keys and isinstance(obs_to_norm_keys, list) and len(obs_to_norm_keys) > 0:
    #     envs = VecNormalize(envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew, obs_to_norm=obs_to_norm_keys)
    # ---------------------------------------------
    # ---------------------------------------------
    # --- make_vec_envs, just above VecNormalize -------------
    
    obs_to_norm_keys = cfg.MODEL.get('OBS_TO_NORM', [])
    # if isinstance(envs.observation_space, gym.spaces.Dict):
    #     obs_to_norm_keys = [
    #         k for k in obs_to_norm_keys if k in envs.observation_space.spaces
    #     ]
    #     if len(obs_to_norm_keys) < cfg.MODEL.get('OBS_TO_NORM', []).__len__():
    #         print(f"[VecNormalize] skipped missing keys: "
    #             f"{set(cfg.MODEL.OBS_TO_NORM) - set(obs_to_norm_keys)}")
    # else:
    #     obs_to_norm_keys = []  # flatten – nothing to normalise per‑key

    if obs_to_norm_keys and isinstance(obs_to_norm_keys, list) and len(obs_to_norm_keys) > 0:
             envs = VecNormalize(envs, gamma=cfg.PPO.GAMMA, training=training, ret=norm_rew, obs_to_norm=obs_to_norm_keys)
    # ---------------------------------------------
    # ---------------------------------------------
    envs = VecPyTorch(envs, device)
    return envs


def get_vec_normalize(venv): 
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, "venv"): 
        return get_vec_normalize(venv.venv)
    return None

def get_ob_rms(venv): 
    vec_norm = get_vec_normalize(venv)
    return getattr(vec_norm, "ob_rms", None) if vec_norm else None

def set_ob_rms(venv, ob_rms): 
    vec_norm = get_vec_normalize(venv)
    if vec_norm and ob_rms is not None: vec_norm.ob_rms = ob_rms

class TimeLimitMask(gym.Wrapper): 
    def __init__(self, env): 
        super().__init__(env)
        self._max_episode_steps = getattr(env,'horizon',None) or getattr(env,'_max_episode_steps',None)
    
    def step(self, action):
        obs,rew,done,info=self.env.step(action)
        truncated=info.get("TimeLimit.truncated",False)
        if not truncated and hasattr(self.env,'_elapsed_steps') and self._max_episode_steps is not None and hasattr(self.env, '_elapsed_steps'):
            # Use >= to handle potential off-by-one
            if self.env._elapsed_steps >= self._max_episode_steps: 
                truncated=True 
                info["TimeLimit.truncated"]=True
        # Add timeout key for buffer if done and truncated
        if done and truncated: info["timeout"]=True
        return obs,rew,done,info

    def reset(self, **kwargs): 
        return self.env.reset(**kwargs) 

class RecordEpisodeStatistics(gym.Wrapper): 
    def __init__(self, env, deque_size=100): 
        super().__init__(env)
        self.t0=time.time()
        self.episode_return=0.0
        self.episode_length=0
        self.return_queue=deque(maxlen=deque_size)
        self.length_queue=deque(maxlen=deque_size)
        self.success_queue=deque(maxlen=deque_size)
        self.episode_return_components = defaultdict(float)\
        
    def reset(self,**kwargs): 
        observation=super().reset(**kwargs)
        self.episode_return=0.0
        self.episode_length=0
        self.episode_return_components.clear()
        return observation
    
    def step(self,action):
        observation,reward,done,info=super().step(action)
        self.episode_return+=reward
        self.episode_length+=1
        for key, value in info.items():
            if "__reward__" in key and isinstance(value,(int,float)):
                self.episode_return_components[key]+=value
        if done:
            ep_info={"r":self.episode_return,"l":self.episode_length,"t":round(time.time()-self.t0,6)}
            for key, value in self.episode_return_components.items(): 
                ep_info[key]=value
            current_success=info.get("success",False)
            ep_info["success"]=current_success
            self.success_queue.append(current_success)
            info["episode"]=ep_info
            self.return_queue.append(self.episode_return)
            self.length_queue.append(self.episode_length)
            self.episode_return=0.0
            self.episode_length=0
            self.episode_return_components.clear()

        return observation,reward,done,info


# ------------------------
# ---------TEST-----------
# ------------------------

if __name__ == "__main__":
    print("--- Running envs.py Test ---")
    # --- Manually set necessary config values for the test ---
    # Load from a test yaml file
    try:
        # Assumes test_robosuite_config.yaml is in the same directory or reachable
        # You might need to adjust the path
        test_config_path = os.path.join('/var/autofs/misc/home/alimidhat/Documents/PythonProjects/ModuMorph//configs/test_envs.yaml') # Adjust path as needed
        if not os.path.exists(test_config_path):
             print(f"Test config not found at: {test_config_path}")
             # Create a dummy one if not found for basic execution
             test_config_path = "dummy_test_cfg.yaml"
             with open(test_config_path, 'w') as f: f.write("ENV_NAME: Robosuite-v0\nROBOSUITE: {TRAINING_MORPHOLOGIES: [Panda], ENV_NAMES: [Lift], CONTROLLERS: [JOINT_VELOCITY], ENV_ARGS: {horizon: 50}}\nMODEL: {TYPE: transformer, MAX_LIMBS: 11, MAX_JOINTS: 10, MAX_LIMBS_PER_ROBOT: 11, MAX_JOINTS_PER_ROBOT: 10, OBS_TYPES: [proprioceptive, context, edges, obs_padding_mask, act_padding_mask], OBS_TO_NORM: [proprioceptive]}\nPPO: {NUM_ENVS: 2, GAMMA: 0.99}\nVECENV: {TYPE: DummyVecEnv}\nRNG_SEED: 1\nDEVICE: cpu")

        cfg.merge_from_file(test_config_path)
        print(f"Loaded test config from: {test_config_path}")
    except Exception as e:
        print(f"Error loading test config: {e}")
        print("Falling back to minimal manual config setting...")
        # Manual fallback if YAML loading fails
        cfg.ENV_NAME = "Robosuite-v0"
        cfg.ROBOSUITE.TRAINING_MORPHOLOGIES = ["Panda", "Sawyer", ["Panda", "Panda"]]
        cfg.ROBOSUITE.ENV_NAMES = ["Lift", "Lift", "TwoArmLift"]
        cfg.ROBOSUITE.CONTROLLERS = ["JOINT_VELOCITY", "JOINT_VELOCITY", ["OSC_POSE", "OSC_POSE"]]
        cfg.ROBOSUITE.ENV_ARGS.horizon = 100
        cfg.MODEL.TYPE = "transformer"
        cfg.MODEL.MAX_LIMBS = 22
        cfg.MODEL.MAX_JOINTS = 20
        cfg.MODEL.MAX_LIMBS_PER_ROBOT=11 # Define for node offset calc
        cfg.MODEL.MAX_JOINTS_PER_ROBOT=10 # Define for node offset calc (if used)
        cfg.MODEL.OBS_TYPES = ["proprioceptive", "context", "edges", "obs_padding_mask", "act_padding_mask"]
        cfg.MODEL.OBS_TO_NORM = ["proprioceptive"]
        cfg.PPO.NUM_ENVS = 3 # Test with 3 envs to see cycling
        cfg.PPO.GAMMA = 0.99
        cfg.VECENV.TYPE = "DummyVecEnv"
        cfg.RNG_SEED = 42
        cfg.DEVICE = "cpu"


    # --- Call make_vec_envs ---
    print("\nCalling make_vec_envs...")
    # Use training=True to trigger the multi-config logic
    # Use norm_rew=False for simplicity in checking raw returns if needed
    vec_env = make_vec_envs(training=True, norm_rew=False)
    print("\nVecEnv created.")
    print("Observation Space:", vec_env.observation_space)
    print("Action Space:", vec_env.action_space)

    # --- Test Reset ---
    print("\nTesting vec_env.reset()...")
    obs = vec_env.reset()
    print("Reset successful. Observation batch type:", type(obs))
    if isinstance(obs, dict):
        for key, val in obs.items():
            print(f"  Obs['{key}'] batch shape: {val.shape}")
    else:
        print(f"  Observation batch shape: {obs.shape}")


    # --- Test Step ---
    print("\nTesting vec_env.step()...")
    # Sample random actions compatible with the Pytorch wrapper space
    action_sample = torch.stack([torch.from_numpy(vec_env.action_space.sample())
                                 for _ in range(cfg.PPO.NUM_ENVS)]).to(cfg.DEVICE)
    print("Sampled action batch shape:", action_sample.shape)
    try:
        next_obs, reward, done, infos = vec_env.step(action_sample)
        print("Step successful.")
        print("  Reward batch shape:", reward.shape)
        print("  Done batch shape:", done.shape)
        print("  Infos length:", len(infos))
        if infos: print("  Info[0] keys:", infos[0].keys())
    except Exception as e:
        print(f"!!! Error during vec_env.step(): {e}")


    # --- Cleanup ---
    print("\nClosing vec_env...")
    vec_env.close()
    print("--- Test Complete ---")

    # Clean up dummy config if created
    if 'test_config_path' in locals() and test_config_path == "dummy_test_cfg.yaml" and os.path.exists(test_config_path):
         os.remove(test_config_path)