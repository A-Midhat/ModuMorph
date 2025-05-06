# metamorph/envs/wrappers/robosuite_wrappers.py
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict
import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS 

from collections import OrderedDict
import time # For potential timing debug

from metamorph.config import cfg
from metamorph.utils import spaces as spu

# class RobosuiteEnvWrapper(gym.Env):
#     """
#     Wraps a robosuite environment to make it compatible with gym.Env.
#     """
#     metadata = {'render.modes': ['human', 'rgb_array']}

#     def __init__(self, robosuite_env_name, robot_name, horizon=100, controller_name=None, robosuite_cfg=None):
#         """
#         robotsuite_env_name: the task (i.e. Lift).
#         robot_name: the robot (i.e. Panda).
#         controller_name: Method of controlling joints (i.e. JOINT_POSITION).
#         robosuite_cfg: a dict of robosuite.make() kwargs.
#         """
#         super().__init__()
#         self.robosuite_env_name = robosuite_env_name
#         self.robot_name = robot_name
#         self.robosuite_cfg = robosuite_cfg if robosuite_cfg is not None else {}

#         # --- Controller Setup --- 
#         # The problem with the different controllers is that they have different action spaces.
#         # TODO: try and make the action space the same for all controllers. Or generalize the action space.

#         controller_name = controller_name 
#         if controller_name not in ALL_CONTROLLERS:
#             print("[RobosuiteEnvWrapper] Warning: controller_name not in ALL_CONTROLLERS. Using default controller. (JOINT_POSITION)")
#             self.controller_name = "JOINT_POSITION"
        
#         try: 
#             self.controller_config = load_controller_config(default_controller=controller_name)
#         except Exception as e:
#             print(f"[RobosuiteEnvWrapper] Error loading controller_config: {e}")
#             self.controller_config = load_controller_config(default_controller="JOINT_POSITION")

#         # --- Env Setup --- 
#         # TODO: Add to config file
#         robosuite_init_args = {
#             "env_name": self.robosuite_env_name,
#             "robots": self.robot_name,
#             "controller_configs": self.controller_config,
#             "has_renderer": False, # False for training
#             "has_offscreen_renderer": False, # TODO: Set True to use camera observations as exteroceptive observations
#             "ignore_done": False, # Use PPO for termniation (TODO:Uncomment)
#             "use_camera_obs": False, # TODO: Set True to use camera observations as exteroceptive observations
#             "control_freq":20,
#             "hard_reset": False, # TODO: check if it is really faster during training 
#             "horizon":100,
#             "reward_shaping": True
#         }

#         robosuite_init_args.update(self.robosuite_cfg.get("ENV_ARGS", {}))
#         # print("Robosuite init args: ", robosuite_init_args)

#         # Create the env
#         self.env = robosuite.make(**robosuite_init_args)
        
#         self.horizon = self.env.horizon
#         self._max_episode_steps = self.horizon


#         # --- Gym Interfase Setup ---
#         act_low, act_high = self.env.action_spec
#         self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

#         # the following obs will be overriden by the observation wrapper
#         obs_spec = self.env.observation_spec()
#         gym_obs_spaces = {}
#         for key, spec in obs_spec.items():
#             try:
#                 _shape = spec.shape
#                 _dtype = spec.dtype

#                 _dtype = np.float32 if _dtype == np.float64 else _dtype
#                 gym_obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=_shape, dtype=_dtype)
#             except Exception as e:
#                 print(f"[RobosuiteEnvWrapper] Error processing obs spec: {e}")
#                 continue
#         self.observation_space = spaces.Dict(gym_obs_spaces) # new obs space 
        
#         # ---Metadata--- 
#         self._robot_metadata = self._extract_robot_metadata()
#         self.metadata["robot_metadata"] = self._robot_metadata 
#         # for multi-robot envs, we will use robot_name 
#         #TODO: add some kind of unique Id (when using same robot but differen kinamtics)
#         self._elapsed_steps = 0 # for timelimt wrapper  

#     def _extract_robot_metadata(self):

#         metadata = {}
#         if hasattr(self.env, "robots") and len(self.env.robots) > 0:
#             robot = self.env.robots[0]
#             metadata["robot_name"] = robot.name 
#             metadata['num_arm_joints'] = len(robot._ref_joint_indexes) # use _ref_joint_indexes instead of arm_joint_names, returns list of len (num of joints) without gripper
#             metadata['num_gripper_joints'] = robot.gripper.dof # for 1 gripper it returns 1, if i want the real joints we can use i.e. robot._gripper_joint_pos_indexes (2)
            
#             # Number of nodes (DoF + Gripper)
#             metadata['num_nodes'] = metadata['num_arm_joints'] + metadata['num_gripper_joints']
#             # TODO: add the following (lower, upper, damping, armature, friction) physical properties (check dir(robot))
#             metadata['joint_names'] = robot.robot_joints 
#             metadata['g_joint_name'] = robot.gripper_joints 
#             # should I combine them? TODO: check again
#             #metadata['joint_names'] = robot.robot_joints + robot.gripper_joints

#         else: 
#             # dummyy values
#             print("[RobosuiteEnvWrapper] Warning: No robot found in the environment. Metadata will set to None.")
#             metadata['robot_name'] = "Unkown"
#             metadata['num_arm_joints'] = 0
#             metadata['num_gripper_joints'] = 0
#             metadata['num_nodes'] = 0
#             metadata['joint_names'] = []
#             #metadata['link_names'] = []
        
#         #print("Robot metadata: ", metadata)
#         return metadata
    
#     def step(self, action):
#         action = np.clip(action, self.action_space.low, self.action_space.high) # enusre [-1, 1] range
#         try:
#             # done here is ignored 
#             obs_dict, reward, done, info = self.env.step(action)
#             #done = False # TODO: UNCOMMNET
#             #---Debug ---
#             # if self.ep_step_count %20==0 or reward !=0:
#             #     print(f" [RobosuiteEnvWrapper Step {self.ep_step_count}] Raw Reward: {reward}")
#             #-------------
#         except Exception as e:
#             print(f"[RobosuiteEnvWrapper] ERROR during Robosuite step: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Action=`{action}`, Error={e}")
#             raise e # for debugging 
#             # TODO: add a flag to skip the step and return None for training stablity (UNCOMMENT)
#             obs_dict = self.observation_space.sample() # dummy obs
#             reward = -10 # heavily penelized 
#             done = True 
#             info = {'error': str(e), 'name': self.robot_name}
        
#         # --- Post-processing of obs---
#         processed_obs = self._convert_observation(obs_dict)
        
#         # ---Custom info---
#         #info['robot_name'] = self.robot_name # causing error at meter
#         info['name'] = self.robot_name
#         info['raw_reward'] = reward
#         info['action'] = action
#         info['success'] = self.env._check_success()

#         self._elapsed_steps += 1

#         return processed_obs, reward, done, info

#     def reset(self):
#         try: 
#             obs_dict = self.env.reset()
#         except Exception as e:
#             print(f"[RobosuiteEnvWrapper] ERROR during Robosuite reset: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Error={e}")
#             raise e # for debugging
#             # TODO: add a flag to skip the reset and return None for training stablity (UNCOMMENT)
#             #obs_dict = self.observation_space.sample() # dummy obs
#             #reward = -10 # heavily penelized
#             # done = True
#         processed_obs = self._convert_observation(obs_dict)
#         return processed_obs
    
#     def _convert_observation(self, obs_dict):
#         """
#         Converts the flat observation dict from robosuite into 
#         preliminary structured dict, grouping related info logically, 
#         before we pass it through the NodeCentricObservationWrapper.
#         """

#         prelim_obs = {}
#         if self.robosuite_env_name.startswith("TwoArm"):
#             """
#             odict_keys(['gripper0_to_handle0', 'gripper1_to_handle1', 'handle0_xpos',
#              'handle1_xpos', 'object-state', 'pot_pos', 'pot_quat', 'robot0_eef_pos', 
#              'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 
#              'robot0_joint_vel', 'robot0_proprio-state', 'robot1_eef_pos', 
#              'robot1_eef_quat', 'robot1_gripper_qpos', 'robot1_gripper_qvel', 'robot1_joint_pos_cos',
#              'robot1_joint_pos_sin', 'robot1_joint_vel', 'robot1_proprio-state'])
#             """
#             robot_proprio_keys = [f'robot{i}_proprio-state' for i in range(2)] 
#             for key in robot_proprio_keys:
#                 if key in obs_dict:
#                     prelim_obs[key] = obs_dict[key]
              
#             return prelim_obs            
#         else: 
#             print(f"[RobosuiteEnvWrapper] Converting obs for SingleArm")
#             # ---Propricopetive State--- 
#             # we can use the combined state provided by robotsuite 
#             # TODO: try and handle this for multiple robots 
#             robot_proprio_key = f'robot0_proprio-state' # change the '0' to the robot index if we have multiple robots (guess?)
#             if robot_proprio_key in obs_dict:
#                 prelim_obs[robot_proprio_key] = obs_dict[robot_proprio_key]
#             else:
#                 print(f"[RobosuiteEnvWrapper] Warning: {robot_proprio_key} state not found for robot {self.robot_name} in the observation dictionary.")
#                 # the problem is this is robot-dependent naming [Not Generalized]
#                 parts = []
#                 keys_to_try = [
#                     f'robot0_joint_pos_cos', f'robot0_joint_pos_sin', # Use sin/cos for angles
#                     f'robot0_joint_vel',
#                     f'robot0_eef_pos', f'robot0_eef_quat',
#                     f'robot0_gripper_qpos', f'robot0_gripper_qvel'
#                 ]
#                 for key in keys_to_try:
#                     if key in obs_dict:
#                         parts.append(obs_dict[key].astype(np.float32))
#                     else:
#                         # either pad or raise an error ? 
#                         print(f"[RobosuiteEnvWrapper] Missing key: {key}")
#                         # skip for now 
#                         pass 
#                 if not parts:
#                     # create a dummy observation (zeros)
#                     # Problem we need to know the shape of the observation
#                     print(f"[RobosuiteEnvWrapper] ERROR: Could not assemble robot-state for {self.robot_name}.")
#                     prelim_obs['robot0_proprio-state'] = np.zeros(1, dtype=np.float32) # leads to problem better to raise an error
#                 else :
#                     prelim_obs['robot0_proprio-state'] = np.concatenate(parts)
            
#             # ---Object State---
#             if 'object-state' in obs_dict:
#                 prelim_obs['object-state'] = obs_dict['object-state'].astype(np.float32)
#             else:
#                 # Either Env doesnt have object or is it disabled
#                 print(f"[RobosuiteEnvWrapper] Warning: object-state not found in the observation dictionary.")
#                 # create a dummy observation (zeros)
#                 prelim_obs['object-state'] = np.zeros([], dtype=np.float32)
            
#             # TODO: Combined states (proprio + obj) for MLP (single robot)
#             # combined_state = np.concatenate([obs_dict[robot_proprio_key], obs_dict['object-state']])
#             # --------------------------------------------------------------------------------------
#             # ---Gripper to Object Distance--- (This is the last 3 elements of object-state) redundant with object-state
#             # TODO: alt we can use the eef_pos and gripper_to_object directly
#             # obj_pos_key = 'object-state' # gripper to obj is the last 3 elemtns in object-state 
#             # eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
#             # if obj_pos_key in obs_dict and eef_pos_key in obs_dict:
#             #     obj_pos = obs_dict[obj_pos_key][-3:]
#             #     eef_pos = obs_dict[eef_pos_key]
#             #     prelim_obs['gripper_to_object'] = (obj_pos - eef_pos).astype(np.float32)
#             # else:
#             #     print(f"[RobosuiteEnvWrapper] Warning: {obj_pos_key} or {eef_pos_key} not found in the observation dictionary.")
#             #     # create a dummy observation (zeros)
#             #     prelim_obs['gripper_to_object'] = np.zeros(3, dtype=np.float32) # TODO: better to set high value 

#             # ---EEF State--- (TODO: add this As a direct input to EXT_MIX)
#             # TODO: make the name consistent with the other keys
#             eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
#             if eef_pos_key in obs_dict:
#                 prelim_obs['robot0_eef_pos'] = obs_dict[eef_pos_key].astype(np.float32)
#             else:
#                 print(f"[RobosuiteEnvWrapper] Warning: {eef_pos_key} not found in the observation dictionary.")
#                 # create a dummy observation (zeros)
#                 prelim_obs['robot0_eef_pos'] = np.zeros(3, dtype=np.float32)
            
#             eef_quat_key = f'robot0_eef_quat' # change `0` to the robot index if we have multiple robots (guess)
#             if eef_quat_key in obs_dict:
#                 prelim_obs['robot0_eef_quat'] = obs_dict[eef_quat_key].astype(np.float32)
#             else:
#                 print(f"[RobosuiteEnvWrapper] Warning: {eef_quat_key} not found in the observation dictionary.")
#                 prelim_obs['robot0_eef_quat'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Default Identity 
#             # f'robot0_gripper_qpos', f'robot0_gripper_qvel'
#             # ---Gripper State---
#             gripper_qpos_key = f'robot0_gripper_qpos'
#             if gripper_qpos_key in obs_dict:
#                 prelim_obs['robot0_gripper_qpos'] = obs_dict[gripper_qpos_key].astype(np.float32)
#             else:
#                 print(f"[RobosuiteEnvWrapper] Warning: {gripper_qpos_key} not found in the observation dictionary.")
#                 prelim_obs['robot0_gripper_qpos'] = np.zeros(1, dtype=np.float32)

#             gripper_qvel_key = f'robot0_gripper_qvel'
#             if gripper_qvel_key in obs_dict:
#                 prelim_obs['robot0_gripper_qvel'] = obs_dict[gripper_qvel_key].astype(np.float32)
#             else:
#                 print(f"[RobosuiteEnvWrapper] Warning: {gripper_qvel_key} not found in the observation dictionary.")
#                 prelim_obs['robot0_gripper_qvel'] = np.zeros(1, dtype=np.float32)
#             # --- Robot Context (Static Info) --- 
#             # features extracted from the _robot_metadata so node-centric wrapper can easily distribute it to all nodes
#             prelim_obs['robot_context_dict'] = self.metadata['robot_metadata']
#             # TODO: Combine all eef and gripper obs into a single combined state (similar to the object-state/proprio)
#             return prelim_obs
        
#     def close(self):
#         self.env.close()
    
#     def render(self, mode='human', width=512, height=512, camera_name=None):
#         if mode == "rgb_array":
#             if self.env.has_offscreen_renderer:
#                 cam_name = camera_name if camera_name else self.env.render_camera 
#                 img = self.env.sim.render(
#                     camera_name=cam_name,
#                     width=width,
#                     height=height,
#                     depth=False
#                 )
#                 img = img[::-1]             
#                 return img  
#             else :
#                 print("Warning: Offscreen rendering is not available for this environment")
#                 img =  self.env.sim.render(
#                     mode='offscreen',
#                     width=width,
#                     height=height,
#                     depth=False
#                 ) # alt return zeros (blank)
#                 img = img[::-1]             
#                 return img 
#         elif mode == "human":
#             if self.env.has_renderer:
#                 self.env.render()
#             else:
#                 print("Warning: 'human' Rendering is not available for this environment")
#         else:
#             return super().render(mode=mode)
    
#     def seed(self, seed=None):
#         # Normally robosuite handles this when robosuite.make is called

#         if hasattr(self.env, "seed") and callable(self.env.seed):
#             self.env.seed(seed)
#         self.action_space.seed(seed)
    
#     def sample_action(self):
#         # mimic the gym action_space.sample()
#         low, high = self.action_space.low, self.action_space.high 
#         action = np.random.uniform(low, high)
#         return action.astype(self.action_space.dtype)


class RobosuiteMLPFlattener(gym.ObservationWrapper):
    """
    Flattens ['robot0_proprio-state', 'object-state'] into a single vector
    and adds dummy masks / edges expected by MetaMorph's MLP model.
    """
    def __init__(self, env):
        super().__init__(env)

        self.keys_to_flatten = ["robot0_proprio-state", "object-state"]
        self.max_limbs  = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS

        # get a *real* observation
        first_obs_dict = self.env.reset()  # early reset 
        flat           = self._flatten(first_obs_dict)
        self.flat_obs_dim = flat.shape[0]          # will be 42 for Panda‑Lift

        inf  = np.float32(np.inf)
        dmsk = (self.max_limbs,)           # dummy‑mask shape
        dedg = (2 * self.max_joints,)      # dummy‑edge shape

        self.observation_space = spaces.Dict({
            "proprioceptive"   : Box(-inf, inf, (self.flat_obs_dim,), np.float32),
            "obs_padding_mask" : Box(False, True, dmsk, dtype=bool),
            "act_padding_mask" : Box(False, True, dmsk, dtype=bool),
            "edges"            : Box(-inf,  inf, dedg, dtype=np.float32),
            "context"          : Box(-inf,  inf, (1,),  dtype=np.float32),
        })
        self.action_space = env.action_space

        # store the reset we already did so the outer env gets the
        # exact same initial state it expects
        self._cached_initial_obs = first_obs_dict

    def _flatten(self, obs_dict):
        pieces = []
        for k in self.keys_to_flatten:
            if k in obs_dict and obs_dict[k].size:
                pieces.append(obs_dict[k].ravel())
            else:
                print(f"[MLPFlattener] Warning: {k} not found in the observation dictionary.")
                pieces.append(np.zeros(0, dtype=np.float32))  # will never happen after first reset
        return np.concatenate(pieces).astype(np.float32)

    def observation(self, obs_dict):
        flat = self._flatten(obs_dict)

        if flat.shape[0] != self.flat_obs_dim:     # guard against surprises
            raise ValueError(
                f"[MLPFlattener] Expected {self.flat_obs_dim} dims, "
                f"got {flat.shape[0]}"
            )

        dummy_mask   = np.zeros(self.max_limbs,      dtype=bool)
        dummy_edges  = np.zeros(2 * self.max_joints, dtype=np.float32)
        dummy_ctx    = np.zeros(1, dtype=np.float32)

        return OrderedDict({
            "proprioceptive"   : flat,
            "obs_padding_mask" : dummy_mask,
            "act_padding_mask" : dummy_mask,
            "edges"            : dummy_edges,
            "context"          : dummy_ctx,
        })

    def reset(self, **kw):
        if self._cached_initial_obs is not None:
            obs_dict = self._cached_initial_obs
            self._cached_initial_obs = None           # only use once
        else:
            obs_dict = self.env.reset(**kw)
        return self.observation(obs_dict)

    # actions pass straight through
    def action(self, action):
        return action


# metamorph/envs/wrappers/robosuite_wrappers.py
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict
import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from robosuite.utils.mjcf_utils import find_elements # Helper for finding joints

from collections import OrderedDict, defaultdict
import time # For potential timing debug

from metamorph.config import cfg
from metamorph.utils import spaces as spu
# from metamorph.utils import mjpy as mu # For MuJoCo IDs if needed (for context)

# === RobosuiteEnvWrapper (Minor adjustments possible) ===
class RobosuiteEnvWrapper(gym.Env):
    """
    Wraps a robosuite environment to make it compatible with gym.Env.
    Sources dynamic observations from the environment's obs_dict.
    Provides access to sim.model for static context.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    # --- Keep __init__ largely the same ---
    def __init__(self, robosuite_env_name, robot_name, horizon=100, controller_name=None, robosuite_cfg=None):
        super().__init__()
        self.robosuite_env_name = robosuite_env_name
        self.robot_name = robot_name # Store the primary robot name
        self.robosuite_cfg = robosuite_cfg if robosuite_cfg is not None else {}

        # --- Controller Setup ---
        controller_name = controller_name if controller_name else cfg.ROBOSUITE.CONTROLLER
        if controller_name not in ALL_CONTROLLERS:
            print(f"[RobosuiteEnvWrapper] Warning: controller '{controller_name}' not in ALL_CONTROLLERS. Using default JOINT_VELOCITY.")
            controller_name = "JOINT_VELOCITY" # A common default
        self.controller_name = controller_name

        try:
            self.controller_config = load_controller_config(default_controller=self.controller_name)
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] Error loading controller_config for '{self.controller_name}': {e}")
            print("[RobosuiteEnvWrapper] Falling back to JOINT_VELOCITY config.")
            self.controller_config = load_controller_config(default_controller="JOINT_VELOCITY")
            self.controller_name = "JOINT_VELOCITY"

        # --- Env Setup ---
        robosuite_init_args = {
            "env_name": self.robosuite_env_name,
            "robots": self.robot_name, # Accepts single string or list
            "controller_configs": self.controller_config,
            "has_renderer": False,
            "has_offscreen_renderer": False, # Keep false unless camera obs are used
            "ignore_done": False, # Let the env terminate naturally based on horizon or task
            "use_camera_obs": False,
            "control_freq": 20,
            "hard_reset": False,
            "horizon": horizon, # Use passed horizon
            "reward_shaping": True # Usually good for training
        }
        # Allow overrides from config
        robosuite_init_args.update(self.robosuite_cfg.get("ENV_ARGS", {}))

        # Create the env
        print(f"[RobosuiteEnvWrapper] Creating robosuite env with args: {robosuite_init_args}")
        try:
            self.env = robosuite.make(**robosuite_init_args)
        except Exception as e:
            print(f"!!! Failed to create robosuite environment: {e} !!!")
            raise

        self.horizon = self.env.horizon
        self._max_episode_steps = self.horizon # For compatibility with some gym checks

        # --- Gym Interface Setup ---
        # Action Space: Get directly from Robosuite
        act_low, act_high = self.env.action_spec
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # Observation Space: Define based on Robosuite's spec. This will be
        # *overridden* by the NodeCentricObservation wrapper later, but is useful
        # for introspection and provides the keys for _convert_observation.
        obs_spec = self.env.observation_spec()
        gym_obs_spaces = OrderedDict() # Use OrderedDict
        for key, spec in obs_spec.items():
            # Ensure spec has shape and dtype (handle potential non-array obs)
            if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
                # Ensure dtype is float32 or bool, etc. (no float64)
                dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
                # Ensure shape is a tuple
                shape = tuple(spec.shape)
                gym_obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)
            # else:
                # print(f"[RobosuiteEnvWrapper] Skipping observation spec key '{key}' - not a standard array spec.")
        self.observation_space = spaces.Dict(gym_obs_spaces)

        # --- Metadata ---
        self._robot_metadata = self._extract_robot_metadata()
        self.metadata["robot_metadata"] = self._robot_metadata

        self._elapsed_steps = 0

    def _extract_robot_metadata(self):
        # (Keep your existing implementation - it seems reasonable for single robot)
        metadata = {}
        if hasattr(self.env, "robots") and len(self.env.robots) > 0:
            # TODO: Handle multiple robots if necessary in the future
            robot = self.env.robots[0] # Assume single robot for now
            metadata["robot_name"] = robot.name
            # Use robot properties directly
            metadata['num_arm_joints'] = robot.dof - robot.gripper.dof # Arm DoF
            metadata['num_gripper_joints'] = robot.gripper.dof # Gripper DoF (often 1 or 2 conceptually)
            metadata['num_nodes'] = metadata['num_arm_joints'] + metadata['num_gripper_joints'] # Example - might be refined
            metadata['action_dim'] = robot.action_dim # Total action dim for this robot
            metadata['robot_joints'] = robot.robot_joints # List of joint names
            metadata['gripper_joints'] = robot.gripper.joints # List of gripper joint names
            # Potentially add more: base type (fixed/mobile), etc.
        else:
            # Dummy values
            print("[RobosuiteEnvWrapper] Warning: No robot found. Metadata set to defaults.")
            metadata = {'robot_name': "Unknown", 'num_arm_joints': 0, 'num_gripper_joints': 0, 'num_nodes': 0, 'action_dim': 0, 'robot_joints': [], 'gripper_joints': []}
        return metadata

    def _convert_observation(self, obs_dict):
        # This function primarily ensures correct dtypes and potentially
        # prepares keys needed by downstream wrappers.
        # It *doesn't* structure into node-centric format here.
        processed_obs = OrderedDict()
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                # Ensure float32 dtype
                dtype = np.float32 if value.dtype == np.float64 else value.dtype
                processed_obs[key] = value.astype(dtype)
            else:
                processed_obs[key] = value # Pass non-array data through

        # Add the static metadata dict - the node-centric wrapper will use this AND sim.model
        processed_obs['robot_metadata'] = self.metadata['robot_metadata']

        return processed_obs

    def step(self, action):
        # Clip action
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            obs_dict, reward, done, info = self.env.step(action)
            # Note: 'done' might be True due to horizon limit even if ignore_done=True was set in make()
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] ERROR during Robosuite step: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Action=`{action}`")
            print(f"Error details: {e}")
            # Handle error: maybe return last valid obs? or crash?
            # For now, re-raise to make debugging clear
            raise e
            # Alternative:
            # obs_dict = self.observation_space.sample() # Risky, might not have correct keys/dtypes
            # reward = -10
            # done = True
            # info = {'error': str(e)}

        self._elapsed_steps += 1

        # Check for horizon limit separately
        if self._elapsed_steps >= self.horizon:
            done = True
            info['TimeLimit.truncated'] = True # Standard gym key

        processed_obs = self._convert_observation(obs_dict)

        # Add standard diagnostic info
        info['name'] = self.robot_name # For multi-env differentiation
        info['raw_reward'] = reward
        info['action'] = action # Log the action taken
        info['success'] = self.env._check_success() # Standard Robosuite success check

        return processed_obs, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        try:
            obs_dict = self.env.reset()
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] ERROR during Robosuite reset: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`")
            print(f"Error details: {e}")
            raise e
        processed_obs = self._convert_observation(obs_dict)
        return processed_obs

    # --- Keep render, seed, close, sample_action methods as before ---
    def close(self):
        self.env.close()

    def render(self, mode='human', width=512, height=512, camera_name=None):
        # Find default camera if not specified
        if camera_name is None:
            camera_name = self.env.render_camera if hasattr(self.env, 'render_camera') else "frontview"
            if camera_name not in self.env.sim.model.camera_names:
                 camera_name = self.env.sim.model.camera_names[0] if len(self.env.sim.model.camera_names) > 0 else None

        if mode == "rgb_array":
             if camera_name and self.env.viewer is None: # Check if offscreen context exists implicitely
                 try:
                     img = self.env.sim.render(
                         camera_name=camera_name,
                         width=width,
                         height=height,
                         depth=False
                     )
                     return img[::-1] # MuJoCo renders upside down
                 except Exception as e:
                     print(f"Warning: Failed to render rgb_array with camera '{camera_name}': {e}")
                     return np.zeros((height, width, 3), dtype=np.uint8) # Return blank image on error
             elif self.env.viewer is not None and self.env.viewer.is_running: # Onscreen viewer exists
                 self.env.render() # Ensure viewer is up-to-date
                 # The following gets pixels from the *onscreen* viewer
                 img = self.env.viewer.read_pixels(width, height, depth=False)
                 return img[::-1]
             else:
                 print("Warning: RGB array rendering requires an offscreen renderer context or an active onscreen viewer.")
                 return np.zeros((height, width, 3), dtype=np.uint8) # Return blank image
        elif mode == "human":
            self.env.render()
        else:
            return super().render(mode=mode) # Let gym handle other modes

    def seed(self, seed=None):
        # Robosuite envs typically handle seeding internally via np.random.seed,
        # but exposing this follows the gym API.
        if hasattr(self.env, 'seed') and callable(self.env.seed):
            self.env.seed(seed)
        # Seed the action space for sampling consistency if needed
        self.action_space.seed(seed)

    def sample_action(self):
        return self.action_space.sample()


# === Node-Centric Observation Wrapper ===
class RobosuiteNodeCentricObservation(gym.ObservationWrapper):
    """
    Transforms RoboSuite observations into a node-centric format for Transformer.
    Nodes represent links. Extracts dynamic state from obs_dict and static
    context from sim.model. Handles padding.
    """
    def __init__(self, env):
        super().__init__(env)
        print("[Wrapper Init] RobosuiteNodeCentricObservation")

        # --- 1. Get References and Basic Metadata ---
        self.base_env_ref = self._get_base_wrapper_ref(RobosuiteEnvWrapper)
        self.sim = self.base_env_ref.env.sim
        self.model = self.sim.model
        # Use metadata stored by the base wrapper
        self.robot_metadata = self.base_env_ref.metadata['robot_metadata']
        self.robot_name = self.robot_metadata['robot_name']

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS # Max *controllable* joints for edge padding

        # --- 2. Define Node Structure (Link-Based) ---
        self._define_node_structure() # Sets self.node_body_names, self.num_nodes, etc.

        # --- 3. Define Features and Calculate Dimensions ---
        self._define_feature_sets() # Sets self.proprio_feature_keys, self.context_feature_attrs, etc.
        self.limb_obs_size = self._calculate_feature_dim(self.proprio_feature_config)
        self.context_obs_size = self._calculate_feature_dim(self.context_feature_config)
        print(f"  [{self.robot_name}] Nodes Defined: {self.num_nodes}")
        print(f"  [{self.robot_name}] Proprio Dim/Node: {self.limb_obs_size}")
        print(f"  [{self.robot_name}] Context Dim/Node: {self.context_obs_size}")

        # --- 4. Build Static Structures (Edges, Masks) ---
        self._create_kinematic_structure() # Sets self.edges_padded
        self._create_padding_masks()       # Sets self.obs_padding_mask, self.act_padding_mask
        # Store action mask in metadata for the action wrapper
        self.metadata['act_padding_mask'] = self.act_padding_mask

        # --- 5. Define Final Observation Space ---
        self.observation_space = self._define_observation_space()
        print(f"  [{self.robot_name}] Observation Space Defined. Keys: {list(self.observation_space.spaces.keys())}")

        # --- 6. Initialize Feature Arrays (Pre-allocation) ---
        self._init_feature_arrays()


    def _get_base_wrapper_ref(self, base_wrapper_type):
        """Finds the instance of base_wrapper_type in the wrapper stack."""
        env = self.env
        while hasattr(env, 'env'):
            if isinstance(env, base_wrapper_type):
                return env
            env = env.env
        # If loop finishes, check the final env itself
        if isinstance(env, base_wrapper_type):
             return env
        raise TypeError(f"Could not find {base_wrapper_type.__name__} in wrapper stack.")

    def _define_node_structure(self):
        """Identifies bodies for nodes: Base, Arm Links, Hand, Gripper Conceptual Node."""
        robot_model = self.base_env_ref.env.robots[0] # Assume single robot
        self.node_body_names = []
        self.node_types = [] # Store type ('base', 'arm', 'hand', 'gripper')

        # 1. Base Node
        base_name = robot_model.robot_model.root_body # Often 'robot0_base' or similar
        self.node_body_names.append(base_name)
        self.node_types.append('base')

        # 2. Arm Link Nodes (Iterate through controlled joints)
        # Assumes arm joints connect links sequentially from base/link0
        self.arm_joint_names = robot_model.robot_joints
        self.arm_link_names = [] # Store names of links driven by arm joints
        last_link_name = base_name
        for j_name in self.arm_joint_names:
            joint_id = self.model.joint_name2id(j_name)
            body_id = self.model.jnt_bodyid[joint_id]
            body_name = self.model.body_id2name(body_id)
            # Avoid duplicates and ensure it's not the base again
            if body_name != base_name and body_name not in self.node_body_names:
                self.node_body_names.append(body_name)
                self.node_types.append('arm')
                self.arm_link_names.append(body_name)
            last_link_name = body_name # Keep track of the last arm link

        # 3. Hand Node (Body connected to the last arm joint, often EEF mount)
        # We might have already added it if it was directly driven by last joint
        hand_name = last_link_name # Assume hand is the last link body
        if hand_name not in self.node_body_names: # Should usually be there already
             self.node_body_names.append(hand_name)
             self.node_types.append('hand')
             print(f"Warning: Hand body '{hand_name}' added separately.")
        else:
             # Find its index and change type
             hand_idx = self.node_body_names.index(hand_name)
             self.node_types[hand_idx] = 'hand'

        # 4. Conceptual Gripper Node (Anchored to Hand for structure)
        self.gripper_node_anchor_name = hand_name # Use hand as structural parent
        self.node_body_names.append(self.gripper_node_anchor_name + "_gripper_conceptual") # Unique name
        self.node_types.append('gripper')

        self.num_nodes = len(self.node_body_names)
        self.num_node_pads = self.max_limbs - self.num_nodes

        if self.num_nodes > self.max_limbs:
             raise ValueError(f"Robot {self.robot_name} requires {self.num_nodes} nodes ({self.node_body_names}), but MAX_LIMBS={self.max_limbs}")

        # Create mapping for easy lookup
        self.body_name_to_node_idx = {name: i for i, name in enumerate(self.node_body_names) if not name.endswith("_gripper_conceptual")}
        self.gripper_node_idx = self.node_body_names.index(self.gripper_node_anchor_name + "_gripper_conceptual")

    def _define_feature_sets(self):
        """Define the keys/attributes to extract for proprioceptive and context features."""
        # --- Proprioceptive Features (Dynamic, from obs_dict) ---
        self.proprio_feature_config = {
            # 'common': [], # Features common to all nodes?
            'base': [ # Fixed base assumed for now
                ('fixed_encoding', 3), # e.g., [1,0,0,0...] one-hot/indicator
            ],
            'arm': [
                # Joint state (relative to this link's parent joint)
                ('joint_pos_cos', 1),
                ('joint_pos_sin', 1),
                ('joint_vel', 1),
                # Link state (relative to world or base?) - Use world for now
                ('body_pos_world', 3), # We might need RobosuiteEnvWrapper to add this if not standard
                ('body_quat_world', 4),
                ('body_velp_world', 3),
                ('body_velr_world', 3),
            ],
            'hand': [ # Often same as arm, plus EEF state
                ('joint_pos_cos', 1),
                ('joint_pos_sin', 1),
                ('joint_vel', 1),
                # EEF world state
                ('eef_pos', 3), # Key from RobosuiteEnvWrapper
                ('eef_quat', 4), # Key from RobosuiteEnvWrapper
                 # Link velocities (optional but potentially useful)
                ('body_velp_world', 3),
                ('body_velr_world', 3),
            ],
            'gripper': [ # Conceptual node state
                 ('gripper_qpos', cfg.ROBOSUITE.GRIPPER_DIM), # Use configured dim
                 ('gripper_qvel', cfg.ROBOSUITE.GRIPPER_DIM),
            ]
        }
        # We need RobosuiteEnvWrapper to ensure keys like 'body_pos_world' are available,
        # potentially by calculating them from sim.data if not direct observables.
        # Let's assume for now the necessary keys are in the obs_dict passed to observation().

        # --- Context Features (Static, from sim.model) ---
        self.context_feature_config = {
             'base': [
                 ('node_type_encoding', 4), # e.g., [1,0,0,0] for Base, Arm, Hand, Gripper
                 ('is_fixed', 1), # 1 if fixed, 0 if mobile
             ],
            'arm': [
                ('node_type_encoding', 4),
                # Joint properties (connecting to parent)
                ('joint_type', 1), # 0=free, 1=ball, 2=slide, 3=hinge
                ('joint_limits', 2), # range min, max
                ('joint_damping', 1),
                ('joint_frictionloss', 1),
                ('joint_armature', 1),
                # Link Properties
                ('body_mass', 1),
                ('body_inertia_diag', 3),
                ('geom_type', 1), # 0=plane..6=mesh; only take primary geom?
                ('geom_size', 3), # Max 3 size params
            ],
            'hand': [ # Similar to arm, maybe different node type encoding
                ('node_type_encoding', 4),
                ('joint_type', 1),
                ('joint_limits', 2),
                ('joint_damping', 1),
                ('joint_frictionloss', 1),
                ('joint_armature', 1),
                ('body_mass', 1),
                ('body_inertia_diag', 3),
                ('geom_type', 1),
                ('geom_size', 3),
            ],
            'gripper': [ # Properties of the gripper joint(s)
                 ('node_type_encoding', 4),
                 ('joint_type', 1), # Should reflect gripper joint(s) type
                 ('joint_limits', 2 * cfg.ROBOSUITE.GRIPPER_DIM), # Adapt based on gripper dim
                 ('joint_damping', 1 * cfg.ROBOSUITE.GRIPPER_DIM),
                 ('joint_frictionloss', 1 * cfg.ROBOSUITE.GRIPPER_DIM),
                 ('joint_armature', 1 * cfg.ROBOSUITE.GRIPPER_DIM),
                 # Maybe add gripper max force/velocity if available?
            ]
        }

    def _calculate_feature_dim(self, feature_config):
        """Calculates max dimension across node types for padding."""
        max_dim = 0
        for node_type, features in feature_config.items():
            current_dim = sum(dim for _, dim in features)
            max_dim = max(max_dim, current_dim)
        return max_dim

    def _init_feature_arrays(self):
        """Pre-allocate numpy arrays for efficiency."""
        self.node_proprio = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)
        self.node_context = np.zeros((self.max_limbs, self.context_obs_size), dtype=np.float32)

    def _get_required_obs_keys(self):
        """Figure out which keys MUST be in the input obs_dict."""
        keys = set()
        for node_type, features in self.proprio_feature_config.items():
            for key, dim in features:
                 # Simple keys are assumed to be directly from obs_dict
                 # More complex ones might need specific handling later
                 if key not in ['fixed_encoding']: # Skip placeholders
                     keys.add(key)
        # Add global keys needed
        keys.add('object-state') # Example global key
        return list(keys)

    def _create_kinematic_structure(self):
        """Builds padded edge list from sim.model based on defined nodes."""
        parent_ids = self.model.body_parentid
        edges = []
        self.node_idx_to_mujoco_joint_ids = defaultdict(list) # Map node idx to relevant MuJoCo joint IDs

        # Get MuJoCo joint info
        joint_parent_body_ids = []
        joint_child_body_ids = []
        for j_id in range(self.model.njnt):
            j_type = self.model.jnt_type[j_id]
            if j_type == 0: continue # Skip free joints (usually world)
            child_body_id = self.model.jnt_bodyid[j_id]
            parent_body_id = parent_ids[child_body_id]
            joint_parent_body_ids.append(parent_body_id)
            joint_child_body_ids.append(child_body_id)

            # Store mapping for context extraction
            child_body_name = self.model.body_id2name(child_body_id)
            if child_body_name in self.body_name_to_node_idx:
                 child_node_idx = self.body_name_to_node_idx[child_body_name]
                 self.node_idx_to_mujoco_joint_ids[child_node_idx].append(j_id)

        # Build edges based on nodes
        for child_body_name, node_idx in self.body_name_to_node_idx.items():
            if node_idx == 0: continue # Base has no parent edge in this structure
            child_mujoco_id = self.model.body_name2id(child_body_name)
            parent_mujoco_id = parent_ids[child_mujoco_id]
            parent_body_name = self.model.body_id2name(parent_mujoco_id)

            if parent_body_name in self.body_name_to_node_idx:
                parent_node_idx = self.body_name_to_node_idx[parent_body_name]
                edges.extend([node_idx, parent_node_idx]) # Child -> Parent

        # Add conceptual gripper edge (Hand -> Gripper)
        hand_node_idx = self.node_types.index('hand') # Find hand node index
        edges.extend([self.gripper_node_idx, hand_node_idx])

        self.edges_unpadded = np.array(edges, dtype=np.int32)
        num_real_edges = len(self.edges_unpadded) // 2

        if num_real_edges > self.max_joints:
             print(f"Warning: Robot {self.robot_name} has {num_real_edges} edges, exceeding MAX_JOINTS={self.max_joints}. Truncating.")
             self.edges_unpadded = self.edges_unpadded[:2 * self.max_joints]
             num_real_edges = self.max_joints

        # Pad edges
        num_edge_pads = self.max_joints - num_real_edges
        pad_value = self.max_limbs - 1 # Dummy node index for padding
        # Pad with (dummy_idx, dummy_idx) self-loops
        padded_edges_flat = np.full(2 * self.max_joints, pad_value, dtype=np.int32)
        padded_edges_flat[:len(self.edges_unpadded)] = self.edges_unpadded
        self.edges_padded = padded_edges_flat.astype(np.float32) # Store as float

    def _create_padding_masks(self):
        """Creates observation and action padding masks."""
        # Observation Mask: False for real nodes, True for padding
        self.obs_padding_mask = np.array([False] * self.num_nodes + [True] * self.num_node_pads, dtype=bool)

        # Action Mask: True for non-actuated nodes, False for actuated ones
        self.act_padding_mask = np.ones(self.max_limbs, dtype=bool) # Start all masked (True)
        num_arm_joints_actual = len(self.arm_joint_names)
        node_indices_with_arm_action = []
        node_indices_with_gripper_action = []

        for node_idx, node_type in enumerate(self.node_types):
            if node_idx >= self.num_nodes: break # Stop at real nodes
            if node_type == 'arm' or node_type == 'hand':
                 # Arm/Hand nodes correspond to arm joints before them
                 # This assumes a sequential structure mapping
                 if (node_idx - 1) < num_arm_joints_actual: # node_idx 1 maps to joint 0, etc.
                      self.act_padding_mask[node_idx] = False
                      node_indices_with_arm_action.append(node_idx)
            elif node_type == 'gripper':
                 self.act_padding_mask[node_idx] = False
                 node_indices_with_gripper_action.append(node_idx)
            # Base node (index 0) remains True (masked)

        # Store indices for action wrapper
        self.metadata['arm_action_node_indices'] = node_indices_with_arm_action
        self.metadata['gripper_action_node_indices'] = node_indices_with_gripper_action # Should be just one index
        self.metadata['act_padding_mask'] = self.act_padding_mask # Store the final mask


    def _define_observation_space(self):
        """Defines the final Dict observation space."""
        inf = np.float32(np.inf)
        obs_spaces = OrderedDict()

        # Node-centric features (Flattened)
        obs_spaces['proprioceptive'] = Box(-inf, inf, (self.max_limbs * self.limb_obs_size,), np.float32)
        obs_spaces['context'] = Box(-inf, inf, (self.max_limbs * self.context_obs_size,), np.float32)

        # Structure
        obs_spaces['edges'] = Box(-inf, inf, (2 * self.max_joints,), np.float32) # Use inf for consistency

        # Masks
        obs_spaces['obs_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool)
        obs_spaces['act_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool)

        # --- Global Extroceptive Features ---
        # Get the observation space from the *base wrapper* to find available global keys
        base_obs_space = self.base_env_ref.observation_space.spaces
        # Keys we might want to pass globally (these are usually present in RoboSuite)
        global_keys_to_pass = ['object-state']#, 'robot0_eef_pos', 'robot0_eef_quat']
        # Also include any specifically requested keys from cfg
        global_keys_to_pass.extend(cfg.ROBOSUITE.get("EXTERO_KEYS", []))
        global_keys_to_pass = list(set(global_keys_to_pass)) # Unique keys

        for key in global_keys_to_pass:
            if key in base_obs_space:
                 spec = base_obs_space[key]
                 if hasattr(spec, 'shape') and np.prod(spec.shape) > 0:
                     dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
                     obs_spaces[key] = Box(-inf, inf, spec.shape, dtype)
                 else:
                      print(f"[NodeCentricObs] Warning: Global key '{key}' has zero size or invalid spec, omitting.")
            # else:
            #      print(f"[NodeCentricObs] Warning: Global key '{key}' requested but not found in base observation space.")

        return Dict(obs_spaces)

    def _extract_features_per_node(self, obs_dict):
        """Extracts features for each node based on defined config."""
        # Reset pre-allocated arrays
        self.node_proprio.fill(0)
        self.node_context.fill(0)

        # Get robot state keys dynamically (handle potential variations)
        joint_pos_cos_key = f'robot0_joint_pos_cos' # TODO: Adapt for multi-robot (robot{idx}_...)
        joint_pos_sin_key = f'robot0_joint_pos_sin'
        joint_vel_key = f'robot0_joint_vel'
        eef_pos_key = f'robot0_eef_pos'
        eef_quat_key = f'robot0_eef_quat'
        gripper_qpos_key = f'robot0_gripper_qpos'
        gripper_qvel_key = f'robot0_gripper_qvel'
        # Add more keys if needed based on proprio_feature_config

        # --- Extract Dynamic Data Once ---
        joint_pos_cos = obs_dict.get(joint_pos_cos_key)
        joint_pos_sin = obs_dict.get(joint_pos_sin_key)
        joint_vel = obs_dict.get(joint_vel_key)
        eef_pos = obs_dict.get(eef_pos_key)
        eef_quat = obs_dict.get(eef_quat_key)
        gripper_qpos = obs_dict.get(gripper_qpos_key)
        gripper_qvel = obs_dict.get(gripper_qvel_key)
        # Get body poses/vels if needed (might require adding to RobosuiteEnvWrapper._convert_observation)
        # body_pos_world = obs_dict.get('all_body_pos_world') # Example key needed
        # body_quat_world = obs_dict.get('all_body_quat_world')
        # body_velp_world = obs_dict.get('all_body_velp_world')
        # body_velr_world = obs_dict.get('all_body_velr_world')


        # --- Iterate through Nodes ---
        for node_idx in range(self.num_nodes):
            node_type = self.node_types[node_idx]
            is_conceptual_gripper = self.node_body_names[node_idx].endswith("_gripper_conceptual")
            body_name = self.node_body_names[node_idx] if not is_conceptual_gripper else self.gripper_node_anchor_name
            mujoco_body_id = self.model.body_name2id(body_name)

            proprio_features = []
            context_features = []

            # --- A. Extract Proprioceptive Features (Dynamic) ---
            config = self.proprio_feature_config[node_type]
            for key, expected_dim in config:
                feature = None
                # --- Handle different keys ---
                if key == 'fixed_encoding': feature = np.array([1.0, 0.0, 0.0]) # Example
                elif key == 'joint_pos_cos': feature = joint_pos_cos[node_idx - 1:node_idx] if joint_pos_cos is not None and node_idx > 0 and (node_idx-1) < len(joint_pos_cos) else np.zeros(1)
                elif key == 'joint_pos_sin': feature = joint_pos_sin[node_idx - 1:node_idx] if joint_pos_sin is not None and node_idx > 0 and (node_idx-1) < len(joint_pos_sin) else np.zeros(1)
                elif key == 'joint_vel': feature = joint_vel[node_idx - 1:node_idx] if joint_vel is not None and node_idx > 0 and (node_idx-1) < len(joint_vel) else np.zeros(1)
                elif key == 'eef_pos': feature = eef_pos if eef_pos is not None else np.zeros(3)
                elif key == 'eef_quat': feature = eef_quat if eef_quat is not None else np.array([1.,0.,0.,0.])
                elif key == 'gripper_qpos': feature = gripper_qpos if gripper_qpos is not None else np.zeros(cfg.ROBOSUITE.GRIPPER_DIM)
                elif key == 'gripper_qvel': feature = gripper_qvel if gripper_qvel is not None else np.zeros(cfg.ROBOSUITE.GRIPPER_DIM)
                # Add world body states if needed (requires modification to RobosuiteEnvWrapper)
                # elif key == 'body_pos_world': feature = body_pos_world[mujoco_body_id] if body_pos_world is not None else np.zeros(3)
                # ... etc ...
                else: feature = np.zeros(expected_dim) # Default zero padding if key logic is missing

                # Verify and append
                feature = np.asarray(feature).flatten()
                if feature.size != expected_dim:
                     # Pad or truncate feature if dimension mismatch
                     actual_dim = feature.size
                     corrected_feature = np.zeros(expected_dim, dtype=np.float32)
                     copy_len = min(actual_dim, expected_dim)
                     corrected_feature[:copy_len] = feature[:copy_len]
                     feature = corrected_feature
                     # print(f"Warning: Node {node_idx} ({body_name}), Proprio Key '{key}': Dim mismatch (Expected {expected_dim}, Got {actual_dim}). Corrected.")
                proprio_features.append(feature)
            # Concatenate and pad the whole node feature vector
            node_proprio_vec = np.concatenate(proprio_features).astype(np.float32)
            if node_proprio_vec.size < self.limb_obs_size:
                padding = np.zeros(self.limb_obs_size - node_proprio_vec.size, dtype=np.float32)
                node_proprio_vec = np.concatenate([node_proprio_vec, padding])
            self.node_proprio[node_idx, :] = node_proprio_vec[:self.limb_obs_size] # Truncate just in case

            # --- B. Extract Context Features (Static) ---
            config = self.context_feature_config[node_type]
            # Define node type one-hot encoding
            type_encoding = {'base': 0, 'arm': 1, 'hand': 2, 'gripper': 3}
            node_type_one_hot = np.zeros(4)
            node_type_one_hot[type_encoding[node_type]] = 1.0

            # Get associated MuJoCo joint IDs (if any) for this node
            mujoco_joint_ids = self.node_idx_to_mujoco_joint_ids.get(node_idx, [])
            # Use first joint for properties if multiple drive the body (simplification)
            # For gripper, use the gripper joints from robot metadata
            joint_id_for_context = mujoco_joint_ids[0] if mujoco_joint_ids else None
            if node_type == 'gripper':
                 # Find gripper joint IDs from robot metadata
                 robot_model_instance = self.base_env_ref.env.robots[0]
                 gripper_jnt_names = robot_model_instance.gripper.joints
                 if gripper_jnt_names:
                      joint_id_for_context = [self.model.joint_name2id(j) for j in gripper_jnt_names]
                 else: joint_id_for_context = None # No specific joint

            for key, expected_dim in config:
                feature = None
                # --- Handle different keys ---
                if key == 'node_type_encoding': feature = node_type_one_hot
                elif key == 'is_fixed': feature = np.array([1.0]) # Assume fixed base for now
                # --- Joint Properties ---
                elif key == 'joint_type':
                     j_type = self.model.jnt_type[joint_id_for_context[0]] if isinstance(joint_id_for_context, list) and joint_id_for_context else (self.model.jnt_type[joint_id_for_context] if joint_id_for_context is not None else -1)
                     feature = np.array([j_type])
                elif key == 'joint_limits':
                     # Handle single vs multiple joints (for gripper)
                     if isinstance(joint_id_for_context, list): # Gripper with potentially >1 DoF
                          limits = np.concatenate([self.model.jnt_range[j_id] for j_id in joint_id_for_context]) if joint_id_for_context else np.zeros(2*cfg.ROBOSUITE.GRIPPER_DIM)
                     elif joint_id_for_context is not None:
                          limits = self.model.jnt_range[joint_id_for_context]
                     else: limits = np.zeros(2)
                     feature = limits
                elif key == 'joint_damping' or key == 'joint_frictionloss' or key == 'joint_armature':
                    # These are DoF properties, need DoF ID
                    dof_id = None
                    if isinstance(joint_id_for_context, list): # Gripper
                        if joint_id_for_context: dof_id = self.model.jnt_dofadr[joint_id_for_context[0]] # Use first DoF
                    elif joint_id_for_context is not None:
                        dof_id = self.model.jnt_dofadr[joint_id_for_context]

                    if dof_id is not None:
                        if key == 'joint_damping': feature = self.model.dof_damping[dof_id:dof_id+expected_dim]
                        elif key == 'joint_frictionloss': feature = self.model.dof_frictionloss[dof_id:dof_id+expected_dim]
                        elif key == 'joint_armature': feature = self.model.dof_armature[dof_id:dof_id+expected_dim]
                    else: feature = np.zeros(expected_dim) # Default if no DoF
                # --- Body Properties ---
                elif key == 'body_mass': feature = np.array([self.model.body_mass[mujoco_body_id]])
                elif key == 'body_inertia_diag': feature = self.model.body_inertia[mujoco_body_id] # Diagonal elements
                elif key == 'geom_type' or key == 'geom_size':
                     # Find first geom attached to this body
                     geom_ids = [g_id for g_id, b_id in enumerate(self.model.geom_bodyid) if b_id == mujoco_body_id]
                     if geom_ids:
                          g_id = geom_ids[0]
                          if key == 'geom_type': feature = np.array([self.model.geom_type[g_id]])
                          elif key == 'geom_size': feature = self.model.geom_size[g_id] # Size (up to 3 params)
                     else: feature = np.zeros(1 if key == 'geom_type' else 3) # Default if no geom
                else: feature = np.zeros(expected_dim) # Default

                # Verify and append
                feature = np.asarray(feature).flatten()
                if feature.size != expected_dim:
                     actual_dim = feature.size
                     corrected_feature = np.zeros(expected_dim, dtype=np.float32)
                     copy_len = min(actual_dim, expected_dim)
                     corrected_feature[:copy_len] = feature[:copy_len]
                     feature = corrected_feature
                     # print(f"Warning: Node {node_idx} ({body_name}), Context Key '{key}': Dim mismatch (Expected {expected_dim}, Got {actual_dim}). Corrected.")
                context_features.append(feature)
            # Concatenate and pad the whole node feature vector
            node_context_vec = np.concatenate(context_features).astype(np.float32)
            if node_context_vec.size < self.context_obs_size:
                padding = np.zeros(self.context_obs_size - node_context_vec.size, dtype=np.float32)
                node_context_vec = np.concatenate([node_context_vec, padding])
            self.node_context[node_idx, :] = node_context_vec[:self.context_obs_size] # Truncate


        return self.node_proprio, self.node_context

    def observation(self, obs):
        """Processes the observation dictionary into the final node-centric format."""
        # Extract node-centric features
        node_proprio, node_context = self._extract_features_per_node(obs)

        # --- Assemble the Final Observation Dictionary ---
        final_obs = OrderedDict()
        # Store FLATTENED node features
        final_obs['proprioceptive'] = node_proprio.flatten()
        final_obs['context'] = node_context.flatten()
        final_obs['edges'] = self.edges_padded
        final_obs['obs_padding_mask'] = self.obs_padding_mask
        final_obs['act_padding_mask'] = self.act_padding_mask

        # --- Pass Through Global Features ---
        for key in self.observation_space.spaces:
            if key not in final_obs and key in obs:
                # Check if the shape matches the defined space (important if base obs changes)
                if obs[key].shape == self.observation_space[key].shape:
                     final_obs[key] = obs[key]
                # else:
                #      print(f"Warning: Shape mismatch for global key '{key}'. Env: {obs[key].shape}, Space: {self.observation_space[key].shape}. Skipping.")

        # Ensure all keys defined in the space are present
        for key in self.observation_space.spaces:
             if key not in final_obs:
                  print(f"Warning: Key '{key}' defined in observation space but missing from final observation dict. Adding zeros.")
                  final_obs[key] = np.zeros(self.observation_space[key].shape, dtype=self.observation_space[key].dtype)


        return final_obs

    def reset(self, **kwargs):
        """Resets the environment and processes the initial observation."""
        observation = self.env.reset(**kwargs)
        # Re-sync metadata and potentially rebuild structures if needed
        # (For ST-SR, structure is fixed, so just process observation)
        # self.robot_metadata = self.base_env_ref.metadata.get('robot_metadata', {}) # Refresh if needed
        # self._define_node_structure() # Re-run if structure could change
        # self._create_kinematic_structure()
        # self._create_padding_masks()
        self._init_feature_arrays() # Reset internal arrays
        return self.observation(observation)


# === Node-Centric Action Wrapper ===
class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Maps the policy's padded action output (per-node) to the base
    RoboSuite environment's action space.
    """
    def __init__(self, env):
        super().__init__(env)
        print("[Wrapper Init] RobosuiteNodeCentricAction")
        # --- Get References and Metadata ---
        self.base_env_ref = self._get_base_wrapper_ref(RobosuiteEnvWrapper)
        self.obs_wrapper_ref = self._get_base_wrapper_ref(RobosuiteNodeCentricObservation)
        self.base_action_space = self.base_env_ref.action_space
        self.robot_metadata = self.base_env_ref.metadata['robot_metadata']
        self.max_limbs = cfg.MODEL.MAX_LIMBS

        # Action dimensions from metadata (set by RobosuiteEnvWrapper)
        self.num_arm_actions_expected = self.robot_metadata.get('num_arm_joints', 0)
        self.num_gripper_actions_expected = self.robot_metadata.get('num_gripper_joints', 0) # Conceptual DoF
        self.real_action_dim = self.base_action_space.shape[0]

        # Verify consistency
        if self.num_arm_actions_expected + self.num_gripper_actions_expected != self.real_action_dim:
             print(f"[ActionWrapper] Warning: Metadata action dims ({self.num_arm_actions_expected}+{self.num_gripper_actions_expected}) != base action space dim ({self.real_action_dim}). Trusting base space.")
             # Heuristic: Assume difference is gripper if arm count seems right based on common robots
             if self.num_arm_actions_expected in [6, 7]: # Sawyer, Panda
                  self.num_gripper_actions_expected = self.real_action_dim - self.num_arm_actions_expected
             else: # Fallback: assume all are arm actions if unsure
                  self.num_arm_actions_expected = self.real_action_dim
                  self.num_gripper_actions_expected = 0


        # --- Define Padded Action Space (What the Policy Sees/Outputs) ---
        # Policy outputs one value per node slot.
        self.padded_action_dim = self.max_limbs
        low = -1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        high = 1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        # This wrapper CHANGES the action space the agent interacts with
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # Store indices from metadata (set by obs wrapper) for easier mapping
        self._update_indices_from_metadata()


    def _get_base_wrapper_ref(self, base_wrapper_type):
        """Finds the instance of base_wrapper_type in the wrapper stack."""
        # Needs to access the *immediately wrapping* env to get metadata reliably
        env = self.env
        while not isinstance(env, base_wrapper_type):
             if not hasattr(env, 'env'):
                  raise TypeError(f"Could not find {base_wrapper_type.__name__} in parent wrapper stack.")
             env = env.env
        return env

    def _update_indices_from_metadata(self):
        """Gets the latest action mask and indices from the obs wrapper metadata."""
        if not hasattr(self.obs_wrapper_ref, 'metadata'):
             raise ValueError("Action wrapper cannot find metadata in observation wrapper.")

        self.act_padding_mask = self.obs_wrapper_ref.metadata.get('act_padding_mask')
        self.arm_action_node_indices = self.obs_wrapper_ref.metadata.get('arm_action_node_indices')
        self.gripper_action_node_indices = self.obs_wrapper_ref.metadata.get('gripper_action_node_indices')

        if self.act_padding_mask is None or self.arm_action_node_indices is None or self.gripper_action_node_indices is None:
            raise ValueError("Action relevant metadata missing from observation wrapper. Ensure Observation wrapper runs first and sets metadata.")

        # Verify mask length
        if len(self.act_padding_mask) != self.padded_action_dim:
             raise ValueError(f"Action mask length ({len(self.act_padding_mask)}) doesn't match padded dim ({self.padded_action_dim}).")

    def action(self, action):
        """Maps the policy's padded action to the base environment's action space."""
        action = np.asarray(action) # Ensure numpy array

        # --- 1. Check Input Shape ---
        if action.shape[-1] != self.padded_action_dim:
             raise ValueError(f"Received action shape {action.shape} doesn't match expected padded dimension {self.padded_action_dim}")

        # --- 2. Refetch Mask/Indices (might change on reset in multi-robot case) ---
        self._update_indices_from_metadata()

        # --- 3. Select Valid Actions using Mask ---
        # Mask is True where action should be IGNORED
        valid_action_values = action[~self.act_padding_mask]

        # --- 4. Separate Arm and Gripper Actions ---
        # We need to know how many arm vs gripper actions are in `valid_action_values`.
        # Assumes the order in valid_action_values corresponds to the order
        # of False entries in act_padding_mask, which should match the node order
        # defined in the observation wrapper.
        num_valid_arm = len(self.arm_action_node_indices)
        num_valid_gripper = len(self.gripper_action_node_indices) # Should be 1

        if len(valid_action_values) != (num_valid_arm + num_valid_gripper):
            print(f"ERROR: Number of valid actions extracted ({len(valid_action_values)}) doesn't match expected arm({num_valid_arm}) + gripper({num_valid_gripper}).")
            # Fallback: Assume it's mostly arm actions and maybe a gripper
            if len(valid_action_values) >= self.num_arm_actions_expected:
                arm_actions = valid_action_values[:self.num_arm_actions_expected]
                if len(valid_action_values) > self.num_arm_actions_expected and self.num_gripper_actions_expected > 0:
                    gripper_action_raw = valid_action_values[self.num_arm_actions_expected : self.num_arm_actions_expected + self.num_gripper_actions_expected]
                else:
                     gripper_action_raw = np.ones(self.num_gripper_actions_expected) # Default fallback
            else: # Not enough actions extracted, severe error
                 print("Fallback failed, using zero action.")
                 return np.zeros_like(self.base_action_space.low)

        else:
            # Assuming order is arm nodes then gripper node
            arm_actions = valid_action_values[:num_valid_arm]
            gripper_action_raw = valid_action_values[num_valid_arm:]

        # --- 5. Process Gripper Action ---
        # Translate the single conceptual value [-1, 1] if needed
        # For many grippers, directly using the value is fine (e.g., Panda, Robotiq in position mode)
        gripper_action_final = np.clip(gripper_action_raw, -1.0, 1.0)
        # Ensure it has the correct dimension expected by the base env
        if gripper_action_final.size != self.num_gripper_actions_expected:
             # Simple case: repeat the single value if base env expects more (e.g., for 2-finger symmetric gripper)
             if self.num_gripper_actions_expected > 0 and gripper_action_final.size == 1:
                 gripper_action_final = np.repeat(gripper_action_final, self.num_gripper_actions_expected)
             else: # More complex mismatch, use default
                  print(f"Warning: Gripper action size mismatch ({gripper_action_final.size} vs {self.num_gripper_actions_expected}). Using default.")
                  gripper_action_final = np.ones(self.num_gripper_actions_expected)

        # --- 6. Combine and Clip ---
        # Ensure arm actions match expected dimension
        if len(arm_actions) != self.num_arm_actions_expected:
             print(f"Warning: Arm action size mismatch ({len(arm_actions)} vs {self.num_arm_actions_expected}). Padding/truncating.")
             corrected_arm_actions = np.zeros(self.num_arm_actions_expected, dtype=np.float32)
             copy_len = min(len(arm_actions), self.num_arm_actions_expected)
             corrected_arm_actions[:copy_len] = arm_actions[:copy_len]
             arm_actions = corrected_arm_actions

        # Concatenate in the order expected by the base controller
        # Usually [arm_joint_actions..., gripper_joint_actions...]
        final_action = np.concatenate([arm_actions, gripper_action_final])

        # Final check on dimension
        if final_action.shape[0] != self.real_action_dim:
             print(f"ERROR: Final combined action dim ({final_action.shape[0]}) != base env dim ({self.real_action_dim}). Clipping to base space shape.")
             # Pad/truncate to avoid crashing downstream
             corrected_final_action = np.zeros(self.real_action_dim, dtype=np.float32)
             copy_len = min(len(final_action), self.real_action_dim)
             corrected_final_action[:copy_len] = final_action[:copy_len]
             final_action = corrected_final_action

        # Clip to the base environment's bounds
        clipped_action = np.clip(final_action, self.base_action_space.low, self.base_action_space.high)

        return clipped_action.astype(np.float32)

# --- debugging ---
if __name__ == "__main__":
    print("--- Testing Robosuite Wrappers ---")
    # Configure for Panda Lift (as an example)
    cfg.defrost()
    cfg.ENV_NAME = "Robosuite-v0"
    cfg.ROBOSUITE.ENV_NAME = "Lift"
    cfg.ROBOSUITE.ROBOTS = ["Panda"]
    cfg.ROBOSUITE.CONTROLLER =  "JOINT_VELOCITY"#"OSC_POSE" # Example controller
    cfg.MODEL.MAX_LIMBS = 11 # Ensure sufficient for Base+7Links+Hand+GripperNode=10
    cfg.MODEL.MAX_JOINTS = 10 # Ensure sufficient for 7 arm + 1 gripper = 8 < 10
    cfg.ROBOSUITE.GRIPPER_DIM = 1 # Panda gripper control dimension
    cfg.freeze()

    # 1. Create Base Environment
    base_env = RobosuiteEnvWrapper(
        robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
        robot_name=cfg.ROBOSUITE.ROBOTS[0],
        controller_name=cfg.ROBOSUITE.CONTROLLER,
        robosuite_cfg=cfg.ROBOSUITE,
        horizon=50 # Short horizon for testing
    )
    print("\n[Base Env]")
    print("Obs Space:", base_env.observation_space)
    print("Act Space:", base_env.action_space)

    # 2. Wrap with Observation Wrapper
    obs_wrapped_env = RobosuiteNodeCentricObservation(base_env)
    print("\n[Obs Wrapped Env]")
    print("Obs Space:", obs_wrapped_env.observation_space)
    print("Act Space:", obs_wrapped_env.action_space) # Should be same as base

    # 3. Wrap with Action Wrapper
    act_wrapped_env = RobosuiteNodeCentricAction(obs_wrapped_env)
    print("\n[Action Wrapped Env]")
    print("Obs Space:", act_wrapped_env.observation_space) # Should be same as obs_wrapped
    print("Act Space:", act_wrapped_env.action_space) # Should now be padded (MAX_LIMBS,)

    # 4. Test Reset
    print("\n--- Testing Reset ---")
    obs = act_wrapped_env.reset()
    print("Reset Obs Keys:", list(obs.keys()))
    for key, val in obs.items():
        expected_shape = act_wrapped_env.observation_space[key].shape
        print(f"  {key}: Shape={val.shape}, Expected={expected_shape}, Dtype={val.dtype}")
        assert val.shape == expected_shape, f"Shape mismatch for {key}"
        assert val.dtype == act_wrapped_env.observation_space[key].dtype, f"Dtype mismatch for {key}"

    # 5. Test Step
    print("\n--- Testing Step ---")
    # Sample action from the *final wrapped* space
    action_from_policy = act_wrapped_env.action_space.sample()
    print("Action from Policy (Padded):", action_from_policy.shape, action_from_policy)
    # The action wrapper will internally convert this before passing to base_env
    next_obs, reward, done, info = act_wrapped_env.step(action_from_policy)
    print("Step Result:")
    print("  Reward:", reward)
    print("  Done:", done)
    print("  Info Keys:", list(info.keys()))
    print("  Next Obs Keys:", list(next_obs.keys()))
    # Check next_obs shapes/dtypes again
    for key, val in next_obs.items():
        expected_shape = act_wrapped_env.observation_space[key].shape
        assert val.shape == expected_shape, f"Shape mismatch for {key} after step"
        assert val.dtype == act_wrapped_env.observation_space[key].dtype, f"Dtype mismatch for {key} after step"

    print("\n--- Wrapper Test Complete ---")
    act_wrapped_env.close()