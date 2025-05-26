import gym 
from gym import spaces
from gym.spaces import Box, Dict

import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

import numpy as np 
from collections import OrderedDict, defaultdict 
import time 
import os

# from metamorph.utils import robosuite_utils as ru # placeholder
from metamorph.utils import swat 
# from metamorph.utils import mjpy 
from metamorph.utils import file as fu

from metamorph.config import cfg 
from metamorph.config import get_list_cfg

# ---------------------------------------------------------
# -----------------Base Wrapper----------------------------
# ---------------------------------------------------------
class RobosuiteEnvWrapper(gym.Env):
    """Making robosuite Env compatible with gym.Env"""
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, env_name , robot_names, controller_names, horizon, robosuite_args=None): 
        """
        Args:
            env_name: (str), name of the robosuite env (e.g. Lift, TwoArmLift)
            robot_names: (str) or (list of str), name of the robots (e.g. "Panda" or ["Panda", "Sawyer"])
            controller_names: (str) or (list of str), name of the controllers (e.g. "OSC_POSE" or ["OSC_POSE", "OSC_POSE"])
            horizon: (int), horizon of the env
            robosuite_args: (dict), arguments for robosuite
        """
        super().__init__() 

        self.env_name = env_name
        self.robot_names = get_list_cfg(robot_names)  # for instance
        self.controller_names = get_list_cfg(controller_names) 
        self._robosuite_args = robosuite_args 
        self._passed_horizon = horizon 

        self.controller_configs = [] # loaded controller configs
        self.loaded_controller_names = [] # names of loaded controllers

        # Those two cases solved by passing correct values in cfg
        if len(self.controller_names) == 1 and len(self.robot_names) > 1:
            # could cause a problem for TwoArmEnv
            self.controller_names = self.controller_names * len(self.robot_names) 

        elif len(self.controller_names) != len(self.robot_names):
            raise ValueError(f"[RobosuiteEnvWrapper] Number of robots and controllers must be the same, got\
                {len(self.robot_names)} robots and {len(self.controller_names)} controllers")

        for c in self.controller_names:
            if c not in ALL_CONTROLLERS:
                print(f"[RobosuiteEnvWrapper] Controller {c} not found, using `JOINT_VELOCITY` instead")
                c = "JOINT_VELOCITY" 
            
            self.controller_configs.append(load_controller_config(default_controller=c))
            self.loaded_controller_names.append(c)
        
        robosuite_init_args = {
            "env_name": self.env_name,
            "robots": self.robot_names,
            "controller_configs": self.controller_configs,
            "horizon": self._passed_horizon,
            **self._robosuite_args # passed from cfg 
        }
        # print(f"[RobosuiteEnvWrapper] Initializing robosuite env with args: {robosuite_init_args}")
        try:
            self.env = robosuite.make(**robosuite_init_args) 
            self.num_robots = len(self.env.robots)

            # check with the instance robot names 
            if len(self.robot_names) != self.num_robots:
                print(f"[RobosuiteEnvWrapper] Number of robots in the env ({self.num_robots}) does not match the number of robots passed ({len(self.robot_names)})")

        except Exception as e:
            raise ValueError(f"[RobosuiteEnvWrapper] Error initializing robosuite env: {e}")

        self.horizon = self.env.horizon 
        self._max_episode_steps = self.horizon # for TimeLimit wrapper

        act_low, act_high = self.env.action_spec 
        self.action_space = Box(act_low, act_high, dtype=np.float32)
        
        obs_spec = self.env.observation_spec()
        gym_obs_spaces = OrderedDict()

        for key, spec in obs_spec.items():
            if hasattr(spec, "shape") and hasattr(spec, "dtype"):
                shape_ = tuple(spec.shape)
                dtype_ = np.float32 if spec.dtype == np.float64 else spec.dtype
                shape_ = (1,) if not shape_ else shape_ # for scalar values 
                gym_obs_spaces[key] = Box(low=-np.inf, high=np.inf, shape=shape_, dtype=dtype_)  

        self.observation_space = Dict(gym_obs_spaces)

        # Metadata per robot 
        self.metadata_per_robot = self._extract_robot_metadata()    
        self.action_dim_per_robot = [md["action_dim"] for md in self.metadata_per_robot]
        self.total_action_dim = sum(self.action_dim_per_robot) # this what the env expects
        self.metadata["robots_metadata_list"] = self.metadata_per_robot # to store in Gym metadata define above 

        if self.total_action_dim != self.action_space.shape[0]:
            print(f"[RobosuiteEnvWrapper] Number of actions per robot must be the same,\
                got {self.total_action_dim} and {self.action_space.shape[0]}")
        
        self._elapsed_steps = 0 
        self.sim = self.env.sim # We will use this for the next wrappers 
        
        self.closed = False

    def _extract_robot_metadata(self):
        metadata_list = []

        if not hasattr(self.env, "robots") or not self.env.robots: 
            print(f"[RobosuiteEnvWrapper] No robots found in the env, cannot extract robot metadata")
            return metadata_list
        
        for robot_idx, robot in enumerate(self.env.robots):
            metadata = {}
            try: 
                # TODO: (optional)
                # - Add identifier for each robot in TwoArmEnv 
                # - e.g. metadata[f"robot{robot_idx}_name"] = robot.name
                metadata["robot_name"] = robot.name
                # robot.robot_joints gets the original joint names without any attachment
                # unlinke robot.dof which retruns the joint names with attachments (gripper)
                metadata["num_arm_joints"] = len(robot.robot_joints) 
                 # Mostly returns 2, but we will deal with gripper as 1, e.g. close/open 
                metadata["num_gripper_joints"] = len(robot.gripper.joints)
                # Nodes = base(1) + n_jnts + gripper(1) 
                # TODO: 
                # - probably will mask the base
                # - The number of actions (action_dim) changes by controller, 
                # this will affect how the transfomer outputs actions, 
                # i.e. if its `osc_pose` we will deal with eef actions + gripper aka (6+1 dof)
                # we control eef_pos and eef_quat. 
                # - We need to either settle on speciic controller or try and fix the logic. 
                metadata["num_nodes"] = 1 + metadata["num_arm_joints"] + 1 + 1 # (base + arm + hand + gripper)
                # This depends on the controller)
                metadata["action_dim"] = robot.action_dim 
                metadata["robot_joints"] = list(robot.robot_joints)
                metadata["gripper_joints"] = list(robot.gripper.joints)
                metadata_list.append(metadata)
            
            except Exception as e:
                print(f"[RobosuiteEnvWrapper] Error extracting robot {robot.name} metadata: {e}")
                
                # dummy metadata 
                metadata_list.append({
                    "robot_name": robot.name,
                    "num_arm_joints": 0,
                    "num_gripper_joints": 0,
                    "num_nodes": 0,
                    "action_dim": 0,
                    "robot_joints": [],
                    "gripper_joints": []
                })

        return metadata_list

    def _convert_observation(self, obs_dict):
        """To convert the observation from robosuite to gym format during reset and step methods"""
        processed_obs = OrderedDict()

        for key, space in self.observation_space.spaces.items():
            if key in obs_dict:
                value = obs_dict[key]
                if isinstance(value, np.ndarray): 
                    dtype_ = np.float32 if value.dtype == np.float64 else value.dtype
                    curr_shape = tuple(value.shape)
                    target_shape = space.shape 
                    if not curr_shape and target_shape == (1,):
                        value = value.reshape(1) 
                        curr_shape = value.shape 
                    
                    if curr_shape == target_shape:
                        processed_obs[key] = value.astype(dtype_)
                    else: 
                        print(f"[RobosuiteEnvWrapper] Observation shape mismatch for key {key}, got {curr_shape} but expected {target_shape}")
                
                else: 
                    try:
                        processed_obs[key] = space.dtype.type(value) 
                    except: 
                        print(f"[RobosuiteEnvWrapper] Error converting observation {key} to gym format: {value}")
                        processed_obs[key] = value # keep oiginal value
            # else: masked in the next wrappers


        # Important because metadata will change during reset for Multi-morph
        processed_obs["robots_metadata_list"] = self.metadata_per_robot

        return processed_obs


    def step(self, action): 
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        obs, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1 

        # # robosuite and timelimt wrapper will handle this, but to be extra safe
        # if self._elapsed_steps >= self.horizon:
        #     done = True
        #     info["TimeLimit.truncated"] = True # std Gym info flag 
        
        processed_obs = self._convert_observation(obs)

        info["robot_names"] = [md["robot_name"] for md in self.metadata_per_robot] 
        info["raw_reward"] = reward 
        info["action"] = action
        info["success"] = self.env._check_success() # returns np.false_ or np.true_ 
        info["name"] = "_".join(info["robot_names"]) + f"_{self.env_name}" # e.g. Panda_Lift
        if done:
            #For RecordEpisodeStatistics wrapper
            info.setdefault("episode", {})
            info["episode"]["success"] = bool(info["success"])
        info["name"] = "_".join(info["robot_names"]) + f"_{self.env_name}" # e.g. Panda_Lift

        return processed_obs, reward, done, info
    
    def reset(self, **kwargs): 
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.seed(seed)
        obs_dict = self.env.reset(**kwargs)
        self.metadata_per_robot = self._extract_robot_metadata() 
        self.metadata["robots_metadata_list"] = self.metadata_per_robot 

        self._elapsed_steps = 0 
        
        processed_obs = self._convert_observation(obs_dict)
        return processed_obs

    def close(self):
        if self.closed:
            return 
        self.env.close() 
    
    def render(self, mode='human', width=512, height=512, camera_name=None):
        if camera_name is None:
             camera_name = getattr(self.env, 'render_camera', "frontview")
             if camera_name not in self.env.sim.model.camera_names and len(self.env.sim.model.camera_names)>0:
                 camera_name = self.env.sim.model.camera_names[0]

        if mode == "rgb_array":
             # Requires has_offscreen_renderer=True or has_renderer=True
             viewer_is_offscreen = getattr(self.env, 'viewer', None) is None and \
                                   self._robosuite_args.get('has_offscreen_renderer', False)
             viewer_is_onscreen = getattr(self.env, 'viewer', None) is not None and \
                                  getattr(self.env.viewer, 'is_running', False)

             if camera_name and (viewer_is_offscreen or viewer_is_onscreen):
                 try:
                     if viewer_is_offscreen:
                          img = self.env.sim.render(camera_name=camera_name, width=width, height=height, depth=False)
                     else: # Read from onscreen viewer
                          self.env.render()
                          img = self.env.viewer.read_pixels(width, height, depth=False)
                     return img[::-1] # Flip vertically
                 except Exception as e:
                     print(f"Warning: Failed rgb_array render cam '{camera_name}'. Error: {e}")
                     return np.zeros((height, width, 3), dtype=np.uint8)
             else:
                 print("Warning: rgb_array requires configured offscreen/onscreen renderer.")
                 return np.zeros((height, width, 3), dtype=np.uint8)

        elif mode == "human":
             if self._robosuite_args.get('has_renderer', False):
                 try: 
                    self.env.render()
                 except Exception as e: 
                    print(f"Error human render: {e}")
             else:
                print("Warning: 'human' mode requires 'has_renderer=True'.")
        else: 
            return super(RobosuiteEnvWrapper, self).render(mode=mode)

    def seed(self, seed=None):
        if hasattr(self.env, 'seed') and callable(self.env.seed):
            self.env.seed(seed)
        self.action_space.seed(seed)
        return [seed] 
    
    def action(self, action):
        return action
# ============================================================
# ============================================================
# ============================================================


# ---------------------------------------------------------
# -----------------SR-ST baseline--------------------------
# -------------------MLP Model-----------------------------
class RobosuiteMLPFlattener(gym.ObservationWrapper):
    """
    Flatten the observation from robosuite (Normally proprioceptive and object states)
    into a single vector. 
    This is only used for the Baseline (SR-10M or SR-fair) as in the paper. 
    """
    def __init__(self, env):
        super().__init__(env)
        # check if env is wrappe by RobosuiteEnvWrapper
        if not isinstance(self.env, RobosuiteEnvWrapper):
            raise ValueError("[RobosuiteMLPFalttener] ObservationWrapper must be wrapped by RobosuiteEnvWrapper")
        self.base_env = self.env 

        base_obs_space = self.base_env.observation_space
        self.action_dim = self.base_env.action_space.shape[0]
        # Mostly for SingleArmEnv will be ['robot0_proprio-state', 'object-state]
        keys_to_flatten = self._get_keys_to_flatten()

        self.max_flat_obs_dim = sum(
            np.prod(base_obs_space.spaces.get(k).shape)
            for k in keys_to_flatten
            if k in base_obs_space.spaces and hasattr(base_obs_space.spaces.get(k), 'shape') and base_obs_space.spaces.get(k).shape is not None
        )

        if self.max_flat_obs_dim == 0: 
            raise ValueError(f"[RobosuiteMLPFalttener] No keys to flatten, check the observation space.")
        # print(f"[RobosuiteMLPFalttener] Flattening {keys_to_flatten} into {self.max_flat_obs_dim} dimensions.")
        # This is only used for dummy obs keys we use in the ac 
        # should be defined in train_ppo 
        self.global_max_limbs = cfg.MODEL.MAX_LIMBS 
        self.global_max_joints = cfg.MODEL.MAX_JOINTS
        # dummy obs keys
        inf = np.float32(np.inf)
        dmsk_obs = (self.global_max_limbs,)
        dmsk_act = (self.action_dim,)
        dedg = (2 * self.global_max_joints,)
        flat_prop_shape = (self.max_flat_obs_dim, )

        self.observation_space = Dict({
            "proprioceptive": Box(-inf, inf, flat_prop_shape, dtype=np.float32),
            "obs_padding_mask": Box(False, True, dmsk_obs, dtype=bool),
            "act_padding_mask": Box(False, True, dmsk_act, dtype=bool),
            "edges": Box(-inf, inf, dedg, dtype=np.float32),
            "context": Box(-inf, inf, flat_prop_shape, dtype=np.float32)
        })

        self.action_space = self.base_env.action_space
    
    def _get_keys_to_flatten(self):
        """Determine which keys to flatten and accounts for SingleArmEnv/TwoArmEnv"""
        num_robots = self.base_env.num_robots
        keys = []

        #TODO: define them in cfg for more cleaning code 
        keys.extend([f"robot{i}_proprio-state" for i in range(num_robots)]) 
        keys.append("object-state")
        # keys.extend(cfg.ROBOSUITE.get("EXTERO_KEYS", [])) # OPTIONAL

        keys = list(set(keys))
        return keys

    def _flatten_and_pad(self, obs_dict):
        """Helper function to flatten and pad the raw observation"""
        pieces = []
        keys_to_use =  self._get_keys_to_flatten()
        for k in keys_to_use:
            if k in obs_dict and isinstance(obs_dict[k], np.ndarray) and obs_dict[k].size > 0:
                pieces.append(obs_dict[k].ravel())
        
        if not pieces:
            print(f"[RobosuiteMLPFalttener] No keys to flatten, check the observation space.")
            return np.zeros(self.max_flat_obs_dim, dtype=np.float32)
        
        flat_vec = np.concatenate(pieces).astype(np.float32)
        current_dim = flat_vec.shape[0]
        
        # pad with zeros
        if current_dim < self.max_flat_obs_dim:
            padded_vec = np.zeros(self.max_flat_obs_dim, dtype=np.float32)
            padded_vec[:current_dim] = flat_vec # copy the flattened vector to the beginning
            return padded_vec
        
        # truncate if too long (Would not happen)
        elif current_dim > self.max_flat_obs_dim:
            print(f"[RobosuiteMLPFalttener] Flattened observation dimension {current_dim} exceeds max dimension {self.max_flat_obs_dim}.")
            return flat_vec[:self.max_flat_obs_dim]
        else:
            return flat_vec

    def observation(self, obs_dict):
        """
        Process the raw observation from the environment
        into final flattend observation plus dummy keys
        """
        flat_padded_prop = self._flatten_and_pad(obs_dict)

        dummy_obs_mask = np.zeros(self.global_max_limbs, dtype=bool)
        dummy_act_mask = np.zeros(self.action_dim, dtype=bool)
        dummy_edges = np.zeros(2 * self.global_max_joints, dtype=np.float32)
        dummy_context = flat_padded_prop.copy()

        return OrderedDict({
            "proprioceptive": flat_padded_prop,
            "obs_padding_mask": dummy_obs_mask,
            "act_padding_mask": dummy_act_mask,
            "edges": dummy_edges,
            "context": dummy_context
        })


    def reset(self, **kwargs):
        obs_dict_raw = self.base_env.reset(**kwargs)
        return self.observation(obs_dict_raw)

    def action(self, action):
        """Pass action from policy to the wrapped env"""
        return action
    
    def close(self):
        self.env.close()
        self.env.closed = True 



# ----------------------------------------------------------
# ---------------Node Centric Observation-------------------
# ---------------------Transormer model---------------------
class RobosuiteNodeCentricObservation(gym.ObservationWrapper):
    """
    Wrap a RobosuiteEnvWrapper to produce fixed‑size, node‑centric observations for a Transformer policy.

    This wrapper takes the per‑robot, per‑body observations emitted by RobosuiteEnvWrapper,
    identifies each “node” (base, each link, end‑effector (hand), conceptual gripper) in a shared global
    index space of size MAX_LIMBS, and then:

      1. Pads out or masks unused slots so every morphology instance yields the same-length vectors.
      2. Builds a fixed‑size “proprioceptive” array of shape (MAX_LIMBS, limb_obs_size) by
         extracting only the features relevant to each node (joint angles/velocities, link poses, etc.)
         and zero‑padding to limb_obs_size.
      3. Builds a matching “context” array of shape (MAX_LIMBS, context_obs_size) by extracting
         static physical properties (link mass, inertia, joint limits, etc.) per node.
      4. Constructs a fixed‑length “edges” list of length 2·MAX_JOINTS encoding the robot’s tree
         connectivity (child, parent pairs), padded with dummy indices.
      5. Emits boolean masks “obs_padding_mask” and “act_padding_mask” of length MAX_LIMBS to
         indicate which node slots are real versus padded, and which nodes have actuators.
      6. TODO: computes and emits “traversals” (pre, in‑order, post indices) and “SWAT_RE”
         structural attention masks for structure‑aware Transformer attention.

    The final observation dict has keys:
        - "proprioceptive": float32 array, shape (MAX_LIMBS * limb_obs_size,)
        - "context":       float32 array, shape (MAX_LIMBS * context_obs_size,)
        - "edges":         float32 array, shape (2 * MAX_JOINTS,)
        - "obs_padding_mask": bool array, shape (MAX_LIMBS,)
        - "act_padding_mask": bool array, shape (MAX_LIMBS,)
        - "traversals":    int32 array, shape (MAX_LIMBS, 3) # see cfg.MODEL.TRANSFORMER.TRAVERSALS
        - "SWAT_RE":       bool array, shape (MAX_LIMBS, MAX_LIMBS) 

        - plus any passthrough keys (e.g. "object-state")

    This fixed‑width, node‑centric format is exactly what the MetaMorph/ModuMorph Transformer
    models expect as input, allowing them to attend over a robot’s morphology graph.
    """

    def __init__(self, env):
        super().__init__(env) 

        if not isinstance(self.env, RobosuiteEnvWrapper):
            raise ValueError("[RobosuiteNodeCentricObservation] requires a RobosuiteEnvWrapper")
        
        self.base_env_ref = self.env 
        self.sim = self.base_env_ref.sim # mj sim instance 
        self.model = self.sim.model # mj data (Table, body, joint, geom, site, actuator)

        self.global_max_limbs = cfg.MODEL.MAX_LIMBS 
        self.global_max_joints = cfg.MODEL.MAX_JOINTS 
        self._define_feat_sets()
        # dim calculation for padding node-level proprio and context features
        self.limb_obs_size = self._calc_feat_dim(self.proprio_feat_cfg)
        self.context_obs_size = self._calc_feat_dim(self.context_feat_cfg)
        
        # should holds the globaly padded obs data
        self._init_global_arrays()

        # Final observation space (globaly padded)
        self.observation_space = self._define_observation_space()
        # print(f"[RobosuiteNodeCentricObservation] observation_space keys: {list(self.observation_space.spaces.keys())}")

        # Flag to track if the structure has been analayzed for the current episode
        self._structure_initialized = False 
        
        try:
            self._define_structure_and_masks()
        except:
            print("[RobosuiteNodeCentricObservation] Failed to initialize structure and masks")
            self._structure_initialized = False
            

    def _init_global_arrays(self):
        """Initialize all the fixed‑size, node‑centric buffers to their padding defaults."""

        self.node_proprio_global = np.zeros((self.global_max_limbs, self.limb_obs_size), dtype=np.float32)
        self.node_context_global = np.zeros((self.global_max_limbs, self.context_obs_size), dtype=np.float32)
        self.edges_padded_global = np.full(2 * self.global_max_joints, self.global_max_limbs - 1, dtype=np.float32)
        self.obs_padding_mask_global = np.ones(self.global_max_limbs, dtype=bool)
        policy_node_output_dim = cfg.MODEL.TRANSFORMER.DECODER_OUT_DIM
        self.act_padding_mask_global = np.ones(self.global_max_limbs * policy_node_output_dim, dtype=bool)
        # we hard code the num_trav as 3 from cfg 
        self.traversals_global = np.full((self.global_max_limbs, 3), self.global_max_limbs - 1, dtype=np.int64)
        self.swat_re_global = np.zeros((self.global_max_limbs, self.global_max_limbs, 3), dtype=np.float32)
        
        self.metadata['per_robot_action_indices'] = [] 
        self.robot_specific_edges = []

    def _define_feat_sets(self):
        """
        Define the feature sets for proprioceptive and context observations per node type.
        Maps those types (base, arm, hand (eef), gripper). to list of (feat_key, expected_dim)
        """
        self.proprio_feat_cfg = {
            'base': [
                ('fixed_encoding', 3),
            ],
            'arm': [
                ('joint_pos_cos', 1),
                ('joint_pos_sin', 1),
                ('joint_vel', 1),
            ],
            'hand': [
                ('eef_pos', 3),
                ('eef_quat', 4),
            ],
            'gripper': [
                ('gripper_qpos', cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
                ('gripper_qvel', cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
            ],
        }
        self.context_feat_cfg = {
            'base': [
                ('node_type_encoding', 4),
                ('is_fixed', 1),
            ],
            'arm': [
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
            'hand': [
                ('node_type_encoding', 4),
                ('joint_type', 1), # Could be fixed joint connecting to arm, or free joint
                ('joint_limits', 2),
                ('joint_damping', 1),
                ('joint_frictionloss', 1),
                ('joint_armature', 1),
                ('body_mass', 1),
                ('body_inertia_diag', 3),
            ],
            'gripper': [
                ('node_type_encoding', 4),
                ('joint_type', 1),
                ('joint_limits', 2 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
                ('joint_damping', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
                ('joint_frictionloss', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
                ('joint_armature', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)),
            ],
        }


    def _calc_feat_dim(self, feat_cfg):
        max_dim = 0
        for node_type, feats in feat_cfg.items():
            current_dim = sum(dim for _, dim in feats)
            max_dim = max(max_dim, current_dim)
        
        return max_dim if max_dim > 0 else 1

    def _define_observation_space(self):

        L = self.global_max_limbs 
        inf = np.float32(np.inf)
        obs_spaces = OrderedDict()
        obs_spaces['proprioceptive'] = Box(-inf, inf, (L * self.limb_obs_size,), np.float32)
        obs_spaces['context'] = Box(-inf, inf, (L * self.context_obs_size,), np.float32)
        obs_spaces['edges'] = Box(-inf, inf, (2 * self.global_max_joints,), np.float32)
        obs_spaces['obs_padding_mask'] = Box(False, True, (L,), dtype=bool)
        policy_node_output_dim = cfg.MODEL.TRANSFORMER.DECODER_OUT_DIM
        obs_spaces['act_padding_mask'] = Box(False, True, (L * policy_node_output_dim,), dtype=bool)
        
        # traversals for SWAT-RE/PE 
        num_traversals = len(cfg.MODEL.TRANSFORMER.TRAVERSALS)
        obs_spaces['traversals'] = Box(-inf, inf, (L, num_traversals), np.int64) 
        obs_spaces['SWAT_RE'] = Box(-inf, inf, (L, L, 3), np.float32) # (PPR, SymLap, NG-distance) from swat 

        base_obs_space = self.base_env_ref.observation_space.spaces
        global_keys_to_pass = ['object-state']
        global_keys_to_pass.extend(cfg.ROBOSUITE.get("EXTERO_KEYS", [])) 
        global_keys_to_pass = list(set(global_keys_to_pass))
        for key in global_keys_to_pass:
            if key in base_obs_space:
                spec = base_obs_space[key]
                shape_ = spec.shape
                dtype_ = np.float32 if spec.dtype == np.float64 else spec.dtype
                if np.prod(shape_) > 0: 
                    obs_spaces[key] = Box(-inf, inf, shape_, dtype_)
        
        return Dict(obs_spaces)


    def _define_structure_and_masks(self):
        """
        Analyzes the current morphology instance's structure and defines padding/masks/graph info.
        This method dynamically builds the graph based on semantic relationships:
        - base node's parent is conceptually "World".
        - Arm links' parents are their direct kinematic parents OR the base node if direct parent is World.
        - Hand node's parent is its direct kinematic parent.
        - Gripper node's parent is explicitly the Hand node.
        """
        self._init_global_arrays() # Reset global padded arrays

        self.num_robots_instance = self.base_env_ref.num_robots
        self.metadata_per_robot_instance = self.base_env_ref.metadata_per_robot 

        self.node_start_indices = [] # Global starting index for nodes of each robot
        current_global_node_offset = 0

        self.nodes_per_robot_instance = [] # List of lists: [(body_name, node_type), ...] for each robot
        self.body_name_to_local_node_idx_per_robot = [] # List of dicts: {body_name: local_node_idx}
        self.gripper_anchor_names_per_robot = [] # List of hand body names (gripper parent)
        self.local_node_idx_to_mujoco_joint_ids_per_robot = [] # List of dicts: {local_node_idx: [mujoco_joint_ids]}
        self.local_parents_list_per_robot = [] # Store the local parent list for SWAT utilities
        self.local_node_to_body_id_per_robot = [] # Map local node index to MuJoCo body ID
        self.robot_specific_edges = [] # (Re-)Initialize list for local edges per robot


        for robot_idx in range(self.num_robots_instance):
            robot_instance = self.base_env_ref.env.robots[robot_idx]
            current_robot_nodes = [] # Nodes for this robot [(body_name, node_type)]
            current_robot_arm_joint_names = self.metadata_per_robot_instance[robot_idx]['robot_joints']
            current_robot_gripper_joint_names = self.metadata_per_robot_instance[robot_idx]['gripper_joints']
            current_robot_body_name_to_local_node_idx = {}
            current_robot_local_node_to_body_id = {} 
            local_node_idx_to_mujoco_joint_ids = defaultdict(list) # Map local node index to list of MuJoCo joint IDs it's associated with (e.g., driving joint)

            # Explicitly define the semantic nodes in their logical order (DFS-like)
            # and map them to their MuJoCo body IDs and local indices.
            
            # Base Node
            base_name = robot_instance.robot_model.root_body
            current_robot_nodes.append((base_name, 'base'))
            current_robot_body_name_to_local_node_idx[base_name] = 0
            current_robot_local_node_to_body_id[0] = self.model.body_name2id(base_name)
            local_node_counter = 1 # Start counter for next nodes
            
            # Arm Links (ordered by robot_joints)
            for j_name in current_robot_arm_joint_names:
                mujoco_body_id = self.model.jnt_bodyid[self.model.joint_name2id(j_name)]
                body_name = self.model.body_id2name(mujoco_body_id)
                # Only add if it's not the base, hand, or gripper (which are added specifically)
                if body_name != base_name and \
                   body_name != robot_instance.robot_model.eef_name and \
                   not any(g_body_name == body_name for g_body_name in robot_instance.gripper.bodies):
                    current_robot_nodes.append((body_name, 'arm'))
                    current_robot_body_name_to_local_node_idx[body_name] = local_node_counter
                    current_robot_local_node_to_body_id[local_node_counter] = mujoco_body_id
                    local_node_idx_to_mujoco_joint_ids[local_node_counter].append(self.model.joint_name2id(j_name))
                    local_node_counter += 1

            # Hand Node (End-effector)
            hand_name = robot_instance.robot_model.eef_name
            if hand_name not in current_robot_body_name_to_local_node_idx: # Ensure it hasn't been added as an arm link by chance
                current_robot_nodes.append((hand_name, 'hand'))
                current_robot_body_name_to_local_node_idx[hand_name] = local_node_counter
                current_robot_local_node_to_body_id[local_node_counter] = self.model.body_name2id(hand_name)
                # Add any joints directly associated with the hand body (e.g., fixed joint to last link, or wrist joint if applicable)
                for j_idx in range(self.model.njnt):
                    if self.model.jnt_bodyid[j_idx] == self.model.body_name2id(hand_name):
                        local_node_idx_to_mujoco_joint_ids[local_node_counter].append(j_idx)
                local_node_counter += 1
            self.gripper_anchor_names_per_robot.append(hand_name) # Store hand name for gripper parentage

            # Gripper Node
            # or could pass just gripper0_eef 
            # gripper_eef_name = "gripper0_eef"
            gripper_eef_name = None
            for g_body_name in robot_instance.gripper.bodies:
                if "gripper" in g_body_name and "eef" in g_body_name: 
                    gripper_eef_name = g_body_name
                    break
            
            if gripper_eef_name and gripper_eef_name not in current_robot_body_name_to_local_node_idx:
                current_robot_nodes.append((gripper_eef_name, 'gripper'))
                current_robot_body_name_to_local_node_idx[gripper_eef_name] = local_node_counter
                current_robot_local_node_to_body_id[local_node_counter] = self.model.body_name2id(gripper_eef_name)
                # Map gripper joint IDs to the gripper node
                for j_name in current_robot_gripper_joint_names:
                    try: 
                        gripper_local_idx = current_robot_body_name_to_local_node_idx[gripper_eef_name] 
                        local_node_idx_to_mujoco_joint_ids[gripper_local_idx].append(self.model.joint_name2id(j_name))
                    except KeyError:
                        print(f"[RobosuiteNodeCentricObservation] Warning: Joint {j_name} not found in the model for gripper node. Skipping.")
                        pass
                local_node_counter += 1 # Increment counter for gripper node

            # Store local structure info for this robot
            local_num_nodes = len(current_robot_nodes)
            self.nodes_per_robot_instance.append(current_robot_nodes)
            self.body_name_to_local_node_idx_per_robot.append(current_robot_body_name_to_local_node_idx)
            self.local_node_idx_to_mujoco_joint_ids_per_robot.append(local_node_idx_to_mujoco_joint_ids)
            self.local_node_to_body_id_per_robot.append(current_robot_local_node_to_body_id) 

            # Build local_parents_list and current_robot_edges based on Semantic Kinematics ---
            local_parents_list = [-1] * local_num_nodes # -1 for root (base)
            current_robot_edges = [] # List of local (child, parent) node index pairs

            for local_child_idx, (child_body_name, child_node_type) in enumerate(current_robot_nodes):
                if child_node_type == 'base':
                    # Base node's parent is conceptual "World", for SWAT we use -1.
                    continue 
                
                elif child_node_type == 'gripper':
                    # Gripper's parent is explicitly the hand node
                    hand_name_eef = robot_instance.robot_model.eef_name
                    if hand_name_eef in current_robot_body_name_to_local_node_idx:
                        parent_local_idx = current_robot_body_name_to_local_node_idx[hand_name_eef]
                        local_parents_list[local_child_idx] = parent_local_idx
                        current_robot_edges.extend([local_child_idx, parent_local_idx])
                    else:
                        print(f"[RobosuiteNodeCentricObservation] Warning: Hand node {hand_name_eef} not found for gripper parent of {child_body_name}. Defaulting to base as parent.")
                        parent_local_idx = 0 # Fallback to base
                        local_parents_list[local_child_idx] = parent_local_idx
                        current_robot_edges.extend([local_child_idx, parent_local_idx])
                
                else: # Arm or Hand link node
                    # Find its direct MuJoCo parent body
                    child_mujoco_id = self.model.body_name2id(child_body_name)
                    parent_mujoco_id = self.model.body_parentid[child_mujoco_id] 
                    parent_body_name = self.model.body_id2name(parent_mujoco_id) 

                    if parent_body_name == 'world': # If MuJoCo parent is the world body
                        parent_local_idx = 0 # Base node (local index 0) is the conceptual parent
                    elif parent_body_name in current_robot_body_name_to_local_node_idx: # If MuJoCo parent is another node in our logical set
                        parent_local_idx = current_robot_body_name_to_local_node_idx[parent_body_name]
                    else:
                        # Fallback: if MuJoCo parent is not in our logical nodes, default to base as parent.
                        # print(f"[RobosuiteNodeCentricObservation] Warning: Parent body {parent_body_name} for {child_body_name} not found in local node map. Defaulting to base as parent.")
                        parent_local_idx = 0 

                    local_parents_list[local_child_idx] = parent_local_idx
                    current_robot_edges.extend([local_child_idx, parent_local_idx])

            # Ensure all nodes (except base) have a valid parent; set to base if disconnected
            # TODO: CHECK FOR THE GRIPPER PARENT IT SHOULD BE HAND
            for i in range(1, local_num_nodes): # Skip root (index 0)
                if local_parents_list[i] == -1:
                    local_parents_list[i] = 0 # Default to base node as parent
                    current_robot_edges.extend([i, 0]) # Add edge to base node

            self.local_parents_list_per_robot.append(local_parents_list) 
            self.robot_specific_edges.append(current_robot_edges) # Store local edges for this robot

            # SWAT/TRAVERSALS
            if local_num_nodes > 0: # Ensure there are nodes to process
                # Critical check: Ensure the local_num_nodes does not exceed cfg.MODEL.MAX_LIMBS
                if local_num_nodes > cfg.MODEL.MAX_LIMBS:
                    raise ValueError(f"[RobosuiteNodeCentricObservation] Error: Robot {robot_instance.name} has {local_num_nodes} logical nodes, but cfg.MODEL.MAX_LIMBS is {cfg.MODEL.MAX_LIMBS}. Increase MAX_LIMBS in config or review node selection.")

                local_traversals = swat.getTraversal(local_parents_list) 
                local_swat_re = swat.getGraphDict(local_parents_list) 
                
                # Copy local results into global padded arrays, respecting cfg.MODEL.MAX_LIMBS
                global_node_start = current_global_node_offset
                global_node_end = global_node_start + local_num_nodes # Actual end in global array
                
                trav_types_count = self.traversals_global.shape[1] 
                re_feat_count = self.swat_re_global.shape[2] 
                
                # Copy traversals
                self.traversals_global[global_node_start:global_node_end, :min(local_traversals.shape[1], trav_types_count)] = \
                    local_traversals[:local_num_nodes, :min(local_traversals.shape[1], trav_types_count)].astype(np.int64)
                
                # Copy SWAT_RE
                self.swat_re_global[global_node_start:global_node_end, global_node_start:global_node_end, :min(local_swat_re.shape[2], re_feat_count)] = \
                    local_swat_re[:local_num_nodes, :local_num_nodes, :min(local_swat_re.shape[2], re_feat_count)].astype(np.float32)
 
            # Update global offset for next robot
            self.node_start_indices.append(current_global_node_offset) 
            current_global_node_offset += local_num_nodes 

            # Unmask observation slots for this robot's real nodes within global padding limits
            global_node_start = self.node_start_indices[robot_idx]
            global_node_end = global_node_start + local_num_nodes
            self.obs_padding_mask_global[global_node_start:min(global_node_end, self.global_max_limbs)] = False

            # Create Action Mask and store indices for Action Wrapper
            robot_action_indices = {'arm': [], 'gripper': []}
            num_arm_joints_in_metadata = self.metadata_per_robot_instance[robot_idx]['num_arm_joints'] 
            num_gripper_joints_in_metadata = self.metadata_per_robot_instance[robot_idx]['num_gripper_joints']
            policy_node_output_dim = cfg.MODEL.TRANSFORMER.DECODER_OUT_DIM

            for local_node_idx, (body_name, node_type) in enumerate(current_robot_nodes):
                 global_node_idx = global_node_start + local_node_idx
                 if global_node_idx >= self.global_max_limbs: 
                    # Skip if outside global bounds
                    continue

                 start_idx_mask = global_node_idx * policy_node_output_dim

                 # JOINT_VELOCITY Controller
                 if self.base_env_ref.controller_names[robot_idx] == 'JOINT_VELOCITY':
                    if node_type == 'arm':
                        # This heuristic attempts to map node index to the joint index.
                        # It relies on the assumption that arm links (excluding base) correspond to arm joints in order.
                        if local_node_idx -1 < num_arm_joints_in_metadata: # local_node_idx 0 is base, so first arm link is 1
                            if (start_idx_mask + 0) < self.act_padding_mask_global.shape[0]:
                                self.act_padding_mask_global[start_idx_mask + 0] = False # Unmask 1st scalar
                            robot_action_indices['arm'].append(global_node_idx) # Store global index for joint actions
                    # elif node_type == 'hand': # Remains masked
                        # pass
                    elif node_type == 'gripper' and num_gripper_joints_in_metadata > 0:
                        for i in range(cfg.ROBOSUITE.GRIPPER_DIM):
                            if (start_idx_mask + i) < self.act_padding_mask_global.shape[0]:
                                self.act_padding_mask_global[start_idx_mask + i] = False # Unmask GRIPPER_DIM scalars
                        robot_action_indices['gripper'].append(global_node_idx)
                 
                 # OSC_POSE Controller
                 elif self.base_env_ref.controller_names[robot_idx] == 'OSC_POSE':
                    # if node_type == 'arm': # Remains masked
                        # pass
                    if node_type == 'hand':
                        for i in range(6): # OSC hand uses 6 dimensions
                            if (start_idx_mask + i) < self.act_padding_mask_global.shape[0]:
                                self.act_padding_mask_global[start_idx_mask + i] = False
                        robot_action_indices['arm'].append(global_node_idx) # (reusing 'arm' for simplicity)
                    elif node_type == 'gripper' and num_gripper_joints_in_metadata > 0:
                        if (start_idx_mask + 0) < self.act_padding_mask_global.shape[0]:
                             self.act_padding_mask_global[start_idx_mask + 0] = False # Unmask 1st scalar
                        robot_action_indices['gripper'].append(global_node_idx)
                 
                 # Fallback: if controller not matched, or node type not handled for masking,
                 # ensure relevant robot_action_indices are still populated if needed,
                 # though their scalars in act_padding_mask_global would remain True (masked).
                 # This part might need refinement based on how RobosuiteNodeCentricAction uses these.
                 # For now, populating robot_action_indices as before for unhandled cases.
                 else:
                    if node_type == 'arm' and local_node_idx -1 < num_arm_joints_in_metadata:
                        robot_action_indices['arm'].append(global_node_idx)
                    elif node_type == 'hand': # OSC_POSE might be default, or other controllers
                        robot_action_indices['arm'].append(global_node_idx)
                    elif node_type == 'gripper' and num_gripper_joints_in_metadata > 0:
                        robot_action_indices['gripper'].append(global_node_idx)


            self.metadata['per_robot_action_indices'].append(robot_action_indices)
            
        # EDGES
        # This should be done once after the loop over robots.
        self.edges_padded_global.fill(self.global_max_limbs - 1) 
        global_edge_idx_counter = 0
        for robot_idx in range(self.num_robots_instance):
            global_node_offset = self.node_start_indices[robot_idx]
            for local_child, local_parent in zip(self.robot_specific_edges[robot_idx][0::2], self.robot_specific_edges[robot_idx][1::2]):
                global_child_idx = global_node_offset + local_child
                global_parent_idx = global_node_offset + local_parent

                # Ensure global indices are within MAX_LIMBS to avoid out-of-bounds
                if global_child_idx >= cfg.MODEL.MAX_LIMBS or global_parent_idx >= cfg.MODEL.MAX_LIMBS:
                    print(f"[RobosuiteNodeCentricObservation] Warning: Generated global edge ({global_child_idx}, {global_parent_idx}) out of bounds for MAX_LIMBS ({cfg.MODEL.MAX_LIMBS}). Skipping.")
                    continue

                if global_edge_idx_counter * 2 + 1 < 2 * self.global_max_joints: # Check if space available for 2 values (child,parent)
                    self.edges_padded_global[global_edge_idx_counter * 2] = global_child_idx
                    self.edges_padded_global[global_edge_idx_counter * 2 + 1] = global_parent_idx
                    global_edge_idx_counter += 1
                else:
                    print(f"[RobosuiteNodeCentricObservation] Warning: Exceeded global_max_joints ({self.global_max_joints}) for edges. Skipping edge ({global_child_idx}, {global_parent_idx}).")
                    break 


        self.metadata['act_padding_mask'] = self.act_padding_mask_global.copy()
        self._structure_initialized = True
 

    def _extract_feat_per_node(self, obs_dict):
        """Extracts and pads proprioceptive and context features for each node."""
        if not hasattr(self, '_structure_initialized') or not self._structure_initialized:
             print("[RobosuiteNodeCentricObservation] Error: Structure not initialized in _extract_features_per_node.")
             # Return zero arrays matching the expected global shape
             return np.zeros((self.global_max_limbs, self.limb_obs_size), dtype=np.float32), \
                    np.zeros((self.global_max_limbs, self.context_obs_size), dtype=np.float32)

        # Reset global arrays for current observation
        self.node_proprio_global.fill(0)
        self.node_context_global.fill(0)

        # Retrieve the per-robot metadata for current robot's joint names etc.
        # This will be available from self.metadata_per_robot_instance populated by RobosuiteEnvWrapper
        
        for robot_idx in range(self.num_robots_instance):
            robot_prefix = f"robot{robot_idx}_"
            # Get robot-specific joint names (needed for feature extraction logic)
            current_robot_arm_joint_names = self.metadata_per_robot_instance[robot_idx]['robot_joints']
            current_robot_gripper_joint_names = self.metadata_per_robot_instance[robot_idx]['gripper_joints']

            # Extract raw observation data from obs_dict for current robot
            joint_pos_cos = obs_dict.get(f'{robot_prefix}joint_pos_cos')
            joint_pos_sin = obs_dict.get(f'{robot_prefix}joint_pos_sin')
            joint_vel = obs_dict.get(f'{robot_prefix}joint_vel')
            eef_pos = obs_dict.get(f'{robot_prefix}eef_pos')
            eef_quat = obs_dict.get(f'{robot_prefix}eef_quat')
            gripper_qpos = obs_dict.get(f'{robot_prefix}gripper_qpos')
            gripper_qvel = obs_dict.get(f'{robot_prefix}gripper_qvel')
            
            # World-frame body observations (if included in config)
            # TODO: ADD THEM TO EEF ONLY
            body_pos_world = obs_dict.get(f'{robot_prefix}body_pos') 
            body_quat_world = obs_dict.get(f'{robot_prefix}body_quat')
            body_velp_world = obs_dict.get(f'{robot_prefix}body_velp')
            body_velr_world = obs_dict.get(f'{robot_prefix}body_velr')

            current_robot_nodes = self.nodes_per_robot_instance[robot_idx]
            local_node_idx_to_mujoco_joint_ids = self.local_node_idx_to_mujoco_joint_ids_per_robot[robot_idx]

            for local_node_idx, (body_name, node_type) in enumerate(current_robot_nodes):
                 global_node_idx = self.node_start_indices[robot_idx] + local_node_idx
                 if global_node_idx >= self.global_max_limbs: 
                    print(f"[RobosuiteNodeCentricObservation] Warning: Node index {global_node_idx} exceeds max limbs {self.global_max_limbs}. Skipping feature extraction for this node.")
                    continue
                 
                 # This check prevents KeyError for mujoco_body_id.
                 mujoco_body_id = -1
                 try: 
                    mujoco_body_id = self.model.body_name2id(body_name)
                 except KeyError: 
                    print(f"[RobosuiteNodeCentricObservation] Warning: Body name {body_name} not found in model. Skipping feature extraction for this node.")
                    continue 
                 
                 proprio_features = []
                 context_features = []
                 
                 # Proprioceptive Features
                 for key, expected_dim in self.proprio_feat_cfg[node_type]:
                     feature = np.zeros(expected_dim, dtype=np.float32)
                     
                     if key == 'fixed_encoding' and node_type == 'base': 
                         feature = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                     elif node_type == 'arm':
                         arm_joint_local_idx_heuristic = local_node_idx -1 
                         if 0 <= arm_joint_local_idx_heuristic < len(current_robot_arm_joint_names):
                             if key == 'joint_pos_cos' and joint_pos_cos is not None:
                                feature = joint_pos_cos[arm_joint_local_idx_heuristic:arm_joint_local_idx_heuristic+1]
                             elif key == 'joint_pos_sin' and joint_pos_sin is not None:
                                feature = joint_pos_sin[arm_joint_local_idx_heuristic:arm_joint_local_idx_heuristic+1]
                             elif key == 'joint_vel' and joint_vel is not None:
                                feature = joint_vel[arm_joint_local_idx_heuristic:arm_joint_local_idx_heuristic+1]
                     elif node_type == 'hand':
                         if key == 'eef_pos' and eef_pos is not None:
                            feature = eef_pos
                         elif key == 'eef_quat' and eef_quat is not None:
                            feature = eef_quat
                     elif node_type == 'gripper':
                         if key == 'gripper_qpos' and gripper_qpos is not None:
                            feature = gripper_qpos
                         elif key == 'gripper_qvel' and gripper_qvel is not None:
                            feature = gripper_qvel
                     
                     # General body-level world-frame features (if included in config)
                     # Check if mujoco_body_id is valid before accessing sim.data
                     if mujoco_body_id != -1:
                        # TODO: ADD THESE ONLY FOR HAND NODE
                        if key == 'body_pos_world' and body_pos_world is not None:
                            feature = body_pos_world[mujoco_body_id]
                        elif key == 'body_quat_world' and body_quat_world is not None:
                            feature = body_quat_world[mujoco_body_id]
                        elif key == 'body_velp_world' and body_velp_world is not None:
                            feature = body_velp_world[mujoco_body_id]
                        elif key == 'body_velr_world' and body_velr_world is not None:
                            feature = body_velr_world[mujoco_body_id]
                     
                     # Ensure feature has correct shape, pad if necessary
                     feature_flat = np.asarray(feature).flatten()
                     if feature_flat.shape[0] != expected_dim:
                         if feature_flat.shape[0] < expected_dim:
                            padded_feature = np.zeros(expected_dim, dtype=np.float32)
                            padded_feature[:feature_flat.shape[0]] = feature_flat
                            feature_flat = padded_feature
                         else: # Truncate if too long
                            feature_flat = feature_flat[:expected_dim]
                            # print(f"[RobosuiteNodeCentricObservation] Warning: Proprioceptive feature '{key}' for {body_name} truncated from {feature_flat.shape[0]} to {expected_dim}.")
                     
                     proprio_features.append(feature_flat)
                 
                 # Concatenate all features for this node and pad/truncate to self.limb_obs_size
                 node_proprio_vec = np.concatenate(proprio_features).astype(np.float32)
                 if node_proprio_vec.size != self.limb_obs_size:
                    padded_prop = np.zeros(self.limb_obs_size, dtype=np.float32)
                    copy_len = min(node_proprio_vec.size, self.limb_obs_size)
                    padded_prop[:copy_len] = node_proprio_vec[:copy_len]
                    node_proprio_vec = padded_prop
                 self.node_proprio_global[global_node_idx, :] = node_proprio_vec
 
                 # Context (Static) Features
                 type_encoding = {'base': 0, 'arm': 1, 'hand': 2, 'gripper': 3}
                 node_type_one_hot = np.zeros(4, dtype=np.float32); node_type_one_hot[type_encoding[node_type]] = 1.0
                 
                 for key, expected_dim in self.context_feat_cfg[node_type]:
                     feature = np.zeros(expected_dim, dtype=np.float32) # Default to zeros
                     
                     if key == 'node_type_encoding': feature = node_type_one_hot
                     elif key == 'is_fixed' and node_type == 'base': feature = np.array([1.0], dtype=np.float32) # Base node is fixed
                     
                     elif key in ['joint_type', 'joint_limits', 'joint_damping', 'joint_frictionloss', 'joint_armature']:
                         mujoco_joint_ids = local_node_idx_to_mujoco_joint_ids.get(local_node_idx, [])
                         if mujoco_joint_ids:
                             j_id0 = mujoco_joint_ids[0] # Use the first associated joint
                             if key == 'joint_type': feature = np.array([self.model.jnt_type[j_id0]], dtype=np.float32)
                             elif key == 'joint_limits': 
                                 if len(mujoco_joint_ids) * 2 == expected_dim: # If multiple joints contribute (e.g. 2 joints for one node)
                                     feature = np.concatenate([self.model.jnt_range[j_id] for j_id in mujoco_joint_ids]).astype(np.float32)
                                 else: # Single joint or default
                                     feature = self.model.jnt_range[j_id0].astype(np.float32)
                             elif key in ['joint_damping', 'joint_frictionloss', 'joint_armature']:
                                 dof_ids = []
                                 # Correctly get DOF count from joint type (e.g., hinge is 1 DOF, ball is 3 DOFs)
                                 dof_count_for_jnt_type = {0: 1, 1: 1, 2: 3, 3: 0} # 0:hinge, 1:slide, 2:ball, 3:free (unused)
                                 for j_id in mujoco_joint_ids:
                                     dof_start = self.model.jnt_dofadr[j_id]
                                     # Get actual dof count for this joint type
                                     dof_num = dof_count_for_jnt_type.get(self.model.jnt_type[j_id], 0) 
                                     if dof_num > 0:
                                        dof_ids.extend(list(range(dof_start, dof_start + dof_num)))
                                 
                                 if dof_ids:
                                     if key == 'joint_damping':
                                        feature = self.model.dof_damping[dof_ids].astype(np.float32)
                                     elif key == 'joint_frictionloss':
                                        feature = self.model.dof_frictionloss[dof_ids].astype(np.float32)
                                     elif key == 'joint_armature':
                                        feature = self.model.dof_armature[dof_ids].astype(np.float32)
                     elif key == 'body_mass' and mujoco_body_id != -1:
                        feature = np.array([self.model.body_mass[mujoco_body_id]], dtype=np.float32)
                     elif key == 'body_inertia_diag' and mujoco_body_id != -1:
                        feature = self.model.body_inertia[mujoco_body_id].astype(np.float32)
                     elif key in ['geom_type', 'geom_size']:
                         # Check if mujoco_body_id is valid before accessing geom_bodyid
                         if mujoco_body_id != -1:
                            geom_ids = [g_id for g_id, b_id in enumerate(self.model.geom_bodyid) if b_id == mujoco_body_id]
                            if geom_ids:
                                g_id = geom_ids[0] # Take first geom associated with the body
                                if key == 'geom_type': feature = np.array([self.model.geom_type[g_id]], dtype=np.float32)
                                elif key == 'geom_size': 
                                    feature = self.model.geom_size[g_id].astype(np.float32) # Shape (3,)
                                     # Ensure geom_size is always length 3 for consistency
                                    if feature.shape[0] < 3:
                                        padded_size = np.zeros(3, dtype=np.float32)
                                        padded_size[:feature.shape[0]] = feature
                                        feature = padded_size
                                    elif feature.shape[0] > 3:
                                         feature = feature[:3]
                             
                         else: # No geom found
                            print(f"[RobosuiteNodeCentricObservation] Warning: No geom found for {body_name} for key '{key}'. Using zeros.")
                            feature = np.zeros(1 if key == 'geom_type' else 3, dtype=np.float32)
 
                     # Ensure feature has correct shape, pad if necessary
                     feature_flat = np.asarray(feature).flatten()
                     if feature_flat.shape[0] != expected_dim:
                         if feature_flat.shape[0] < expected_dim:
                            padded_feature = np.zeros(expected_dim, dtype=np.float32)
                            padded_feature[:feature_flat.shape[0]] = feature_flat
                            feature_flat = padded_feature
                         else: # Truncate if too long
                            feature_flat = feature_flat[:expected_dim]
                            # print(f"[RobosuiteNodeCentricObservation] Warning: Context feature '{key}' for {body_name} truncated from {feature_flat.shape[0]} to {expected_dim}.")
                     
                     context_features.append(feature_flat)
 
                 # Concatenate and pad/truncate to self.context_obs_size
                 node_context_vec = np.concatenate(context_features).astype(np.float32)
                 if node_context_vec.size != self.context_obs_size:
                    padded_ctx = np.zeros(self.context_obs_size, dtype=np.float32)
                    copy_len = min(node_context_vec.size, self.context_obs_size)
                    padded_ctx[:copy_len] = node_context_vec[:copy_len]
                    node_context_vec = padded_ctx
                 self.node_context_global[global_node_idx, :] = node_context_vec
 
        return self.node_proprio_global, self.node_context_global
 
    def observation(self, obs):
         """Processes the raw obs dict into the final node-centric format."""
         if not hasattr(self, '_structure_initialized') or not self._structure_initialized:
             try:
                 self._define_structure_and_masks()
             except Exception as e:
                 print(f"[RobosuiteNodeCentricObservation] Error defining structure in observation(): {e}")
                 return {k: np.zeros(s.shape, dtype=s.dtype) for k, s in self.observation_space.spaces.items()}
 
         # Extract features into global arrays
         try:
              node_proprio, node_context = self._extract_feat_per_node(obs)
             # check not zeros
             #  print(f"[RobosuiteNodeCentricObservation] Debug: node_proprio_global: {node_proprio}, node_context_global: {node_context}")
 
         except Exception as e:
              print(f"[RobosuiteNodeCentricObservation] Error during feature extraction: {e}. Returning zero arrays.")
              node_proprio = np.zeros_like(self.node_proprio_global)
              node_context = np.zeros_like(self.node_context_global)
 
         final_obs = OrderedDict()
         final_obs['proprioceptive'] = node_proprio.flatten()
         final_obs['context'] = node_context.flatten()
         final_obs['edges'] = self.edges_padded_global
         final_obs['obs_padding_mask'] = self.obs_padding_mask_global
         policy_node_output_dim = cfg.MODEL.TRANSFORMER.DECODER_OUT_DIM
         final_obs['act_padding_mask'] = self.metadata.get('act_padding_mask', np.ones(self.global_max_limbs * policy_node_output_dim, dtype=bool)) # Fetch from metadata
         
         final_obs['traversals'] = self.traversals_global 
         final_obs['SWAT_RE'] = self.swat_re_global
 
         for key in self.observation_space.spaces:
              if key not in final_obs and key in obs:
                  if obs[key].shape == self.observation_space[key].shape: 
                     final_obs[key] = obs[key]
 
         for key, space in self.observation_space.spaces.items():
              if key not in final_obs: 
                 print(f"[RobosuiteNodeCentricObservation] Warning: key {key} not in final_obs. Adding zero-filled array.")
                 final_obs[key] = np.zeros(space.shape, dtype=space.dtype)
 
         return final_obs
 
    def reset(self, **kwargs):
         """
         Resets the environment, and re-analyze the structure of the robot
         for new episode.
         """
         obs_raw = self.env.reset(**kwargs)
         self._structure_initialized = False
 
         return self.observation(obs_raw)
 
    def close(self):
         self.env.close()
         self.env.closed = True 
 

# ----------------------------------------------------------
# ---------------Node Centric Action------------------------
# ---------------------Transormer model---------------------
class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Maps the policy's globally padded, node-centric action output to the base
    RoboSuite environment's flat action space for the current morphology instance.
    """
    def __init__(self, env):
        super().__init__(env)

        # We need access to the Obs wrapper to get metadata (masks, indices)
        # and the Base wrapper to get the base action space dims.
        self.obs_wrapper_ref = self._find_wrapper_ref(RobosuiteNodeCentricObservation)
        self.base_env_ref = self._find_wrapper_ref(RobosuiteEnvWrapper)

        if not self.obs_wrapper_ref or not self.base_env_ref:
            raise TypeError("[RobosuiteNodeCentricAction] requires RobosuiteNodeCentricObservation and RobosuiteEnvWrapper in the stack.")

        self.global_max_limbs = cfg.MODEL.MAX_LIMBS

        # The policy outputs one action value potentially relevant to each *global* node slot.
        # The action mask will determine which ones are actually used.
        self.node_output_dim = cfg.MODEL.TRANSFORMER.get('DECODER_OUT_DIM', 1) 
        self.padded_action_dim_global = self.global_max_limbs * self.node_output_dim

        low = -1.0 * np.ones(self.padded_action_dim_global, dtype=np.float32)
        high = 1.0 * np.ones(self.padded_action_dim_global, dtype=np.float32)
        # This wrapper *redefines* the action space the agent interacts with.
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # These will be updated on reset via _update_mapping_info()
        self.current_act_mask = None
        self.current_per_robot_action_indices = None # for TwoArmEnv
        self.current_action_dims_per_robot = None
        self.current_total_action_dim = None
        self.current_num_robots = None
        self.current_base_action_space = self.base_env_ref.action_space # Store initial for clipping

    def _find_wrapper_ref(self, wrapper_type):
        """Helper to find a specific wrapper type in the stack below this one."""
        env = self.env 
        while hasattr(env, 'env'):
            if isinstance(env, wrapper_type):
                return env
            env = env.env
        if isinstance(env, wrapper_type):
             return env
        print(f"[RobosuiteNodeCentricAction] Warning: Could not find {wrapper_type.__name__} in wrapper stack below NodeCentricAction.")
        return None

    def _update_mapping_info(self):
        """
        Fetches the latest masks, indices, and action dims needed for mapping.
        Used during init and reset.
        """
        self.current_act_mask = self.obs_wrapper_ref.act_padding_mask_global # Direct access to global mask
        self.current_per_robot_action_indices = self.obs_wrapper_ref.metadata.get('per_robot_action_indices') # From metadata

        self.current_action_dims_per_robot = self.base_env_ref.action_dim_per_robot 
        self.current_total_action_dim = self.base_env_ref.total_action_dim
        self.current_num_robots = self.base_env_ref.num_robots
        self.current_base_action_space = self.base_env_ref.action_space
 
        if self.current_act_mask is None or \
           self.current_per_robot_action_indices is None or \
           not isinstance(self.current_per_robot_action_indices, list) or \
           len(self.current_per_robot_action_indices) != self.current_num_robots:
            raise ValueError(f"[RobosuiteNodeCentricAction] Action wrapper could not fetch valid mapping metadata from Observation wrapper. {len(self.current_per_robot_action_indices)} != {self.current_num_robots}")
 
        if len(self.current_act_mask) != self.padded_action_dim_global: # Mismatch in total action dimensions
             raise ValueError(f"[RobosuiteNodeCentricAction] Action mask length ({len(self.current_act_mask)}) or node_output_dim ({self.node_output_dim}) doesn't match padded_action_dim_global ({self.padded_action_dim_global}).")
 
        if sum(self.current_action_dims_per_robot) != self.current_total_action_dim:
              print(f"[RobosuiteNodeCentricAction] Warning: Sum of per-robot action dims ({sum(self.current_action_dims_per_robot)}) != total action dim ({self.current_total_action_dim}) in _update_mapping_info.")
 
        if self.current_total_action_dim != self.current_base_action_space.shape[0]: 
            print(f"[RobosuiteNodeCentricAction] Warning: Total action dim ({self.current_total_action_dim}) != base action space shape ({self.current_base_action_space.shape[0]}) in _update_mapping_info.")
 
    def action(self, action):
        """Maps the policy's padded action to the base environment's action space."""
        action = np.asarray(action)
 
        if action.shape != self.action_space.shape:
             raise ValueError(f"[RobosuiteNodeCentricAction] Input action shape {action.shape} != expected wrapper action space {self.action_space.shape}")
 
        self._update_mapping_info() # Ensure mapping info is up-to-date (esp. on reset)
        
        try:
            action_unpacked = action.reshape(self.global_max_limbs, self.node_output_dim)
            
            per_robot_final_actions = []
            for robot_idx in range(self.current_num_robots):
                controller_name = self.base_env_ref.loaded_controller_names[robot_idx]
                current_robot_nodes = self.obs_wrapper_ref.nodes_per_robot_instance[robot_idx] # (body_name, node_type) tuples
                
                robot_action_components = []

                for local_node_idx, (body_name, node_type) in enumerate(current_robot_nodes):
                    global_node_idx = self.obs_wrapper_ref.node_start_indices[robot_idx] + local_node_idx
                    
                    if global_node_idx >= self.global_max_limbs:
                        continue # Skip if outside global bounds
                    # Removed: if self.current_act_mask[global_node_idx]: continue

                    node_output = action_unpacked[global_node_idx, :] # Get the full output for this node (node_output_dim,)
                    
                    # --- Start of new/modified section ---
                    # Determine the starting index for this node's scalars in the flat act_mask
                    scalar_mask_start_idx = global_node_idx * self.node_output_dim

                    if controller_name == 'JOINT_VELOCITY':
                        if node_type == 'arm':
                            # Check if the first scalar for this arm joint is masked
                            if self.current_act_mask[scalar_mask_start_idx + 0]:
                                continue 
                            robot_action_components.append(node_output[0:1])
                        elif node_type == 'gripper':
                            # Check if the first scalar for this gripper joint is masked
                            if self.current_act_mask[scalar_mask_start_idx + 0]:
                                continue
                            # Also ensure cfg.ROBOSUITE.GRIPPER_DIM does not exceed self.node_output_dim
                            num_gripper_actions_to_take = min(cfg.ROBOSUITE.GRIPPER_DIM, self.node_output_dim)
                            robot_action_components.append(node_output[0:num_gripper_actions_to_take])

                    elif controller_name == 'OSC_POSE':
                        if node_type == 'hand':
                            # Check if the first scalar for the hand is masked
                            if self.current_act_mask[scalar_mask_start_idx + 0]:
                                continue
                            robot_action_components.append(node_output[0:6]) # OSC_POSE for hand is 6D
                        elif node_type == 'gripper':
                            # Check if the first scalar for the gripper is masked
                            if self.current_act_mask[scalar_mask_start_idx + 0]:
                                continue
                            robot_action_components.append(node_output[0:1]) # Gripper action is 1D for OSC_POSE
                    # --- End of new/modified section ---
                
                # Concatenate all components for this robot
                final_action_for_robot = np.concatenate(robot_action_components).astype(np.float32) if robot_action_components else np.array([], dtype=np.float32)

                # Ensure the final action dimension matches the expected for this robot's specific controller
                expected_dim_for_robot = self.current_action_dims_per_robot[robot_idx] # This is the true action dim for this robot from RobosuiteEnvWrapper

                # Handle potential dimension mismatch (should ideally not happen if logic is perfect, but for robustness)
                if final_action_for_robot.shape[0] != expected_dim_for_robot and expected_dim_for_robot > 0:
                    print(f"[RobosuiteNodeCentricAction] WARNING: Mismatch in final action dim for robot {robot_idx} ({final_action_for_robot.shape[0]}) vs expected ({expected_dim_for_robot}). Padding/truncating.")
                    padded_or_truncated_action = np.zeros(expected_dim_for_robot, dtype=np.float32)
                    copy_len = min(final_action_for_robot.shape[0], expected_dim_for_robot)
                    padded_or_truncated_action[:copy_len] = final_action_for_robot[:copy_len]
                    final_action_for_robot = padded_or_truncated_action

                per_robot_final_actions.append(final_action_for_robot)
            
            final_action_global = np.concatenate(per_robot_final_actions)

        except Exception as e: # Catch any other errors during action reconstruction
            print(f"[RobosuiteNodeCentricAction] Critical Error during action reconstruction: {e}. Returning zero action.")             
            return np.zeros_like(self.current_base_action_space.low)

        clipped_action = np.clip(final_action_global,
                                self.current_base_action_space.low,
                                self.current_base_action_space.high)

        return clipped_action.astype(self.current_base_action_space.dtype)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self._update_mapping_info()
        return ret


# ----------------------------------------------------------
# ------------------------MR SAMPLER------------------------
# ---------------------Transormer model---------------------
class RobosuiteSampleWrapper(gym.Wrapper):
    """
    Wrapper that samples a morphology configuration on each reset and instantiates
    the corresponding Robosuite environment stack. Handles dynamic morphology training
    based on a sampling sequence read from a file (e.g., sampling.json).
    """
    # Corrected __init__ signature to accept worker_rank and num_workers
    def __init__(self, env, all_morphology_configs, inner_stack_builder_fn, worker_rank, num_workers):
        """
        Args:
            env: A dummy environment object (often None) passed by the VecEnv.
                 This wrapper discards it and creates its own internal env stack.
            all_morphology_configs (list[dict]): List of all possible morphology
                                                 configuration dictionaries.
            inner_stack_builder_fn (callable): Function that takes a single morphology
                                               config dict and returns a Gym env instance
                                               (the top wrapper of the inner stack).
            worker_rank (int): The index (rank) of this worker process.
            num_workers (int): The total number of worker processes.
        """
        # Do NOT call super().__init__(env) here. We will call it later with the active_env_stack.
        # super().__init__(env)

        self.all_morphology_configs = all_morphology_configs
        self.inner_stack_builder_fn = inner_stack_builder_fn
        # Store worker rank and total number of workers
        self.worker_rank = worker_rank
        self.num_workers = num_workers

        self.active_env_stack = None # Holds the currently active environment stack
        self.active_morphology_index = None # Index of the config currently active

        # State for sampling sequence
        self._sampling_sequence = None
        self._sequence_episode_idx = 0 # Tracks current position in the sequence for this worker

        # The observation and action spaces will come from the currently active env stack,
        # but we need to define them after the first environment is created on reset.
        self.observation_space = None
        self.action_space = None

        # Need to call reset once to set up the initial environment and spaces
        # The reset method will call _sample_morphology and build the stack
        self.reset()

        # Now call the base Gym Wrapper __init__ with the active environment
        # This properly sets up self.env and attribute forwarding
        # Note: gym.Wrapper __init__ expects a Gym env. The active_env_stack is our Gym env instance.
        super().__init__(self.active_env_stack)
        # Re-assign spaces because super().__init__ might overwrite them with self.env's initial ones
        # These are already set by self.reset() just before super().__init__
        self.observation_space = self.active_env_stack.observation_space
        self.action_space = self.active_env_stack.action_space

        # print(f"Worker {self.worker_rank}: RobosuiteSampleWrapper initialized.") # Debug


    def _load_sampling_sequence(self):
        """Loads or reloads the sampling sequence from sampling.json."""
        sampling_file_path = os.path.join(cfg.OUT_DIR, "sampling.json")
        try:
            full_sequence = fu.load_json(sampling_file_path)

            if not full_sequence:
                 print(f"[RobosuiteSampleWrapper] Warning: Sampling sequence file is empty: {sampling_file_path}. Sampling randomly.")
                 # Fallback to random sampling if file is empty
                 self._sampling_sequence = [np.random.randint(len(self.all_morphology_configs)) for _ in range(1000)] # Create a dummy random sequence
            else:
                 # Divide the full sequence into chunks, one for each worker.
                 # PPO trainer should write a sequence long enough and divisible by num_workers.
                 # Need to handle potential index errors if not divisible or too short.
                 total_sequence_len = len(full_sequence)
                #  chunk_size = total_sequence_len // self.num_workers
                 start_idx = self.worker_rank * chunk_size
                 end_idx = start_idx + chunk_size if self.worker_rank < self.num_workers - 1 else total_sequence_len # Ensure last worker gets remaining

                 if start_idx >= total_sequence_len:
                     print(f"[RobosuiteSampleWrapper] Warning: Worker {self.worker_rank}: Start index ({start_idx}) >= total sequence len ({total_sequence_len}). Sequence exhausted? Using empty chunk.")
                     self._sampling_sequence = [] # Empty chunk
                 else:
                    #  self._sampling_sequence = full_sequence[start_idx:end_idx]
                    self._sampling_sequence = full_sequence[self.worker_rank :: self.num_workers]

                    # print(f"Worker {self.worker_rank}: Loaded chunk {self.worker_rank} (indices {start_idx}-{end_idx-1}) of sampling sequence (total len {total_sequence_len}, chunk size {chunk_size}). Chunk len {len(self._sampling_sequence)}") # Debug

            # Reset sequence index after loading (for this worker's chunk)
            self._sequence_episode_idx = 0

        except FileNotFoundError:
            print(f"[RobosuiteSampleWrapper] Warning: Sampling sequence file not found at {sampling_file_path}. Sampling randomly.")
            # Fallback to random sampling if the file doesn't exist
            self._sampling_sequence = [np.random.randint(len(self.all_morphology_configs)) for _ in range(1000)] # Create a dummy random sequence
            self._sequence_episode_idx = 0
        except Exception as e:
             print(f"[RobosuiteSampleWrapper] Error loading or processing sampling sequence file: {e}. Sampling randomly.")
             self._sampling_sequence = [np.random.randint(len(self.all_morphology_configs)) for _ in range(1000)] # Create a dummy random sequence
             self._sequence_episode_idx = 0

    # ------------------------------------------------------------------+
    #  Balanced fallback sampler                                         |
    # ------------------------------------------------------------------
    def _sample_morphology_index(self) -> int:
        """
        Deterministic, balanced sampling when `sampling.json` is absent.
        - If #morphs ≥ #workers, each worker gets a distinct morph first.
        - If #morphs <  #workers, morphs are repeated so every worker is
          assigned something, but the list is shuffled each cycle.
        After the list is exhausted it is regenerated and reshuffled.
        """
        if (
            self._sampling_sequence is None
            or self._sequence_episode_idx >= len(self._sampling_sequence)
        ):
            if len(self.all_morphology_configs) >= self.num_workers:
                # ↳ one unique morph per worker until list is exhausted
                shuffled = np.random.permutation(len(self.all_morphology_configs))
                self._sampling_sequence = shuffled.tolist()
            else:
                # ↳ need repeats so every worker gets something
                repeats = int(
                    np.ceil(self.num_workers / len(self.all_morphology_configs))
                )
                seq = np.tile(
                    np.arange(len(self.all_morphology_configs)), repeats
                )
                np.random.shuffle(seq)
                self._sampling_sequence = seq.tolist()
            self._sequence_episode_idx = 0

        # idx = self._sampling_sequence[self._sequence_episode_idx]
        idx = self._sampling_sequence[
            (self._sequence_episode_idx + self.worker_rank) % len(self._sampling_sequence)
        ]
        self._sequence_episode_idx += 1
        return idx


    def reset(self, sample_index=None, **kwargs):
        """
        Resets the wrapper, samples a morphology, builds/resets the inner env stack.

        Args:
            sample_index (int, optional): Explicitly provide the index of the morphology
                                          configuration to sample for this episode.
                                          If None (default), samples from the sequence file.
                                          This index is the index into the *all_morphology_configs* list.
            **kwargs: Additional arguments for the inner environment's reset method (e.g., seed).
        """
        # Close the previously active environment stack if it exists
        if self.active_env_stack is not None:
            try:
                # Pass any kwargs to the inner stack's close if it supports them
                self.active_env_stack.close(**kwargs)
            except Exception as e:
                 print(f"[RobosuiteSampleWrapper] Warning: Error closing previous environment stack for worker {self.worker_rank}: {e}")
                 # Attempt to force deletion if close fails
                 del self.active_env_stack
                 self.active_env_stack = None


        # --- Morphology Sampling ---
        if sample_index is not None:
            # Use the provided index (e.g., for specific evaluation scenarios)
            self.active_morphology_index = sample_index
            # Validate the provided index
            if self.active_morphology_index < 0 or self.active_morphology_index >= len(self.all_morphology_configs):
                 print(f"[RobosuiteSampleWrapper] Error: Provided sample_index {sample_index} is out of bounds (0-{len(self.all_morphology_configs)-1}). Defaulting to index 0 for worker {self.worker_rank}.")
                 self.active_morphology_index = 0
            # Do NOT increment sequence index when explicitly sampling.
        else:
            # Sample morphology index from the sequence file for training
            self.active_morphology_index = self._sample_morphology_index()

        # Get the morphology config dictionary for the selected index
        selected_config = self.all_morphology_configs[self.active_morphology_index]
        # print(f"Worker {self.worker_rank}: Building env for morph index {self.active_morphology_index}") # Debug

        # --- Instantiate Inner Environment Stack ---
        # The inner_stack_builder_fn takes the config_dict and returns the top wrapper.
        try:
            self.active_env_stack = self.inner_stack_builder_fn(selected_config)
        except Exception as e:
            print(f"[RobosuiteSampleWrapper] Error building inner env stack for morph index {self.active_morphology_index} (worker {self.worker_rank}): {e}")
            raise RuntimeError(f"[RobosuiteSampleWrapper] Failed to build inner env stack for morph index {self.active_morphology_index} (worker {self.worker_rank})") from e


        # Call reset on the newly created inner environment stack
        # Pass kwargs (including seed+rank) down to the inner stack's reset
        try:
            observation = self.active_env_stack.reset(**kwargs)
            # Ensure the inner stack's reset method propagates kwargs, especially the seed.
        except Exception as e:
            print(f"[RobosuiteSampleWrapper] Error resetting inner env stack for morph index {self.active_morphology_index} (worker {self.worker_rank}): {e}")
            # If reset fails, attempt to close the failed env and raise error.
            try: self.active_env_stack.close()
            except: pass # Ignore close error during cleanup
            raise RuntimeError(f"[RobosuiteSampleWrapper] Failed to reset inner env stack for morph index {self.active_morphology_index} (worker {self.worker_rank})") from e


        # Update observation and action spaces based on the new environment stack
        # These might vary slightly (e.g., Dict space keys kept by SelectKeysWrapper)
        self.observation_space = self.active_env_stack.observation_space
        self.action_space = self.active_env_stack.action_space

        self.metadata['active_morph_config_index'] = self.active_morphology_index

        # add inside RobosuiteSampleWrapper.reset()
        # DEBUG: to know how we assign each worker to each morph idx
        # print(f"[Worker {self.worker_rank}] uses morph {self.active_morphology_index} "
        # f"at episode {self._sequence_episode_idx}")

        # Return the observation from the inner environment
        return observation

    def step(self, action):
        """Steps the currently active environment stack."""
        if self.active_env_stack is None:
            # This should not happen if reset is called first
            raise RuntimeError("[RobosuiteSampleWrapper] Step called before environment was initialized. Call reset first for worker {self.worker_rank}.")
        # Pass action to the active inner environment stack
        obs, rew, done, info = self.active_env_stack.step(action)

        # Add the active morphology config index to the info dict if needed by TrainMeter
        # RecordEpisodeStatistics might already propagate this from self.metadata
        info['morph_config_index'] = self.active_morphology_index # Add index to info

        return obs, rew, done, info

    def close(self):
        """Closes the currently active environment stack."""
        if self.closed: # Check base gym.Wrapper closed flag
            return
        if self.active_env_stack is not None:
            try:
                self.active_env_stack.close()
            except Exception as e:
                 print(f"[RobosuiteSampleWrapper] Warning: Error closing active environment stack for worker {self.worker_rank}: {e}")
            self.active_env_stack = None
        # Set the base gym.Wrapper closed flag
        self.closed = True
        # print(f"Worker {self.worker_rank}: RobosuiteSampleWrapper closed.") # Debug

    # get_unimal_idx method
    def get_unimal_idx(self):
        """Returns the index of the currently active morphology."""
        # This method is called by VecPyTorch/PPO to get the task ID.
        # For multi-morph training, the active morphology index is the task ID.
        # Return a list/array matching num_envs dimension (which is 1 for this wrapper instance)
        # The PPO trainer will collect these across all workers.
        return [self.active_morphology_index] # Return a list containing the single index


    def seed(self, seed=None):
        """Sets the seed for this wrapper and passes it down to the inner stack."""
        # Seed the random state used by this wrapper (if any, e.g., for fallback random sampling)
        # Add a np_random attribute to SampleWrapper for random fallback sampling
        if not hasattr(self, 'np_random') or self.np_random is None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        else:
             self.np_random.seed(seed)

        # Pass the worker-specific seed down to the active inner environment stack's seed method.
        # The inner stack's seed method should then handle seeding its base env and spaces.
        if self.active_env_stack is not None:
             # The seed passed here should be the worker-specific seed (base seed + rank)
             self.active_env_stack.seed(seed)
        # Store the seed that was set
        self._last_seed = seed
        return [seed] # Return the list of seeds set (standard Gym convention)


