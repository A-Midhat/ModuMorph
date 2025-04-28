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

class RobosuiteEnvWrapper(gym.Env):
    """
    Wraps a robosuite environment to make it compatible with gym.Env.
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, robosuite_env_name, robot_name, controller_name=None, robosuite_cfg=None):
        """
        robotsuite_env_name: the task (i.e. Lift).
        robot_name: the robot (i.e. Panda).
        controller_name: Method of controlling joints (i.e. JOINT_POSITION).
        robosuite_cfg: a dict of robosuite.make() kwargs.
        """
        super().__init__()
        self.robosuite_env_name = robosuite_env_name
        self.robot_name = robot_name
        self.robosuite_cfg = robosuite_cfg if robosuite_cfg is not None else {}

        # --- Controller Setup --- 
        # The problem with the different controllers is that they have different action spaces.
        # TODO: try and make the action space the same for all controllers. Or generalize the action space.

        controller_name = controller_name 
        if controller_name not in ALL_CONTROLLERS:
            print("[RobosuiteEnvWrapper] Warning: controller_name not in ALL_CONTROLLERS. Using default controller. (JOINT_POSITION)")
            self.controller_name = "JOINT_POSITION"
        
        try: 
            self.controller_config = load_controller_config(default_controller=controller_name)
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] Error loading controller_config: {e}")
            self.controller_config = load_controller_config(default_controller="JOINT_POSITION")

        # --- Env Setup --- 
        # TODO: Add to config file
        robosuite_init_args = {
            "env_name": self.robosuite_env_name,
            "robots": self.robot_name,
            "controller_configs": self.controller_config,
            "has_renderer": False, # False for training
            "has_offscreen_renderer": False, # TODO: Set True to use camera observations as exteroceptive observations
            #"ignore_done": True, # Use PPO for termniation (TODO:Uncomment)
            "use_camera_obs": False, # TODO: Set True to use camera observations as exteroceptive observations
            "control_freq":20,
            "hard_reset": False, # TODO: check if it is really faster during training 
            "horizon":1000,
            "reward_shaping": True
        }

        robosuite_init_args.update(self.robosuite_cfg.get("ENV_ARGS", {}))
        #print("Robosuite init args: ", robosuite_init_args)

        # Create the env
        self.env = robosuite.make(**robosuite_init_args)

        # --- Gym Interfase Setup ---
        act_low, act_high = self.env.action_spec
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)

        # the following obs will be overriden by the observation wrapper
        obs_spec = self.env.observation_spec()
        gym_obs_spaces = {}
        for key, spec in obs_spec.items():
            try:
                _shape = spec.shape
                _dtype = spec.dtype

                _dtype = np.float32 if _dtype == np.float64 else _dtype
                gym_obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=_shape, dtype=_dtype)
            except Exception as e:
                print(f"[RobosuiteEnvWrapper] Error processing obs spec: {e}")
                continue
        self.observation_space = spaces.Dict(gym_obs_spaces) # new obs space 
        
        # ---Metadata--- 
        self._robot_metadata = self._extract_robot_metadata()
        self.metadata["robot_metadata"] = self._robot_metadata 
        # for multi-robot envs, we will use robot_name 
        #TODO: add some kind of unique Id (when using same robot but differen kinamtics)
        self.ep_step_count = 0 # optional 

    def _extract_robot_metadata(self):
        """
        Extracts static metadata from the robot.
        # TODO: add more metadata if needed 
        """

        metadata = {}
        if hasattr(self.env, "robots") and len(self.env.robots) > 0:
            robot = self.env.robots[0]
            metadata["robot_name"] = robot.name 
            metadata['num_arm_joints'] = len(robot._ref_joint_indexes) # use _ref_joint_indexes instead of arm_joint_names, returns list of len (num of joints) without gripper
            metadata['num_gripper_joints'] = robot.gripper.dof # for 1 gripper it returns 1, if i want the real joints we can use i.e. robot._gripper_joint_pos_indexes (2)
            
            # Number of nodes (DoF + Gripper)
            metadata['num_nodes'] = metadata['num_arm_joints'] + metadata['num_gripper_joints']
            # TODO: add the following (lower, upper, damping, armature, friction) physical properties (check dir(robot))
            metadata['joint_names'] = robot.robot_joints 
            metadata['g_joint_name'] = robot.gripper_joints 
            # should I combine them? TODO: check again
            #metadata['joint_names'] = robot.robot_joints + robot.gripper_joints

        else: 
            # dummyy values
            print("[RobosuiteEnvWrapper] Warning: No robot found in the environment. Metadata will set to None.")
            metadata['robot_name'] = "Unkown"
            metadata['num_arm_joints'] = 0
            metadata['num_gripper_joints'] = 0
            metadata['num_nodes'] = 0
            metadata['joint_names'] = []
            #metadata['link_names'] = []
        
        #print("Robot metadata: ", metadata)
        return metadata
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high) # enusre [-1, 1] range
        try:
            # done here is ignored 
            obs_dict, reward, done, info = self.env.step(action)
            done = False # TODO: UNCOMMNET
            #---Debug ---
            # if self.ep_step_count %20==0 or reward !=0:
            #     print(f" [RobosuiteEnvWrapper Step {self.ep_step_count}] Raw Reward: {reward}")
            #-------------
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] ERROR during Robosuite step: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Action=`{action}`, Error={e}")
            raise e # for debugging 
            # TODO: add a flag to skip the step and return None for training stablity (UNCOMMENT)
            obs_dict = self.observation_space.sample() # dummy obs
            reward = -10 # heavily penelized 
            done = True 
            info = {'error': str(e), 'name': self.robot_name}
        
        # --- Post-processing of obs---
        processed_obs = self._convert_observation(obs_dict)
        
        # ---Custom info---
        #info['robot_name'] = self.robot_name # causing error at meter
        info['name'] = self.robot_name
        info['raw_reward'] = reward
        info['action'] = action

        self.ep_step_count += 1 

        return processed_obs, reward, done, info

    def reset(self):
        try: 
            obs_dict = self.env.reset()
        except Exception as e:
            print(f"[RobosuiteEnvWrapper] ERROR during Robosuite reset: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Error={e}")
            raise e # for debugging
            # TODO: add a flag to skip the reset and return None for training stablity (UNCOMMENT)
            #obs_dict = self.observation_space.sample() # dummy obs
            #reward = -10 # heavily penelized
            # done = True
        processed_obs = self._convert_observation(obs_dict)
        return processed_obs
    
    def _convert_observation(self, obs_dict):
        """
        Converts the flat observation dict from robosuite into 
        preliminary structured dict, grouping related info logically, 
        before we pass it through the NodeCentricObservationWrapper.
        """

        prelim_obs = {}

        # ---Propricopetive State--- 
        # we can use the combined state provided by robotsuite 
        # TODO: try and handle this for multiple robots 
        robot_proprio_key = f'robot0_proprio-state' # change the '0' to the robot index if we have multiple robots (guess?)
        if robot_proprio_key in obs_dict:
            prelim_obs[robot_proprio_key] = obs_dict[robot_proprio_key]
        else:
            print(f"[RobosuiteEnvWrapper] Warning: {robot_proprio_key} state not found for robot {self.robot_name} in the observation dictionary.")
            # the problem is this is robot-dependent naming 
            parts = []
            keys_to_try = [
                f'robot0_joint_pos_cos', f'robot0_joint_pos_sin', # Use sin/cos for angles
                f'robot0_joint_vel',
                f'robot0_eef_pos', f'robot0_eef_quat',
                f'robot0_gripper_qpos', f'robot0_gripper_qvel'
            ]
            for key in keys_to_try:
                if key in obs_dict:
                    parts.append(obs_dict[key].astype(np.float32))
                else:
                    # either pad or raise an error ? 
                    print(f"[RobosuiteEnvWrapper] Missing key: {key}")
                    # skip for now 
                    pass 
            if not parts:
                # create a dummy observation (zeros)
                # Problem we need to know the shape of the observation
                print(f"[RobosuiteEnvWrapper] ERROR: Could not assemble robot-state for {self.robot_name}.")
                prelim_obs['robot0_proprio-state'] = np.zeros(1, dtype=np.float32) # leads to problem better to raise an error
            else :
                prelim_obs['robot0_proprio-state'] = np.concatenate(parts)
        
        # ---Object State---
        if 'object-state' in obs_dict:
            prelim_obs['object-state'] = obs_dict['object-state'].astype(np.float32)
        else:
            # Either Env doesnt have object or is it disabled
            print(f"[RobosuiteEnvWrapper] Warning: object-state not found in the observation dictionary.")
            # create a dummy observation (zeros)
            prelim_obs['object-state'] = np.zeros([], dtype=np.float32)
        
        # TODO: Combined states (proprio + obj) for MLP (single robot)
        # combined_state = np.concatenate([obs_dict[robot_proprio_key], obs_dict['object-state']])
        # --------------------------------------------------------------------------------------
        # ---Gripper to Object Distance--- (This is the last 3 elements of object-state) redundant with object-state
        # TODO: alt we can use the eef_pos and gripper_to_object directly
        # obj_pos_key = 'object-state' # gripper to obj is the last 3 elemtns in object-state 
        # eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
        # if obj_pos_key in obs_dict and eef_pos_key in obs_dict:
        #     obj_pos = obs_dict[obj_pos_key][-3:]
        #     eef_pos = obs_dict[eef_pos_key]
        #     prelim_obs['gripper_to_object'] = (obj_pos - eef_pos).astype(np.float32)
        # else:
        #     print(f"[RobosuiteEnvWrapper] Warning: {obj_pos_key} or {eef_pos_key} not found in the observation dictionary.")
        #     # create a dummy observation (zeros)
        #     prelim_obs['gripper_to_object'] = np.zeros(3, dtype=np.float32) # TODO: better to set high value 

        # ---EEF State--- (TODO: add this As a direct input to EXT_MIX)
        # TODO: make the name consistent with the other keys
        eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
        if eef_pos_key in obs_dict:
            prelim_obs['robot0_eef_pos'] = obs_dict[eef_pos_key].astype(np.float32)
        else:
            print(f"[RobosuiteEnvWrapper] Warning: {eef_pos_key} not found in the observation dictionary.")
            # create a dummy observation (zeros)
            prelim_obs['robot0_eef_pos'] = np.zeros(3, dtype=np.float32)
        
        eef_quat_key = f'robot0_eef_quat' # change `0` to the robot index if we have multiple robots (guess)
        if eef_quat_key in obs_dict:
            prelim_obs['robot0_eef_quat'] = obs_dict[eef_quat_key].astype(np.float32)
        else:
            print(f"[RobosuiteEnvWrapper] Warning: {eef_quat_key} not found in the observation dictionary.")
            prelim_obs['robot0_eef_quat'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Default Identity 
        # f'robot0_gripper_qpos', f'robot0_gripper_qvel'
        # ---Gripper State---
        gripper_qpos_key = f'robot0_gripper_qpos'
        if gripper_qpos_key in obs_dict:
            prelim_obs['robot0_gripper_qpos'] = obs_dict[gripper_qpos_key].astype(np.float32)
        else:
            print(f"[RobosuiteEnvWrapper] Warning: {gripper_qpos_key} not found in the observation dictionary.")
            prelim_obs['robot0_gripper_qpos'] = np.zeros(1, dtype=np.float32)

        gripper_qvel_key = f'robot0_gripper_qvel'
        if gripper_qvel_key in obs_dict:
            prelim_obs['robot0_gripper_qvel'] = obs_dict[gripper_qvel_key].astype(np.float32)
        else:
            print(f"[RobosuiteEnvWrapper] Warning: {gripper_qvel_key} not found in the observation dictionary.")
            prelim_obs['robot0_gripper_qvel'] = np.zeros(1, dtype=np.float32)
        # --- Robot Context (Static Info) --- 
        # features extracted from the _robot_metadata so node-centric wrapper can easily distribute it to all nodes
        prelim_obs['robot_context_dict'] = self.metadata['robot_metadata']
        # TODO: Combine all eef and gripper obs into a single combined state (similar to the object-state/proprio)
        return prelim_obs
        
    def close(self):
        self.env.close()
    
    def render(self, mode='human', width=512, height=512, camera_name=None):
        if mode == "rgb_array":
            if self.env.has_offscreen_renderer:
                cam_name = camera_name if camera_name else self.env.render_camera 
                return self.env.sim.render(
                    camera_name=cam_name,
                    width=width,
                    height=height,
                    depth=False
                )
            else :
                print("Warning: Offscreen rendering is not available for this environment")
                return self.env.sim.render(
                    mode='offscreen',
                    width=width,
                    height=height,
                    depth=False
                ) # alt return zeros (blank)
        elif mode == "human":
            if self.env.has_renderer:
                self.env.render()
            else:
                print("Warning: 'human' Rendering is not available for this environment")
        else:
            return super().render(mode=mode)
    
    def seed(self, seed=None):
        # Normally robosuite handles this when robosuite.make is called

        if hasattr(self.env, "seed") and callable(self.env.seed):
            self.env.seed(seed)
        self.action_space.seed(seed)
    
    def sample_action(self):
        # mimic the gym action_space.sample()
        low, high = self.action_space.low, self.action_space.high 
        action = np.random.uniform(low, high)
        return action.astype(self.action_space.dtype)


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


# --------------Node-Centric Wrappers------------------------
class RobosuiteNodeCentricObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Check inheritance chain
        if not isinstance(self.env, RobosuiteEnvWrapper):
             parent = self.env
             is_base_wrapper = False
             while hasattr(parent, 'env'):
                  if isinstance(parent, RobosuiteEnvWrapper):
                      is_base_wrapper = True
                      break
                  parent = parent.env
             if not is_base_wrapper:
                  raise TypeError(f"[RobosuiteNodeCentricObservation] must wrap a RobosuiteEnvWrapper instance, but got {type(self.env)}.")

        # Need to access metadata from the correct env instance in the chain
        self.base_env_ref = self.env # Keep ref to immediate parent
        while not isinstance(self.base_env_ref, RobosuiteEnvWrapper):
            if not hasattr(self.base_env_ref, 'env'):
                 raise TypeError("[RobosuiteNodeCentricObservation] Could not find RobosuiteEnvWrapper in the wrapper stack.")
            self.base_env_ref = self.base_env_ref.env

        self.robot_metadata = self.base_env_ref.metadata.get('robot_metadata', {})
        if not self.robot_metadata:
            raise ValueError("[RobosuiteNodeCentricObservation] Robot metadata not found in underlying env.")
        self.robot_name = self.robot_metadata.get('robot_name', 'Unknown')

        # --- Define Node Structure (Simplified Plan) ---
        # Node 0: Base (Placeholder features) Assuming Fixed Base (TODO: account for mobile base)
        # Node 1-N: Arm Link/Joint (N=num_arm_joints)
        self.num_arm_joints = self.robot_metadata.get('num_arm_joints', 0)
        # Use N+1 nodes: Base + one node per *arm* joint/link it controls
        self.num_nodes = self.num_arm_joints + 1
        self.num_gripper_joints = self.robot_metadata.get('num_gripper_joints', 0)

        self.max_limbs = cfg.MODEL.MAX_LIMBS # Max sequence length (nodes)
        self.max_joints = cfg.MODEL.MAX_JOINTS # Max controllable joints (related to edges)

        # TODO: This should never happen
        # Add code to get the MAX_LIMBS and MAX_JOINTS from the list of ROBOTS and overwrite the config
        if self.num_nodes > self.max_limbs:
             raise ValueError(f"Robot {self.robot_name} needs {self.num_nodes} nodes, but config MODEL.MAX_LIMBS is only {self.max_limbs}.")
        # Check edge count against max_joints
        num_real_edges = self.num_nodes - 1
        if num_real_edges > self.max_joints:
             raise ValueError(f"Robot {self.robot_name} has {num_real_edges} kinematic edges, but config MODEL.MAX_JOINTS is only {self.max_joints}.")

        # Calculate padding required
        self.num_node_pads = self.max_limbs - self.num_nodes

        # --- Determine Per-Node Feature Sizes (Simplified Plan) ---
        self.limb_obs_size = self._get_proprio_feature_dim() # Proprioceptive features per node
        self.context_obs_size = self._get_context_feature_dim() # Context features per node
        print(f"[RobosuiteNodeCentricObservation][{self.robot_name}] Nodes: {self.num_nodes}, ProprioDim/Node: {self.limb_obs_size}, ContextDim/Node: {self.context_obs_size}")

        # --- Create Padding Masks and Edge List ---
        self._create_padding_masks_and_edges()
        # Store masks in metadata for the action wrapper and potentially the model
        self.metadata['act_padding_mask'] = self.act_padding_mask
        # self.metadata['num_nodes'] = self.num_nodes # Pass actual node count TODO: remove this redundant metadata metadata['robot_metadata']['num_nodes']

        # --- Define the NEW Observation Space ---
        self.observation_space = self._define_observation_space()
        print(f"[RobosuiteNodeCentricObservation][{self.robot_name}] Final Observation Space Defined.")

    # TODO: This function should be able to calc the proprioceptive observation from the env
    def _get_proprio_feature_dim(self):
        """Calculate the dimension of proprioceptive features per node (Simplified Plan)."""
        # Node 0 (Base): Placeholder = 3 dims (e.g., zeros or fixed encoding).
        # Node i (1-N): Joint i state [sin(pos), cos(pos), vel] = 3 dims.
        # Let's make them consistent for simplicity.
        return 3

    def _get_context_feature_dim(self):
        """Calculate the dimension of context features per node (Simplified Plan)."""
        # Node 0 (Base): Placeholder = 5 dims (e.g., zeros or fixed encoding).
        # Node i (1-N): Joint i static props [limit_low, limit_high, damping, armature, frictionloss] = 5 dims.
        # Let's make them consistent for simplicity.
        return 5

    def _create_padding_masks_and_edges(self):
        """Generates padding masks and the padded edge list based on the simplified plan."""
        # --- Obs Padding Mask (for Transformer input) ---
        self.obs_padding_mask = np.asarray([False] * self.num_nodes + [True] * self.num_node_pads, dtype=bool)

        # --- Edge List (Kinematic Chain 0->1->...->N) ---
        num_real_edges = self.num_nodes - 1 # Base doesn't have a parent
        real_edges = []
        if num_real_edges > 0:
            # Edges: [child1, parent1, child2, parent2, ...]
            parents = list(range(self.num_nodes - 1)) # Nodes 0 to N-1 are parents
            children = list(range(1, self.num_nodes)) # Nodes 1 to N are children
            real_edges = np.array(list(zip(children, parents)), dtype=np.int32).flatten()

        # Pad edges up to max_joints length
        num_real_edge_pairs = len(real_edges) // 2
        self.num_joint_pads = self.max_joints - num_real_edge_pairs # max_joints relates to EDGES here

        # Pad with dummy edges (e.g., self-loops on the dummy padding node index)
        pad_value = self.max_limbs - 1 # Use last possible node index as dummy padding index
        padded_edges = np.full(2 * self.max_joints, pad_value, dtype=np.int32) # Total length matches config
        if len(real_edges) > 0:
             if len(real_edges) > len(padded_edges):
                 print(f"[RobsouiteNodeCentricObservation] Warning: Robot {self.robot_name} has more edges ({num_real_edge_pairs}) than MAX_JOINTS ({self.max_joints}). Truncating edges.")
                 padded_edges[:] = real_edges[:2 * self.max_joints]
             else:
                 padded_edges[:len(real_edges)] = real_edges
        self.edges = padded_edges.astype(np.float32) # Convert to float for obs space compatibility

        # --- Action Padding Mask ---
        # Policy main decoder outputs `max_limbs` potential actions, 1 per node.
        # Valid actions correspond to nodes 1 to num_arm_joints.
        is_valid_action = [False] # Action for base node (node 0) is invalid # TODO: Need to account for mobile base
        is_valid_action.extend([True] * self.num_arm_joints) # Actions for nodes 1 to N_arm are valid arm joints
        # Pad remaining node slots up to max_limbs
        is_valid_action.extend([False] * (self.max_limbs - (self.num_arm_joints + 1)))
        # Invert it for the masking
        self.act_padding_mask = np.array(~np.array(is_valid_action), dtype=bool)

        if len(self.act_padding_mask) != self.max_limbs:
             raise ValueError(f"[RobsouiteNodeCentricObservation] Action padding mask length ({len(self.act_padding_mask)}) doesn't match MAX_LIMBS ({self.max_limbs}). Check logic.")
        if (~self.act_padding_mask).sum() != self.num_arm_joints:
             print(f"[RobsouiteNodeCentricObservation] Warning: Action mask expects {self.num_arm_joints} valid arm actions, but mask has {(~self.act_padding_mask).sum()} valid entries.")


    def _define_observation_space(self):
        """Defines the final, padded observation space dictionary."""
        inf = np.float32(np.inf)
        obs_spaces = OrderedDict()

        # Padded Node-Centric Observations - SHOULD BE FLATTENED
        obs_spaces['proprioceptive'] = Box(-inf, inf, (self.max_limbs * self.limb_obs_size,), np.float32) # Flattened
        obs_spaces['context'] = Box(-inf, inf, (self.max_limbs * self.context_obs_size,), np.float32)    # Flattened

        # Padded Structural Information
        obs_spaces['edges'] = Box(-np.inf, np.inf, (2 * self.max_joints,), np.float32) # Use np.inf for consistency

        # Padding Masks
        obs_spaces['obs_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool) # For transformer input
        obs_spaces['act_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool) # Matches arm action output structure

        # Extroceptive Information (passed through) - Get shapes from underlying env's observation spec
        base_obs_space_dict = self.base_env_ref.observation_space.spaces

        # Define keys expected from RobosuiteEnvWrapper._convert_observation to be passed through
        # obs keys after RobosuiteEnvWrapper => ['robot0_proprio-state', 'object-state', 'eef_pos', 'eef_quat', 'robot_context_dict']
        # extro_keys_to_check = ['object-state', 'gripper_to_object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
        extro_keys_to_check = ['object-state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
        for key in extro_keys_to_check:
            if key in base_obs_space_dict:
                 spec = base_obs_space_dict[key]
                 shape = spec.shape
                 # Ensure shape is valid (not zero size if key exists)
                 if np.prod(shape) > 0:
                     dtype = np.float32 if spec.dtype == np.float64 else spec.dtype # Ensure float32
                     obs_spaces[key] = Box(-inf, inf, shape, dtype)
                 else:
                     print(f"[RobosuiteNodeCentricObservation] Warning: Base observation key '{key}' has size 0, omitting from final obs space.")
            # else: Key simply won't be added to the final space

        return Dict(obs_spaces)


    def _extract_features_per_node(self, obs_dict):

        """
        Distributes features from the preliminary observation dictionary
        into per-node feature vectors according to the simplified plan.
        """
        node_proprio = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)
        node_context = np.zeros((self.max_limbs, self.context_obs_size), dtype=np.float32)

        # --- Extract Base Robosuite Obs ---
        robot_state = obs_dict.get(f'robot0_proprio-state', None) # should never return None
        if robot_state is None:
            # This case should ideally be handled in RobosuiteEnvWrapper, but add check here
            print(f"[RobosuiteNodeCentricObservation] ERROR: 'robot0_proprio-state' missing from input observation.")
            return node_proprio, node_context # Return zeros

        # Need to parse the flat robot_state based on robosuite's internal order
        # Check robosuite/utils/observables.py and specific robot files (e.g., panda.py)
        
        jv_dim = self.num_arm_joints
        gq_dim = self.num_gripper_joints * 1 # qpos
        gqv_dim = self.num_gripper_joints * 1 # qvel

        joint_pos = robot_state[0:jv_dim]
        joint_vel = robot_state[jv_dim : jv_dim * 2]
        # Skip gripper state here, handled separately or via extroceptive keys
        # gripper_qpos = robot_state[jv_dim*2 : jv_dim*2 + gq_dim]
        # gripper_qvel = robot_state[jv_dim*2 + gq_dim : jv_dim*2 + gq_dim + gqv_dim]

        # Convert joint positions to sin/cos
        joint_pos_sin = np.sin(joint_pos)
        joint_pos_cos = np.cos(joint_pos)

        # --- Extract Context (Static Properties) from MuJoCo model ---
        try:
            sim = self.base_env_ref.env.sim
            robot_model = self.base_env_ref.env.robots[0]
            ref_joint_indexes = robot_model._ref_joint_indexes      # MuJoCo joint IDs (e.g., 1, 2, ... 7 for Panda arm)
            ref_joint_vel_indexes = robot_model._ref_joint_vel_indexes   # MuJoCo DoF IDs (e.g., 6, 7, ... 12 for Panda arm)
        except AttributeError as e:
            print(f"[RobosuiteNodeCentricObservation] ERROR: Could not access sim or robot model for context extraction: {e}")
            # Return zeros if context cannot be extracted
            return node_proprio, node_context

        num_arm_joints_to_use = self.num_arm_joints

        # Pre-allocate context arrays
        joint_limits = np.zeros((num_arm_joints_to_use, 2), dtype=np.float32)
        joint_damping = np.zeros(num_arm_joints_to_use, dtype=np.float32)
        joint_armature = np.zeros(num_arm_joints_to_use, dtype=np.float32)
        joint_friction = np.zeros(num_arm_joints_to_use, dtype=np.float32)

        # Extract context for each arm joint (node)
        for i in range(num_arm_joints_to_use):
            try:
                joint_id_model = ref_joint_indexes[i]
                dof_id_model = ref_joint_vel_indexes[i]

                joint_limits[i] = sim.model.jnt_range[joint_id_model]
                joint_damping[i] = sim.model.dof_damping[dof_id_model]
                joint_armature[i] = sim.model.dof_armature[dof_id_model]
                joint_friction[i] = sim.model.dof_frictionloss[dof_id_model]
            except IndexError as e:
                 print(f"[RobosuiteNodeCentricObservation] Error accessing MuJoCo model properties for joint index {i}: {e}. Using zeros.")
                 # Keep default zeros for this joint's context

        # --- Populate Node Features (Nodes 1 to N_arm) ---
        for i in range(num_arm_joints_to_use):
            node_idx = i + 1 # Node index (1-based for arm joints)
            if node_idx < self.max_limbs:
                # Proprioceptive: [sin(pos_i), cos(pos_i), vel_i] # TODO:add more obs
                if i < len(joint_pos_sin): # Check bounds
                    proprio_feat = [joint_pos_sin[i], joint_pos_cos[i], joint_vel[i]]
                    node_proprio[node_idx, :3] = np.array(proprio_feat, dtype=np.float32)
                else:
                    print(f"[RobosuiteNodeCentricObservation] Warning: Index {i} out of bounds for joint pos/vel arrays.")

                # Context: [limit_low, limit_high, damping, armature, frictionloss]
                if i < len(joint_limits): # Check bounds
                    context_feat = [joint_limits[i, 0], joint_limits[i, 1], joint_damping[i], joint_armature[i], joint_friction[i]]
                    node_context[node_idx, :5] = np.array(context_feat, dtype=np.float32)
                else:
                     print(f"[RobosuiteNodeCentricObservation] Warning: Index {i} out of bounds for joint context arrays.")


        # --- Populate Node 0 (Base) ---
        # Keep as zeros for now in this simplified plan.

        return node_proprio, node_context


    def observation(self, obs):
        """Processes the observation dictionary into the final node-centric format."""
        start_time = time.time()

        # Extract features and distribute them into node-based arrays
        node_proprio, node_context = self._extract_features_per_node(obs)

        # --- Assemble the Final Observation Dictionary ---
        final_obs = OrderedDict()
        # Store FLATTENED node features
        final_obs['proprioceptive'] = node_proprio.flatten().astype(np.float32)
        final_obs['context'] = node_context.flatten().astype(np.float32)
        final_obs['edges'] = self.edges # Use pre-computed padded edges
        final_obs['obs_padding_mask'] = self.obs_padding_mask
        final_obs['act_padding_mask'] = self.act_padding_mask

        # --- Pass Through Extroceptive & Other Global Features ---
        # Ensure keys match those defined in self._define_observation_space()
        # for key in ['object-state', 'gripper_to_object', 'eef_pos', 'eef_quat', 'gripper_qpos', 'gripper_qvel']:
        for key in ['object-state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']:
            if key in self.observation_space.spaces and key in obs:
                final_obs[key] = obs[key].astype(np.float32) # Ensure correct type
            elif key in self.observation_space.spaces and key not in obs:
                print(f"[RobosuiteNodeCentricObservation] Warning: Key '{key}' expected in final obs space but missing from intermediate obs.")
                # Add placeholder with correct shape/type if needed
                # final_obs[key] = np.zeros(self.observation_space[key].shape, dtype=np.float32)
            else:
                print(f"[RobosuiteNodeCentricObservation] Warning: Key '{key}' expected in final obs space but missing from intermediate obs.")


        end_time = time.time()
        # print(f"Obs wrapper time: {end_time - start_time:.6f}s")
        return final_obs

    def reset(self, **kwargs):
        """Resets the environment and processes the initial observation."""
        # Reset the underlying environment chain first
        observation = self.env.reset(**kwargs)

        # --- Re-sync metadata and recompute masks/edges after reset ---
        # Access the *correct* base env reference after potential reset changes
        self.base_env_ref = self.env
        while not isinstance(self.base_env_ref, RobosuiteEnvWrapper):
             if not hasattr(self.base_env_ref, 'env'): break # Reached top
             self.base_env_ref = self.base_env_ref.env

        if isinstance(self.base_env_ref, RobosuiteEnvWrapper):
             current_metadata = self.base_env_ref.metadata.get('robot_metadata', {})
             if not current_metadata:
                  print("[RobosuiteNodeCentricObservation] Warning: Robot metadata missing during reset in NodeCentricObservation wrapper.")
             else:
                 # Update internal state based on potentially new metadata
                 self.robot_metadata = current_metadata
                 self.robot_name = self.robot_metadata.get('robot_name', 'Unknown')
                 new_num_arm_joints = self.robot_metadata.get('num_arm_joints', 0)
                 new_num_nodes = new_num_arm_joints + 1
                 new_num_gripper_joints = self.robot_metadata.get('num_gripper_joints', 0)

                 # Recompute only if relevant properties changed
                 if (new_num_nodes != self.num_nodes or
                     new_num_arm_joints != self.num_arm_joints): # Add other checks if needed
                     print(f"[RobosuiteNodeCentricObservation][{self.robot_name}] Metadata changed on reset. Recomputing masks/edges.")
                     self.num_arm_joints = new_num_arm_joints
                     self.num_nodes = new_num_nodes
                     self.num_gripper_joints = new_num_gripper_joints
                     self.num_node_pads = self.max_limbs - self.num_nodes
                     # Regenerate masks and edges
                     self._create_padding_masks_and_edges()
                     # Update metadata dict accessible by other wrappers (like action wrapper)
                     self.metadata['act_padding_mask'] = self.act_padding_mask
                     self.metadata['num_nodes'] = self.num_nodes

        # Process the initial observation using the (potentially updated) internal state
        return self.observation(observation)



class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Wrapper to convert the padded action vector from the policy's main decoder
    (assumed to be arm actions) into the correct format for the underlying
    Robosuite environment. It assumes gripper actions are either constant or
    come from a separate source/policy head (not handled here).
    """
    def __init__(self, env):
        super().__init__(env)

        # Find the base RobosuiteEnvWrapper to get its action space and metadata
        self.base_env_ref = self.env
        while not isinstance(self.base_env_ref, RobosuiteEnvWrapper):
            if not hasattr(self.base_env_ref, 'env'):
                raise TypeError("[RobosuiteNodeCentricAction] Could not find RobosuiteEnvWrapper in the wrapper stack for ActionWrapper.")
            self.base_env_ref = self.base_env_ref.env

        self.base_action_space = self.base_env_ref.action_space
        self.robot_metadata = self.base_env_ref.metadata.get('robot_metadata', {})

        self.num_arm_actions = self.robot_metadata.get('num_arm_joints', 0)
        self.num_gripper_actions = self.robot_metadata.get('num_gripper_joints', 0) 
        self.real_action_dim = self.base_action_space.shape[0]
        # Verify consistency
        if self.num_arm_actions + self.num_gripper_actions != self.real_action_dim:
             print(f"[RobosuiteNodeCentricAction] Warning: Arm ({self.num_arm_actions}) + Gripper ({self.num_gripper_actions}) != Base Action Dim ({self.real_action_dim}). Check metadata.")
             # Adjust arm action count as fallback if gripper seems incorrect
             if self.num_gripper_actions == 0 and self.num_arm_actions != self.real_action_dim:
                  print(f"[RobosuiteNodeCentricAction] Adjusting num_arm_actions to {self.real_action_dim}")
                  self.num_arm_actions = self.real_action_dim

        self.max_limbs = cfg.MODEL.MAX_LIMBS

        # --- Define the PADDED Action Space the Policy Outputs (Main Decoder Arm Actions) ---
        # Assumes policy main decoder outputs `max_limbs` actions (one per potential node) # TODO: we will add gripper actions    
        self.padded_action_dim = self.max_limbs
        low = -1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        high = 1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32) # This wrapper defines the space the policy sees

        # Fetch the action padding mask from metadata (set by the observation wrapper)
        if 'act_padding_mask' not in self.env.metadata:
            raise ValueError("[RobosuiteNodeCentricAction] Action padding mask ('act_padding_mask') not found in env metadata. Ensure RobosuiteNodeCentricObservation runs first.")
        self.act_padding_mask = self.env.metadata['act_padding_mask']

        # Verify mask length consistency
        if len(self.act_padding_mask) != self.padded_action_dim:
            # This might happen if metadata wasn't updated correctly on reset
            print(f"[RobosuiteNodeCentricAction] Warning: Action padding mask length ({len(self.act_padding_mask)}) in metadata does not match expected padded dimension ({self.padded_action_dim}). Using expected dim for mask slicing, but this indicates a potential issue.")
            # Recreate a default mask based on current understanding - this is risky
            # is_valid = [False] + [True] * self.num_arm_actions + [False] * (self.max_limbs - (self.num_arm_actions + 1))
            # self.act_padding_mask = np.array(~np.array(is_valid), dtype=bool)


    def action(self, action):
        """
        Takes the padded action vector (arm actions) from the policy,
        un-pads it, combines with a default gripper action, and clips.
        """
        # Ensure action has the expected padded dimension
        if action.shape[-1] != self.padded_action_dim:
             raise ValueError(f"[RobosuiteNodeCentricAction] Received action shape {action.shape} doesn't match expected padded dimension {self.padded_action_dim}")

        # --- Refetch mask in case env reset changed it ---
        # This relies on the observation wrapper correctly updating metadata on reset
        current_mask = self.env.metadata.get('act_padding_mask', self.act_padding_mask)
        if len(current_mask) != self.padded_action_dim:
             print(f"[RobosuiteNodeCentricAction] Warning: Mask length mismatch after step! Using stored mask. This is likely an error.")
        else:
             self.act_padding_mask = current_mask
        #------------------------------------------------

        # Select the valid ARM action dimensions using the mask
        unpadded_arm_action = action[~self.act_padding_mask]

        # Verify the number of selected actions matches the expected arm DoFs
        if unpadded_arm_action.shape[0] != self.num_arm_actions:
            num_valid_mask = (~self.act_padding_mask).sum()
            print(
                f"[RobosuiteNodeCentricAction] ERROR: Arm action shape mismatch after unpadding! "
                f"Robot: {self.robot_metadata.get('robot_name', 'Unknown')}. "
                f"Expected {self.num_arm_actions} (from metadata), "
                f"but got {unpadded_arm_action.shape[0]} (based on mask with {num_valid_mask} valid entries). "
                f"Action received: {action.shape}, Mask: {self.act_padding_mask}"
            )
            # Fallback: Pad/truncate to expected size? Or raise error?
            # For now, pad/truncate to avoid crashing, but print a clear error.
            if unpadded_arm_action.shape[0] > self.num_arm_actions:
                 unpadded_arm_action = unpadded_arm_action[:self.num_arm_actions]
            else:
                 padded_action = np.zeros(self.num_arm_actions, dtype=np.float32)
                 padded_action[:unpadded_arm_action.shape[0]] = unpadded_arm_action
                 unpadded_arm_action = padded_action


        # --- Gripper Action ---
        # Provide a default gripper action (e.g., '1' to close)
        # TODO: Make this configurable or learnable via a separate head
        default_gripper_action = np.ones(self.num_gripper_actions, dtype=np.float32)

        # --- Combine and Clip ---
        full_action = np.concatenate([unpadded_arm_action, default_gripper_action])

        # Ensure final dimension matches base environment
        if full_action.shape[0] != self.real_action_dim:
             print(f"[RobosuiteNodeCentricAction] ERROR: Final combined action dim ({full_action.shape[0]}) != base env dim ({self.real_action_dim}).")
             # Attempt to pad/truncate as fallback
             final_action = np.zeros(self.real_action_dim, dtype=np.float32)
             copy_len = min(len(full_action), self.real_action_dim)
             final_action[:copy_len] = full_action[:copy_len]
             full_action = final_action


        clipped_action = np.clip(full_action, self.base_action_space.low, self.base_action_space.high)

        return clipped_action
