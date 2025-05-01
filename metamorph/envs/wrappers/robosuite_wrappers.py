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

    def __init__(self, robosuite_env_name, robot_name, horizon=100, controller_name=None, robosuite_cfg=None):
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
            "ignore_done": False, # Use PPO for termniation (TODO:Uncomment)
            "use_camera_obs": False, # TODO: Set True to use camera observations as exteroceptive observations
            "control_freq":20,
            "hard_reset": False, # TODO: check if it is really faster during training 
            "horizon":100,
            "reward_shaping": True
        }

        robosuite_init_args.update(self.robosuite_cfg.get("ENV_ARGS", {}))
        print("Robosuite init args: ", robosuite_init_args)

        # Create the env
        self.env = robosuite.make(**robosuite_init_args)
        
        self.horizon = self.env.horizon
        self._max_episode_steps = self.horizon


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
        self._elapsed_steps = 0 # for timelimt wrapper  

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
            #done = False # TODO: UNCOMMNET
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
        info['success'] = self.env._check_success()

        self._elapsed_steps += 1

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
                img = self.env.sim.render(
                    camera_name=cam_name,
                    width=width,
                    height=height,
                    depth=False
                )
                img = img[::-1]             
                return img  
            else :
                print("Warning: Offscreen rendering is not available for this environment")
                img =  self.env.sim.render(
                    mode='offscreen',
                    width=width,
                    height=height,
                    depth=False
                ) # alt return zeros (blank)
                img = img[::-1]             
                return img 
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
    """
    Transforms RoboSuite observations into a node-centric format suitable
    for graph-based policies like Transformers. Uses a link-based node
    representation derived via DFS for a single robot. Calculates and
    stores metadata needed for the corresponding action wrapper.
    """
    def __init__(self, env):
        super().__init__(env)

        # --- Find Base Wrapper ---
        if not isinstance(self.env, RobosuiteEnvWrapper):
             parent = self.env
             is_base_wrapper = False
             while hasattr(parent, 'env'):
                  if isinstance(parent, RobosuiteEnvWrapper):
                      is_base_wrapper = True
                      break
                  parent = parent.env
             if not is_base_wrapper:
                  raise TypeError(f"{self.__class__.__name__} must wrap RobosuiteEnvWrapper.")

        self.base_env_ref = self.env
        while not isinstance(self.base_env_ref, RobosuiteEnvWrapper):
            if not hasattr(self.base_env_ref, 'env'):
                 raise TypeError(f"{self.__class__.__name__}: Could not find RobosuiteEnvWrapper.")
            self.base_env_ref = self.base_env_ref.env

        # --- Config and Robot Info ---
        self.robot_metadata = self.base_env_ref.metadata.get('robot_metadata', {})
        if not self.robot_metadata:
            raise ValueError("[RobosuiteNodeCentricObservation] Robot metadata missing.")
        self.robot_name = self.robot_metadata.get('robot_name', 'Unknown')

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        self.max_joints = cfg.MODEL.MAX_JOINTS
        self.robots = self.base_env_ref.env.robots
        if len(self.robots) != 1:
             warnings.warn(f"{self.__class__.__name__} expects 1 robot, found {len(self.robots)}. Using first one.")
        self.robot = self.robots[0]
        self._root_body_name = self.robot.robot_model.root_body

        # --- Derive Structure, Feature Dims, and Observation Space ---
        self._derive_node_structure_and_dims() # Defines self.observation_space internally

        # --- Checks and Padding Calculation ---
        if self.num_nodes > self.max_limbs:
             raise ValueError(f"Robot {self.robot_name} needs {self.num_nodes} nodes, but MAX_LIMBS={self.max_limbs}.")
        num_real_edge_pairs = len(self.edges_unpadded) // 2
        if num_real_edge_pairs > self.max_joints:
             warnings.warn(f"Robot {self.robot_name} has {num_real_edge_pairs} edges > MAX_JOINTS={self.max_joints}. Edges truncated.")

        self.num_node_pads = self.max_limbs - self.num_nodes

        # --- Create Masks, Padded Edges, and Action Wrapper Metadata ---
        self._create_padding_masks_and_metadata()
        self.metadata['act_padding_mask'] = self.act_padding_mask # Share with action wrapper

        print(f"[{self.__class__.__name__}][{self.robot_name}] Init complete. Nodes: {self.num_nodes}.")

    # --- Feature Dimension Calculation ---
    def _get_proprio_feature_dim(self):
        # Rel Pos(3) + Rel Quat(4) + Rel LinVel(3) + Rel AngVel(3) + Joint Pos SinCos(2) + Joint Vel(1) = 16
        return 16

    def _get_context_feature_dim(self):
        # Mass(1) + Inertia(3) + GeomType(1) + GeomSize(3) + JointType(1) + JointRange(2) + JointDamp(1) + JointArmature(1) = 13
        return 13

    # --- Structure Derivation ---
    def _find_parent_joint_info(self, body_id, model):
        """Finds the joint connecting this body to its parent."""
        parent_body_id = model.body_parentid[body_id]
        if parent_body_id <= 0: return None, -1, -1, -1

        for jnt_id in range(model.njnt):
            if model.jnt_bodyid[jnt_id] == body_id:
                 # Check if the joint's logical parent body matches the kinematic parent
                 # This relies on the assumption that the joint connects body_id to parent_body_id
                 # A more robust check might involve comparing joint pos/axis with body frames.
                 # For standard robot arms, this check is usually sufficient.
                 if model.body_parentid[model.jnt_bodyid[jnt_id]] == parent_body_id:
                     qpos_adr = model.jnt_qposadr[jnt_id]
                     qvel_adr = model.jnt_qveladr[jnt_id]
                     # dof_adr can be -1 if the joint has no degrees of freedom (e.g., weld)
                     dof_adr = model.jnt_dofadr[jnt_id] if model.jnt_dofadr[jnt_id] >= 0 else -1
                     return jnt_id, qpos_adr, qvel_adr, dof_adr
        return None, -1, -1, -1

    def _derive_structure_dfs(self, body_id, parent_node_idx, model, visited_body_ids):
        """Recursive helper for DFS traversal for a single robot's tree."""
        body_name = model.body_id2name(body_id)

        # Filter out bodies not explicitly listed in the robot model's definition
        # This helps ignore potentially extraneous bodies in the world XML
        if body_name not in self.robot.robot_model.bodies:
             return

        current_node_idx = len(self.node_names)
        self.node_names.append(body_name)
        self.body_id_to_node_idx[body_id] = current_node_idx
        self.node_idx_to_body_id.append(body_id)
        visited_body_ids.add(body_id)

        if parent_node_idx != -1:
            self.edge_list_tuples.append((current_node_idx, parent_node_idx))

        # Recurse for children
        for child_body_id in range(model.nbody):
            if model.body_parentid[child_body_id] == body_id and child_body_id not in visited_body_ids:
                 self._derive_structure_dfs(child_body_id, current_node_idx, model, visited_body_ids)


    def _derive_node_structure_and_dims(self):
        """Determines node/edge structure using DFS and defines observation space."""
        try:
            sim = self.base_env_ref.env.sim
            model = sim.model
        except AttributeError:
            raise RuntimeError("[RobosuiteNodeCentricObservation] Cannot access sim or model.")

        root_body_id = model.body_name2id(self._root_body_name)
        self._root_body_id = root_body_id

        self.node_names = []
        self.node_idx_to_body_id = []
        self.body_id_to_node_idx = {}
        self.edge_list_tuples = []
        visited_body_ids = set()

        self._derive_structure_dfs(root_body_id, -1, model, visited_body_ids)

        self.num_nodes = len(self.node_names)
        self.edges_unpadded = np.array(self.edge_list_tuples, dtype=np.int32).flatten()

        self.limb_obs_size = self._get_proprio_feature_dim()
        self.context_obs_size = self._get_context_feature_dim()
        self.observation_space = self._define_observation_space()

    def _create_padding_masks_and_metadata(self):
        """Generates padding masks, pads edges, and creates metadata for action wrapper."""
        self.obs_padding_mask = np.asarray([False] * self.num_nodes + [True] * self.num_node_pads, dtype=bool)

        # Pad Edges
        real_edges = self.edges_unpadded
        num_real_edge_pairs = len(real_edges) // 2
        self.num_joint_pads = self.max_joints - num_real_edge_pairs
        pad_value = -1
        padded_edges = np.full(2 * self.max_joints, pad_value, dtype=np.int32)
        if len(real_edges) > 0:
             copy_len = min(len(real_edges), len(padded_edges))
             padded_edges[:copy_len] = real_edges[:copy_len]
        self.edges = padded_edges.astype(np.float32)

        # Create Action Padding Mask & Action Wrapper Metadata
        sim = self.base_env_ref.env.sim
        model = sim.model
        is_valid_action_node = np.zeros(self.num_nodes, dtype=bool)
        # Map: Node Index -> Index in base env's action vector (-1 if not actuated)
        node_to_base_act_map = np.full(self.num_nodes, -1, dtype=np.int32)
        gripper_node_idx = -1

        base_act_dim = self.base_env_ref.action_space.shape[0]
        # Iterate through actuators to find corresponding nodes and build map
        for act_idx in range(base_act_dim):
            # actuator_trntype tells if it's joint, tendon, etc.
            # actuator_trnid tells the specific joint/tendon id
            if model.actuator_trntype[act_idx] == mujoco_py.const.OBJ_JOINT:
                joint_id = model.actuator_trnid[act_idx, 0]
                body_id = model.jnt_bodyid[joint_id] # Body the joint controls
                if body_id in self.body_id_to_node_idx:
                    node_idx = self.body_id_to_node_idx[body_id]
                    is_valid_action_node[node_idx] = True
                    node_to_base_act_map[node_idx] = act_idx
                    # Identify gripper node (use first gripper joint's body)
                    if self.robot.gripper and joint_id == model.joint_name2id(self.robot.gripper.joints[0]):
                        gripper_node_idx = node_idx
            # TODO: Add handling for other actuator types if needed (e.g., tendons)

        # Ensure root node action is invalid
        root_node_idx = self.body_id_to_node_idx.get(self._root_body_id, -1)
        if root_node_idx != -1:
            is_valid_action_node[root_node_idx] = False

        num_valid_actions = is_valid_action_node.sum()
        padded_is_valid = np.concatenate((is_valid_action_node, [False] * self.num_node_pads))
        self.act_padding_mask = ~padded_is_valid.astype(bool)

        # Store metadata
        self.metadata['gripper_node_idx'] = gripper_node_idx
        self.metadata['node_to_base_act_indices'] = node_to_base_act_map.tolist() # Store as list for JSON compatibility if needed

        # Validation checks
        if len(self.act_padding_mask) != self.max_limbs:
             raise ValueError("Internal Error: Action padding mask length mismatch.")
        if (~self.act_padding_mask).sum() != num_valid_actions:
             warnings.warn(f"Action mask count mismatch. Derived {num_valid_actions}, mask has {(~self.act_padding_mask).sum()}.")
        if gripper_node_idx == -1 and self.robot.gripper:
             warnings.warn("Could not identify gripper node index for action mapping.")


    def _define_observation_space(self):
        """Defines the final, padded observation space dictionary."""
        inf = np.float32(np.inf)
        obs_spaces = OrderedDict()
        use_flattened_obs = False # Use un-flattened: (max_limbs, feature_dim)

        if use_flattened_obs:
            obs_spaces['proprioceptive'] = Box(-inf, inf, (self.max_limbs * self.limb_obs_size,), np.float32)
            obs_spaces['context'] = Box(-inf, inf, (self.max_limbs * self.context_obs_size,), np.float32)
        else:
            obs_spaces['proprioceptive'] = Box(-inf, inf, (self.max_limbs, self.limb_obs_size), np.float32)
            obs_spaces['context'] = Box(-inf, inf, (self.max_limbs, self.context_obs_size), np.float32)

        obs_spaces['edges'] = Box(-np.inf, np.inf, (2 * self.max_joints,), np.float32)
        obs_spaces['obs_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool)
        obs_spaces['act_padding_mask'] = Box(False, True, (self.max_limbs,), dtype=bool)

        # Pass through extroceptive keys
        base_obs_space_dict = self.base_env_ref.observation_space.spaces
        extro_keys_to_check = ['object-state'] # Global keys
        # Add potentially robot-specific keys IF they are present in the base space
        # (The intermediate obs dict from RobosuiteEnvWrapper should provide these)
        potential_robot_keys = ['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
        extro_keys_to_check.extend([k for k in potential_robot_keys if k in base_obs_space_dict])

        for key in extro_keys_to_check:
            if key in base_obs_space_dict:
                 spec = base_obs_space_dict[key]
                 if np.prod(spec.shape) > 0:
                     dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
                     obs_spaces[key] = Box(-inf, inf, spec.shape, dtype)

        return Dict(obs_spaces)

    # --- Feature Extraction Implementation ---
    def _calculate_node_features(self, node_idx, sim):
        """Extracts dynamic (proprio) and static (context) features for a node."""
        proprio = np.zeros(self.limb_obs_size, dtype=np.float32)
        context = np.zeros(self.context_obs_size, dtype=np.float32)

        model = sim.model
        data = sim.data
        body_id = self.node_idx_to_body_id[node_idx]
        root_id = self._root_body_id

        # --- Proprioceptive Features (16 dims) ---
        prop_idx = 0
        # 1. Relative Position (3)
        node_pos_world = data.body_xpos[body_id]
        base_pos_world = data.body_xpos[root_id]
        base_mat_world_inv = data.body_xmat[root_id].reshape(3, 3).T
        rel_pos_base = (node_pos_world - base_pos_world) @ base_mat_world_inv
        proprio[prop_idx:prop_idx+3] = rel_pos_base
        prop_idx += 3

        # 2. Relative Orientation (Quat, 4)
        node_quat_world = data.body_xquat[body_id]
        base_quat_world = data.body_xquat[root_id]
        q_base_inv = mu.quat_conj(base_quat_world)
        rel_quat = mu.quat_mul(q_base_inv, node_quat_world)
        proprio[prop_idx:prop_idx+4] = rel_quat
        prop_idx += 4

        # 3. Relative Linear Velocity (3)
        node_lin_vel_world = data.body_xvelp[body_id]
        base_lin_vel_world = data.body_xvelp[root_id]
        rel_lin_vel_base = (node_lin_vel_world - base_lin_vel_world) @ base_mat_world_inv
        proprio[prop_idx:prop_idx+3] = rel_lin_vel_base
        prop_idx += 3

        # 4. Relative Angular Velocity (3)
        node_ang_vel_world = data.body_xvelr[body_id]
        base_ang_vel_world = data.body_xvelr[root_id]
        rel_ang_vel_base = (node_ang_vel_world - base_ang_vel_world) @ base_mat_world_inv
        proprio[prop_idx:prop_idx+3] = rel_ang_vel_base
        prop_idx += 3

        # 5. Parent Joint Position (SinCos, 2) & Velocity (1)
        parent_jnt_id, qpos_adr, qvel_adr, _ = self._find_parent_joint_info(body_id, model)
        if parent_jnt_id is not None and qpos_adr != -1 and qvel_adr != -1:
             qpos = data.qpos[qpos_adr] # Assumes 1 DoF joint
             proprio[prop_idx] = np.sin(qpos)
             proprio[prop_idx+1] = np.cos(qpos)
             proprio[prop_idx+2] = data.qvel[qvel_adr] # Assumes 1 DoF joint
             # TODO: Handle multi-DoF joints if needed
        # else: Keep zeros for root
        prop_idx += 3

        if prop_idx != self.limb_obs_size:
             warnings.warn(f"Proprio feature mismatch node {node_idx}: expected {self.limb_obs_size}, got {prop_idx}")

        # --- Context Features (13 dims) ---
        cont_idx = 0
        # 1. Link Mass (1)
        context[cont_idx] = model.body_mass[body_id]
        cont_idx += 1
        # 2. Link Inertia Diagonals (3)
        context[cont_idx:cont_idx+3] = model.body_inertia[body_id]
        cont_idx += 3
        # 3. Geometry Type (1) & Size (3) (Using first geom)
        geom_type_enc = -1.0; geom_size = np.zeros(3)
        body_geomadr = model.body_geomadr[body_id]
        if body_geomadr != -1:
             geom_id = body_geomadr
             g_type = model.geom_type[geom_id]
             if g_type == mujoco_py.const.GEOM_SPHERE: geom_type_enc = 0.0
             elif g_type == mujoco_py.const.GEOM_CAPSULE: geom_type_enc = 1.0
             elif g_type == mujoco_py.const.GEOM_BOX: geom_type_enc = 2.0
             elif g_type == mujoco_py.const.GEOM_MESH: geom_type_enc = 3.0
             elif g_type == mujoco_py.const.GEOM_CYLINDER: geom_type_enc = 4.0
             geom_size[:] = model.geom_size[geom_id]
        context[cont_idx] = geom_type_enc; cont_idx += 1
        context[cont_idx:cont_idx+3] = geom_size; cont_idx += 3
        # 4. Parent Joint Properties: Type(1), Range(2), Damping(1), Armature(1)
        parent_jnt_id, _, _, dof_adr = self._find_parent_joint_info(body_id, model) # Reuse result
        joint_type_enc = 0.0; joint_range = np.zeros(2); joint_damping = 0.0; joint_armature = 0.0
        if parent_jnt_id is not None:
            j_type = model.jnt_type[parent_jnt_id]
            if j_type == mujoco_py.const.JNT_HINGE: joint_type_enc = 1.0
            elif j_type == mujoco_py.const.JNT_SLIDE: joint_type_enc = 2.0
            elif j_type == mujoco_py.const.JNT_BALL: joint_type_enc = 3.0
            joint_range[:] = model.jnt_range[parent_jnt_id]
            if dof_adr != -1:
                 joint_damping = model.dof_damping[dof_adr]
                 joint_armature = model.dof_armature[dof_adr]
        context[cont_idx] = joint_type_enc; cont_idx += 1
        context[cont_idx:cont_idx+2] = joint_range; cont_idx += 2
        context[cont_idx] = joint_damping; cont_idx += 1
        context[cont_idx] = joint_armature; cont_idx += 1

        if cont_idx != self.context_obs_size:
             warnings.warn(f"Context feature mismatch node {node_idx}: expected {self.context_obs_size}, got {cont_idx}")

        return proprio.astype(np.float32), context.astype(np.float32)

    # --- Main Observation Processing ---
    def observation(self, obs):
        """Processes the intermediate observation dict into the final node-centric format."""
        node_proprio = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)
        node_context = np.zeros((self.max_limbs, self.context_obs_size), dtype=np.float32)
        sim = self.base_env_ref.env.sim

        for i in range(self.num_nodes):
            proprio, context = self._calculate_node_features(i, sim)
            node_proprio[i, :] = proprio
            node_context[i, :] = context

        final_obs = OrderedDict()
        use_flattened_obs = (len(self.observation_space['proprioceptive'].shape) == 1)

        if use_flattened_obs:
             final_obs['proprioceptive'] = node_proprio.flatten()
             final_obs['context'] = node_context.flatten()
        else:
             final_obs['proprioceptive'] = node_proprio
             final_obs['context'] = node_context

        final_obs['edges'] = self.edges
        final_obs['obs_padding_mask'] = self.obs_padding_mask
        final_obs['act_padding_mask'] = self.act_padding_mask

        # Pass Through Extroceptive Keys
        for key in self.observation_space.spaces:
            if key not in final_obs and key in obs:
                 final_obs[key] = obs[key].astype(np.float32)

        return final_obs

    def reset(self, **kwargs):
        """Resets the environment and processes the initial observation."""
        observation = self.env.reset(**kwargs)

        # --- Re-sync metadata if robot potentially changed (less likely in single-robot focus) ---
        self.base_env_ref = self.env
        while not isinstance(self.base_env_ref, RobosuiteEnvWrapper):
             if not hasattr(self.base_env_ref, 'env'): break
             self.base_env_ref = self.base_env_ref.env

        if isinstance(self.base_env_ref, RobosuiteEnvWrapper):
             if self.base_env_ref.env.robots[0].name != self.robot.name:
                  warnings.warn(f"[{self.__class__.__name__}] Robot change detected on reset. Re-deriving structure.")
                  self.robots = self.base_env_ref.env.robots
                  self.robot = self.robots[0]
                  self._root_body_name = self.robot.robot_model.root_body
                  self._derive_node_structure_and_dims()
                  self._create_padding_masks_and_metadata()
                  self.metadata['act_padding_mask'] = self.act_padding_mask

        return self.observation(observation)




# metamorph/envs/wrappers/robosuite_wrappers.py
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict
import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
from collections import OrderedDict
import time
import warnings
import mujoco_py

from metamorph.config import cfg
from metamorph.utils import spaces as spu
from metamorph.utils import mjpy as mu

# (RobosuiteEnvWrapper, RobosuiteMLPFlattener, RobosuiteNodeCentricObservation classes should be defined above this)

class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Maps the padded, node-centric actions output by the policy back to the
    unpadded, ordered action vector expected by the base Robosuite environment.
    Handles single-robot case.
    """
    def __init__(self, env):
        super().__init__(env)

        # --- Find Base Wrapper ---
        self.base_env_ref = self.env
        is_node_centric_obs_wrapper_found = False
        while True:
             # Check if the current env has the necessary metadata first
             if isinstance(getattr(self.base_env_ref, 'metadata', None), dict) and \
                'act_padding_mask' in self.base_env_ref.metadata:
                  is_node_centric_obs_wrapper_found = True
                  # We need the RobosuiteEnvWrapper *below* the NodeCentricObservation
                  if isinstance(self.base_env_ref, RobosuiteNodeCentricObservation):
                       if hasattr(self.base_env_ref, 'env') and isinstance(self.base_env_ref.env, RobosuiteEnvWrapper):
                            self.base_env_ref_robosuite = self.base_env_ref.env
                            break
                       else:
                            # This case shouldn't happen if structure is correct
                             raise TypeError("RobosuiteNodeCentricObservation does not wrap RobosuiteEnvWrapper.")
             # Move down the wrapper stack
             if hasattr(self.base_env_ref, 'env'):
                  self.base_env_ref = self.base_env_ref.env
             else:
                  # Reached the bottom without finding the needed wrapper/metadata
                  break

        if not is_node_centric_obs_wrapper_found:
             raise TypeError(f"{self.__class__.__name__} requires metadata ('act_padding_mask', etc.) "
                             "set by RobosuiteNodeCentricObservation. Ensure it runs first.")
        if not hasattr(self, 'base_env_ref_robosuite'):
             raise TypeError(f"{self.__class__.__name__}: Could not find base RobosuiteEnvWrapper.")


        # --- Get Action Space Info & Config ---
        self.base_action_space = self.base_env_ref_robosuite.action_space
        self.real_action_dim = self.base_action_space.shape[0]
        self.base_action_low = self.base_action_space.low
        self.base_action_high = self.base_action_space.high

        self.max_limbs = cfg.MODEL.MAX_LIMBS
        # Read action dim per node from config (default to 1)
        self.action_dim_per_node = getattr(cfg.MODEL, "ACTION_DIM_PER_NODE", 1)
        self.padded_action_dim = self.max_limbs * self.action_dim_per_node

        # --- Define the Action Space this wrapper *accepts* (from policy) ---
        # Policy outputs values typically in [-1, 1] range
        policy_output_low = -1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        policy_output_high = 1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        self.action_space = Box(low=policy_output_low, high=policy_output_high, dtype=np.float32)

        # --- Get Metadata from Observation Wrapper ---
        self._update_metadata() # Initial fetch

        # --- Precompute base action indices for arm vs gripper ---
        self._compute_base_action_indices()


    def _update_metadata(self):
        """Refreshes metadata potentially updated by the observation wrapper."""
        metadata = self.env.metadata # Access metadata from the env we directly wrap
        self.act_padding_mask = np.asarray(metadata.get('act_padding_mask'), dtype=bool)
        self.gripper_node_idx = metadata.get('gripper_node_idx', -1)
        # Map: Base Action Index -> Node Index (-1 if not mapped)
        self.node_to_base_act_indices = metadata.get('node_to_base_act_indices', [])

        if self.act_padding_mask is None or self.node_to_base_act_indices is None:
             raise ValueError(f"{self.__class__.__name__}: Metadata missing from observation wrapper.")
        if len(self.act_padding_mask) != self.max_limbs:
             warnings.warn(f"Action mask length {len(self.act_padding_mask)} != MAX_LIMBS {self.max_limbs}. Metadata might be stale.")

        # Store the number of valid actions the policy should be controlling
        self.num_valid_policy_actions = (~self.act_padding_mask).sum()

    def _compute_base_action_indices(self):
        """Identify which indices in the base action vector are arm vs gripper."""
        robot = self.base_env_ref_robosuite.robot # Get robot object
        sim = self.base_env_ref_robosuite.env.sim
        model = sim.model

        self.base_arm_indices = []
        self.base_gripper_indices = []

        # Determine indices based on actuator joint names
        for act_idx in range(self.real_action_dim):
            if model.actuator_trntype[act_idx] == mujoco_py.const.OBJ_JOINT:
                joint_id = model.actuator_trnid[act_idx, 0]
                joint_name = model.joint_id2name(joint_id)
                is_gripper_joint = robot.gripper and joint_name in robot.gripper.joints
                if is_gripper_joint:
                    self.base_gripper_indices.append(act_idx)
                else:
                    # Assume joints not part of the gripper are arm joints
                    # This might need refinement for robots with other actuated parts
                    is_arm_joint = joint_name in robot.robot_joints
                    if is_arm_joint:
                         self.base_arm_indices.append(act_idx)
                    else:
                         warnings.warn(f"Actuator {act_idx} (joint {joint_name}) is not identified as arm or gripper.")
            # Add handling for other actuator types if necessary

        self.num_gripper_actions = len(self.base_gripper_indices)
        self.num_arm_actions = len(self.base_arm_indices)

        if self.num_arm_actions + self.num_gripper_actions != self.real_action_dim:
             warnings.warn(f"Mismatch: Arm({self.num_arm_actions}) + Gripper({self.num_gripper_actions}) != Total({self.real_action_dim}).")


    def _scale_action(self, action):
        """Scales action from [-1, 1] to environment's action range."""
        low = self.base_action_low
        high = self.base_action_high
        return low + 0.5 * (action + 1.0) * (high - low)

    def action(self, action):
        """
        Transforms the policy's padded node-centric action into the base
        environment's expected action format.
        """
        action_original_shape = action.shape
        action = np.asarray(action, dtype=np.float32)
        is_batched = action.ndim > 1
        if not is_batched:
            action = action[None] # Add batch dimension
        batch_size = action.shape[0]

        if action.shape[1] != self.padded_action_dim:
             raise ValueError(f"Received action shape {action.shape} inconsistent with expected padded dim {self.padded_action_dim}")

        # Ensure mask is boolean
        mask = self.act_padding_mask.astype(bool)
        valid_action_mask = ~mask # True where actions are valid

        # Initialize the final action array for the base environment
        final_action = np.zeros((batch_size, self.real_action_dim), dtype=np.float32)

        # --- Extract and Place Gripper Action(s) ---
        if self.gripper_node_idx != -1 and self.num_gripper_actions > 0:
            # Assuming action_dim_per_node for now
            start_idx = self.gripper_node_idx * self.action_dim_per_node
            end_idx = start_idx + self.action_dim_per_node
            gripper_policy_output = action[:, start_idx:end_idx] # (batch, action_dim_per_node)

            # Map to potentially multiple gripper DoFs in base env
            if self.action_dim_per_node == self.num_gripper_actions:
                final_action[:, self.base_gripper_indices] = gripper_policy_output
            elif self.action_dim_per_node == 1 and self.num_gripper_actions > 1:
                # Replicate single output to all gripper dims (common for parallel jaws)
                final_action[:, self.base_gripper_indices] = gripper_policy_output.repeat(self.num_gripper_actions, axis=1)
            else:
                 warnings.warn(f"Cannot map gripper node output ({self.action_dim_per_node} dims) to base gripper action ({self.num_gripper_actions} dims). Gripper action set to zero.")
                 final_action[:, self.base_gripper_indices] = 0.0 # Fallback

        elif self.gripper_node_idx == -1 and self.num_gripper_actions > 0:
             warnings.warn("Gripper node index not found, but base env expects gripper actions. Setting gripper action to default (1.0).")
             final_action[:, self.base_gripper_indices] = 1.0 # Default action (e.g., close)

        # --- Extract, Reorder, and Place Arm Actions ---
        # node_to_base_act_indices maps base_act_idx -> node_idx
        # We need the inverse: node_idx -> base_act_idx for valid arm nodes
        node_is_valid_arm = valid_action_mask.copy()
        if self.gripper_node_idx != -1:
             node_is_valid_arm[self.gripper_node_idx] = False # Exclude gripper node

        # Iterate through the *base environment's* arm action indices
        for base_idx in self.base_arm_indices:
             # Find which node index corresponds to this base action index
             corresponding_node_idx = -1
             for node_i, base_i in enumerate(self.node_to_base_act_indices):
                 if base_i == base_idx:
                     corresponding_node_idx = node_i
                     break

             if corresponding_node_idx != -1 and node_is_valid_arm[corresponding_node_idx]:
                  # Extract the action for this node from the policy output
                  start_idx = corresponding_node_idx * self.action_dim_per_node
                  end_idx = start_idx + self.action_dim_per_node
                  node_action = action[:, start_idx:end_idx]

                  # Place it in the correct position in the final action vector
                  # This assumes arm joints are 1 DoF
                  final_action[:, base_idx] = node_action.squeeze(-1) # Remove last dim if action_dim_per_node is 1
             else:
                  # This happens if the node mapping is wrong or the node was invalid
                  warnings.warn(f"Could not find valid node action for base arm index {base_idx}. Setting to zero.")
                  final_action[:, base_idx] = 0.0

        # --- Scale and Clip ---
        scaled_action = self._scale_action(final_action)
        clipped_action = np.clip(scaled_action, self.base_action_low, self.base_action_high)

        # Return in original shape (batched or single)
        return clipped_action if is_batched else clipped_action[0]

    def reset(self, **kwargs):
        """Resets the environment and refreshes metadata."""
        obs = self.env.reset(**kwargs)
        try:
             self._update_metadata()
             # Optional: Recompute base action indices if robot could change
             # self._compute_base_action_indices()
        except Exception as e:
             warnings.warn(f"Error updating metadata in {self.__class__.__name__}.reset(): {e}")
        return obs

env = RobosuiteEnvWrapper(
    robosuite_env_name="Lift", 
    robot_name="Panda",
)

env = RobosuiteNodeCentricObservation(env)

print(env.observation_space)
print("\n\n\n", env.metadata)