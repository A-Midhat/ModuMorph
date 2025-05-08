# metamorph/envs/wrappers/robosuite_wrappers.py
import gym
import numpy as np
from gym import spaces
from gym.spaces import Box, Dict
import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS
# from robosuite.utils.mjcf_utils import find_elements
# from metamorph.utils import mjpy as mu 

from collections import OrderedDict, defaultdict
import time #timing debug

from metamorph.config import cfg, get_list_cfg
from metamorph.utils import spaces as spu


class RobosuiteEnvWrapper(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, env_name, robot_names, controller_names, horizon, robosuite_args):
        super().__init__()
        self.robosuite_env_name = env_name
        self.robot_names_for_instance = get_list_cfg(robot_names)
        self.controller_names_for_instance = get_list_cfg(controller_names)
        self._robosuite_make_args = robosuite_args
        self._passed_horizon = horizon

        # Controller Config Creation
        self.controller_configs = []
        self.loaded_controller_names = []
        if len(self.controller_names_for_instance) == 1 and len(self.robot_names_for_instance) > 1:
            self.controller_names_for_instance = self.controller_names_for_instance * len(self.robot_names_for_instance)
        elif len(self.controller_names_for_instance) != len(self.robot_names_for_instance):
             raise ValueError("Controller names length must match robot names length")

        for name in self.controller_names_for_instance:
            # (Error handling and loading logic as before)
            if name not in ALL_CONTROLLERS: name = "JOINT_VELOCITY" # Fallback
            try:
                self.controller_configs.append(load_controller_config(default_controller=name))
                self.loaded_controller_names.append(name)
            except Exception as e:
                print(f"Warning: Error loading controller '{name}', using JOINT_VELOCITY. Error: {e}")
                self.controller_configs.append(load_controller_config(default_controller="JOINT_VELOCITY"))
                self.loaded_controller_names.append("JOINT_VELOCITY")


        # Env Instantiation
        robosuite_init_args = {
            "env_name": self.robosuite_env_name,
            "robots": self.robot_names_for_instance,
            "controller_configs": self.controller_configs,
            "horizon": self._passed_horizon,
            **self._robosuite_make_args
        }
        try:
            self.env = robosuite.make(**robosuite_init_args)
            self.num_robots = len(self.env.robots)
            if self.num_robots != len(self.robot_names_for_instance):
                 print(f"Warning: Requested {len(self.robot_names_for_instance)} robots, got {self.num_robots}.")
        except Exception as e: raise e

        self.horizon = self.env.horizon
        self._max_episode_steps = self.horizon

        # Gym Interface Setup
        act_low, act_high = self.env.action_spec
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        obs_spec = self.env.observation_spec()
        gym_obs_spaces = OrderedDict()
        for key, spec in obs_spec.items():
             if hasattr(spec, 'shape') and hasattr(spec, 'dtype'):
                 dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
                 shape = tuple(spec.shape); shape = (1,) if not shape else shape
                 gym_obs_spaces[key] = spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=dtype)
        self.observation_space = spaces.Dict(gym_obs_spaces)


        # Metadata
        self.metadata_per_robot_instance = self._extract_robot_metadata()
        self.action_dims_per_robot_instance = [md['action_dim'] for md in self.metadata_per_robot_instance]
        self.total_action_dim_instance = sum(self.action_dims_per_robot_instance)
        self.metadata["robots_metadata_list"] = self.metadata_per_robot_instance
        if self.total_action_dim_instance != self.action_space.shape[0]:
             print(f"Warning: Instance action dims ({self.total_action_dim_instance}) != space dim ({self.action_space.shape[0]}).")

        self._elapsed_steps = 0
        self.sim = self.env.sim # Expose sim for NodeCentric wrapper

    def _extract_robot_metadata(self):
        # (Implementation from Phase 1)
        metadata_list = []
        if not hasattr(self.env, "robots") or not self.env.robots: return []
        for robot_idx, robot_instance in enumerate(self.env.robots):
             metadata = {}
             try:
                 metadata["robot_name"] = robot_instance.name
                 metadata['num_arm_joints'] = robot_instance.dof - robot_instance.gripper.dof
                 metadata['num_gripper_joints'] = robot_instance.gripper.dof
                 metadata['num_nodes_estimated'] = 1 + metadata['num_arm_joints'] + 1
                 metadata['action_dim'] = robot_instance.action_dim
                 metadata['robot_joints'] = list(robot_instance.robot_joints)
                 metadata['gripper_joints'] = list(robot_instance.gripper.joints)
                 metadata_list.append(metadata)
             except Exception as e:
                 print(f"Error extracting metadata for robot {robot_idx}: {e}")
                 metadata_list.append({'robot_name': "Error", 'num_arm_joints': 0, 'num_gripper_joints': 0, 'num_nodes_estimated': 0, 'action_dim': 0, 'robot_joints': [], 'gripper_joints': []})
        return metadata_list


    def _convert_observation(self, obs_dict):
        # (Implementation from Phase 1)
        processed_obs = OrderedDict()
        for key, space in self.observation_space.spaces.items():
            if key in obs_dict:
                 value = obs_dict[key]
                 if isinstance(value, np.ndarray):
                     dtype = np.float32 if value.dtype == np.float64 else value.dtype
                     current_shape = tuple(value.shape)
                     target_shape = space.shape
                     if not current_shape and target_shape == (1,): value = value.reshape(1)
                     # Basic check, NodeCentric wrapper handles detailed feature shape validation
                     if value.shape == target_shape:
                          processed_obs[key] = value.astype(dtype)
                     # else:
                     #      print(f"Warning: Shape mismatch for key '{key}': Obs {value.shape}, Space {target_shape}. Skipping.")
                 else: # Handle non-array conversion attempt
                      try: processed_obs[key] = space.dtype.type(value)
                      except: processed_obs[key] = value # Keep original if conversion fails
            # else: # Pad missing keys if needed by downstream wrappers - NodeCentric handles its own needs
            #     processed_obs[key] = np.zeros(space.shape, dtype=space.dtype)

        processed_obs['robots_metadata_list'] = self.metadata_per_robot_instance
        return processed_obs


    def step(self, action):
        # (Implementation from Phase 1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        try:
            obs_dict, reward, done, info = self.env.step(action)
        except Exception as e: raise e # Propagate error
        self._elapsed_steps += 1
        if self._elapsed_steps >= self.horizon:
            done = True; info['TimeLimit.truncated'] = True
        processed_obs = self._convert_observation(obs_dict)
        info['robot_names'] = [md['robot_name'] for md in self.metadata_per_robot_instance]
        info['raw_reward'] = reward; info['action'] = action
        info['success'] = bool(self.env._check_success())
        info['name'] = "_".join(info['robot_names']) + f"_{self.robosuite_env_name}"
        return processed_obs, reward, done, info


    def reset(self):
        # (Implementation from Phase 1)
        self._elapsed_steps = 0
        try:
            obs_dict = self.env.reset()
            self.metadata_per_robot_instance = self._extract_robot_metadata() # Re-fetch
            self.metadata["robots_metadata_list"] = self.metadata_per_robot_instance
        except Exception as e: raise e
        processed_obs = self._convert_observation(obs_dict)
        return processed_obs

    # --- render, seed, close, sample_action (Keep implementations) ---
    def close(self):
        try: self.env.close()
        except Exception as e: print(f"Error closing Robosuite env: {e}")
    def render(self, mode='human', width=512, height=512, camera_name=None):
        # (Implementation from Phase 1)
        if camera_name is None:
             camera_name = getattr(self.env, 'render_camera', "frontview")
             if camera_name not in self.env.sim.model.camera_names and len(self.env.sim.model.camera_names)>0:
                 camera_name = self.env.sim.model.camera_names[0]

        if mode == "rgb_array":
             # Requires has_offscreen_renderer=True OR has_renderer=True
             viewer_is_offscreen = getattr(self.env, 'viewer', None) is None and \
                                   self._robosuite_make_args.get('has_offscreen_renderer', False)
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
             if self._robosuite_make_args.get('has_renderer', False):
                 try: self.env.render()
                 except Exception as e: print(f"Error human render: {e}")
             else: print("Warning: 'human' mode requires 'has_renderer=True'.")
        else: return super(RobosuiteEnvWrapper, self).render(mode=mode)


    def seed(self, seed=None):
         if hasattr(self.env, 'seed') and callable(self.env.seed): self.env.seed(seed)
         self.action_space.seed(seed)
    def sample_action(self): return self.action_space.sample()
class RobosuiteNodeCentricObservation(gym.ObservationWrapper):
    """
    Transforms observations from RobosuiteEnvWrapper into a globally padded,
    node-centric format suitable for the Transformer model.
    Handles single-arm and two-arm morphologies dynamically.
    """
    def __init__(self, env):
        # Initialize base wrapper *first* - this might call reset internally
        super().__init__(env)
        print("[Wrapper Init] RobosuiteNodeCentricObservation - Phase 2")

        # --- 1. Get Base References and Global Config ---
        if not isinstance(self.env, RobosuiteEnvWrapper):
            raise TypeError("RobosuiteNodeCentricObservation must wrap RobosuiteEnvWrapper.")
        self.base_env_ref = self.env
        self.sim = self.base_env_ref.sim # Use sim from the already initialized base env
        self.model = self.sim.model

        self.global_max_limbs = cfg.MODEL.MAX_LIMBS
        self.global_max_joints = cfg.MODEL.MAX_JOINTS

        self._define_feature_sets()
        self.limb_obs_size = self._calculate_feature_dim(self.proprio_feature_config)
        self.context_obs_size = self._calculate_feature_dim(self.context_feature_config)

        self._init_global_arrays()

        # This MUST be the observation space the wrapper exposes
        self.observation_space = self._define_observation_space()
        print(f"  [NodeCentricObs] Global Obs Space Defined. Keys: {list(self.observation_space.spaces.keys())}")

        # The base env was likely reset during super().__init__().
        # We now analyze the structure based on that initial state.
        self._structure_initialized = False # Set explicitly
        try:
            self._define_structure_and_masks() # Call *after* base env is ready
        except Exception as e:
             print(f"Error during initial structure definition in __init__: {e}")
             # Handle error, maybe raise or set a flag indicating failed init
             self._structure_initialized = False # Ensure it's False on error

    #     _define_observation_space (Keep implementations as before) ---
    def _init_global_arrays(self):
        # (Implementation from previous step)
        self.node_proprio_global = np.zeros((self.global_max_limbs, self.limb_obs_size), dtype=np.float32)
        self.node_context_global = np.zeros((self.global_max_limbs, self.context_obs_size), dtype=np.float32)
        self.edges_padded_global = np.full(2 * self.global_max_joints, self.global_max_limbs - 1, dtype=np.float32)
        self.obs_padding_mask_global = np.ones(self.global_max_limbs, dtype=bool)
        self.act_padding_mask_global = np.ones(self.global_max_limbs, dtype=bool)
        self.metadata['per_robot_action_indices'] = []


    def _define_feature_sets(self):
        self.proprio_feature_config = {
            'base': [('fixed_encoding', 3)],
            'arm': [('joint_pos_cos', 1), ('joint_pos_sin', 1), ('joint_vel', 1), ('body_pos_world', 3), ('body_quat_world', 4), ('body_velp_world', 3), ('body_velr_world', 3)],
            'hand': [('joint_pos_cos', 1), ('joint_pos_sin', 1), ('joint_vel', 1), ('eef_pos', 3), ('eef_quat', 4), ('body_velp_world', 3), ('body_velr_world', 3)],
            'gripper': [('gripper_qpos', cfg.ROBOSUITE.get('GRIPPER_DIM', 1)), ('gripper_qvel', cfg.ROBOSUITE.get('GRIPPER_DIM', 1))],
        }
        self.context_feature_config = {
             'base': [('node_type_encoding', 4), ('is_fixed', 1)],
            'arm': [('node_type_encoding', 4), ('joint_type', 1), ('joint_limits', 2), ('joint_damping', 1), ('joint_frictionloss', 1), ('joint_armature', 1), ('body_mass', 1), ('body_inertia_diag', 3), ('geom_type', 1), ('geom_size', 3)],
            'hand': [('node_type_encoding', 4), ('joint_type', 1), ('joint_limits', 2), ('joint_damping', 1), ('joint_frictionloss', 1), ('joint_armature', 1), ('body_mass', 1), ('body_inertia_diag', 3), ('geom_type', 1), ('geom_size', 3)],
            'gripper': [('node_type_encoding', 4), ('joint_type', 1), ('joint_limits', 2 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)), ('joint_damping', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)), ('joint_frictionloss', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1)), ('joint_armature', 1 * cfg.ROBOSUITE.get('GRIPPER_DIM', 1))],
        }

    def _calculate_feature_dim(self, feature_config):
        max_dim = 0
        for node_type, features in feature_config.items():
            current_dim = sum(dim for _, dim in features)
            max_dim = max(max_dim, current_dim)
        return max_dim if max_dim > 0 else 1


    def _define_observation_space(self):
        inf = np.float32(np.inf)
        obs_spaces = OrderedDict()
        obs_spaces['proprioceptive'] = Box(-inf, inf, (self.global_max_limbs * self.limb_obs_size,), np.float32)
        obs_spaces['context'] = Box(-inf, inf, (self.global_max_limbs * self.context_obs_size,), np.float32)
        obs_spaces['edges'] = Box(-inf, inf, (2 * self.global_max_joints,), np.float32)
        obs_spaces['obs_padding_mask'] = Box(False, True, (self.global_max_limbs,), dtype=bool)
        obs_spaces['act_padding_mask'] = Box(False, True, (self.global_max_limbs,), dtype=bool)

        base_obs_space = self.base_env_ref.observation_space.spaces
        global_keys_to_pass = ['object-state']
        global_keys_to_pass.extend(cfg.ROBOSUITE.get("EXTERO_KEYS", []))
        global_keys_to_pass = list(set(global_keys_to_pass))
        for key in global_keys_to_pass:
             if key in base_obs_space:
                 spec = base_obs_space[key]; shape = spec.shape; dtype = np.float32 if spec.dtype == np.float64 else spec.dtype
                 if np.prod(shape) > 0: obs_spaces[key] = Box(-inf, inf, shape, dtype)
        return Dict(obs_spaces)


    # --- Structure Definition ---
    def _define_structure_and_masks(self):
        self._init_global_arrays() # Reset global arrays first

        self.num_robots_instance = self.base_env_ref.num_robots
        self.metadata_per_robot_instance = self.base_env_ref.metadata_per_robot_instance

        self.node_start_indices = [r * cfg.MODEL.MAX_LIMBS_PER_ROBOT for r in range(self.num_robots_instance)]

        all_edges = []
        # --- Add storage for local structures ---
        self.nodes_per_robot_instance = []
        self.body_name_to_local_node_idx_per_robot = []
        self.gripper_anchor_names_per_robot = []
        self.local_node_idx_to_mujoco_joint_ids_per_robot = []
        # --- End storage ---

        for robot_idx in range(self.num_robots_instance):
            robot_instance = self.base_env_ref.env.robots[robot_idx]
            current_robot_nodes = []
            current_robot_arm_joint_names = self.metadata_per_robot_instance[robot_idx]['robot_joints']
            current_robot_gripper_joint_names = self.metadata_per_robot_instance[robot_idx]['gripper_joints']
            current_robot_body_name_to_local_node_idx = {}
            local_node_idx_to_mujoco_joint_ids = defaultdict(list)

            # 1. Base Node
            base_name = robot_instance.robot_model.root_body
            current_robot_nodes.append((base_name, 'base'))
            current_robot_body_name_to_local_node_idx[base_name] = 0 # Explicit index

            # 2. Arm Link Nodes
            last_link_name = base_name
            processed_arm_bodies = {base_name} # Track bodies added
            for j_name in current_robot_arm_joint_names:
                 try:
                     joint_id = self.model.joint_name2id(j_name)
                     body_id = self.model.jnt_bodyid[joint_id]
                     body_name = self.model.body_id2name(body_id)
                     if body_name not in processed_arm_bodies:
                         current_robot_nodes.append((body_name, 'arm'))
                         current_robot_body_name_to_local_node_idx[body_name] = len(current_robot_nodes) - 1
                         processed_arm_bodies.add(body_name)
                     last_link_name = body_name # Update last link name found
                     # Map joint to the local index of the body it drives
                     local_node_idx_to_mujoco_joint_ids[current_robot_body_name_to_local_node_idx[body_name]].append(joint_id)
                 except KeyError: continue

            # 3. Hand Node - Ensure the *last* arm link body is marked as 'hand'
            hand_name = last_link_name
            if hand_name != base_name and hand_name in current_robot_body_name_to_local_node_idx:
                 hand_local_idx = current_robot_body_name_to_local_node_idx[hand_name]
                 # Ensure its type is 'hand', even if it was added as 'arm' initially
                 current_robot_nodes[hand_local_idx] = (hand_name, 'hand')
            # else: # This case should ideally not happen if arm joints processed correctly
            #     print(f"Warning: Could not identify hand node for robot {robot_idx}.")
            #     hand_name = base_name # Fallback anchor for gripper

            # 4. Conceptual Gripper Node
            gripper_conceptual_name = hand_name + "_gripper_conceptual" # Anchor to hand node's name
            current_robot_nodes.append((gripper_conceptual_name, 'gripper'))
            gripper_local_idx = len(current_robot_nodes) - 1
            current_robot_body_name_to_local_node_idx[gripper_conceptual_name] = gripper_local_idx
            # Map gripper joints to this conceptual node
            for j_name in current_robot_gripper_joint_names:
                try: local_node_idx_to_mujoco_joint_ids[gripper_local_idx].append(self.model.joint_name2id(j_name))
                except KeyError: pass

            # Store local structures before processing next robot
            self.nodes_per_robot_instance.append(current_robot_nodes)
            self.body_name_to_local_node_idx_per_robot.append(current_robot_body_name_to_local_node_idx)
            self.gripper_anchor_names_per_robot.append(hand_name) # Store anchor name
            self.local_node_idx_to_mujoco_joint_ids_per_robot.append(local_node_idx_to_mujoco_joint_ids)

            num_nodes_this_robot = len(current_robot_nodes)
            # print(f"Debug: Robot {robot_idx} nodes ({num_nodes_this_robot}): {[n[0] for n in current_robot_nodes]}") # Debug print

            # --- Create Edges & Update Masks (Logic mostly unchanged, uses loop variables) ---
            # ... (rest of the edge creation and mask update logic as in previous snippet) ...
            # --- Create Edges for this Robot (Using Local Indices First) ---
            parent_ids = self.model.body_parentid
            num_joints_this_robot = 0 # Count edges added

            for local_node_idx, (body_name, node_type) in enumerate(current_robot_nodes):
                 # Skip base (no parent edge) and conceptual gripper (edge added separately)
                 if node_type == 'base' or node_type == 'gripper': continue
                 try:
                     child_mujoco_id = self.model.body_name2id(body_name)
                     parent_mujoco_id = parent_ids[child_mujoco_id]
                     parent_body_name = self.model.body_id2name(parent_mujoco_id)

                     if parent_body_name in current_robot_body_name_to_local_node_idx:
                         parent_local_node_idx = current_robot_body_name_to_local_node_idx[parent_body_name]
                         # Calculate GLOBAL indices for the edge list
                         global_child_idx = self.node_start_indices[robot_idx] + local_node_idx
                         global_parent_idx = self.node_start_indices[robot_idx] + parent_local_node_idx

                         # Append edge if indices are valid within global padding
                         if global_child_idx < self.global_max_limbs and global_parent_idx < self.global_max_limbs:
                              all_edges.extend([global_child_idx, global_parent_idx])
                              num_joints_this_robot += 1
                         # else: print(f"Debug: Edge ({global_child_idx}, {global_parent_idx}) out of bounds.")

                 except KeyError:
                     # This might happen if a body/parent name isn't found in the model
                     print(f"Warning: Body name '{body_name}' or its parent not found during edge creation for robot {robot_idx}.")
                     pass # Continue to next node

            # Add conceptual gripper edge (Hand -> Gripper) using GLOBAL indices
            try:
                 # Find local indices first
                 hand_local_idx = next(idx for idx, (_, type) in enumerate(current_robot_nodes) if type == 'hand')
                 gripper_local_idx = next(idx for idx, (_, type) in enumerate(current_robot_nodes) if type == 'gripper')
                 # Calculate global indices
                 global_hand_idx = self.node_start_indices[robot_idx] + hand_local_idx
                 global_gripper_idx = self.node_start_indices[robot_idx] + gripper_local_idx
                 # Append edge if valid
                 if global_hand_idx < self.global_max_limbs and global_gripper_idx < self.global_max_limbs:
                      all_edges.extend([global_gripper_idx, global_hand_idx])
                      num_joints_this_robot += 1 # Count conceptual edge towards padding limit
            except StopIteration:
                 print(f"Warning: Could not find hand or gripper node for conceptual edge in robot {robot_idx}.")
                 pass # Continue if nodes not found


            # --- Update Global Masks ---
            global_node_start = self.node_start_indices[robot_idx]
            global_node_end = global_node_start + num_nodes_this_robot
            # Unmask observation slots for this robot's real nodes
            self.obs_padding_mask_global[global_node_start:min(global_node_end, self.global_max_limbs)] = False

            # --- Create Action Mask & Store Indices for Action Wrapper ---
            robot_action_indices = {'arm': [], 'gripper': []} # Specific to this robot
            num_arm_joints_in_metadata = len(current_robot_arm_joint_names)
            num_gripper_joints_in_metadata = len(current_robot_gripper_joint_names)

            for local_node_idx, (_, node_type) in enumerate(current_robot_nodes):
                 global_node_idx = global_node_start + local_node_idx
                 if global_node_idx >= self.global_max_limbs: break # Stop if exceeding global padding

                 # Arm/Hand nodes are actuated if they correspond to an arm joint
                 if node_type == 'arm' or node_type == 'hand':
                     # Map local node index back to potential arm joint index
                     # Assumes Base=0, Link1=1 (driven by Joint0), Link2=2 (driven by Joint1) TODO: check if our assumption of the base link si good.
                     arm_joint_local_idx = local_node_idx - 1
                     if 0 <= arm_joint_local_idx < num_arm_joints_in_metadata:
                          self.act_padding_mask_global[global_node_idx] = False
                          robot_action_indices['arm'].append(global_node_idx)

                 # Gripper node is actuated if gripper joints exist
                 elif node_type == 'gripper':
                      if num_gripper_joints_in_metadata > 0:
                          self.act_padding_mask_global[global_node_idx] = False
                          robot_action_indices['gripper'].append(global_node_idx)

            # Append this robot's action indices to the list stored in metadata
            self.metadata['per_robot_action_indices'].append(robot_action_indices)

        # --- Finalize Global Edges Padding (after loop) ---
        num_real_edges = len(all_edges) // 2
        if num_real_edges > self.global_max_joints:
             # print(f"Warning: Total real edges ({num_real_edges}) > MAX_JOINTS ({self.global_max_joints}). Truncating.")
             all_edges = all_edges[:2 * self.global_max_joints]
             num_real_edges = self.global_max_joints

        num_edge_pads = self.global_max_joints - num_real_edges
        pad_value = self.global_max_limbs - 1 # Use last index as dummy
        padded_edges_flat = np.full(2 * self.global_max_joints, pad_value, dtype=np.int32)
        if all_edges: padded_edges_flat[:len(all_edges)] = np.array(all_edges, dtype=np.int32)
        self.edges_padded_global = padded_edges_flat.astype(np.float32)

        # Store final global action mask in metadata for action wrapper
        self.metadata['act_padding_mask'] = self.act_padding_mask_global.copy()
        self._structure_initialized = True


    def _extract_features_per_node(self, obs_dict):
        # (Implementation from previous step - should work if structure is initialized)
        if not hasattr(self, '_structure_initialized') or not self._structure_initialized:
             print("Error: Structure not initialized in _extract_features_per_node.")
             # Return zero arrays matching the expected global shape
             zero_prop = np.zeros((self.global_max_limbs, self.limb_obs_size), dtype=np.float32)
             zero_ctx = np.zeros((self.global_max_limbs, self.context_obs_size), dtype=np.float32)
             return zero_prop, zero_ctx

        # Reset global arrays
        self.node_proprio_global.fill(0)
        self.node_context_global.fill(0)

        # --- Iterate through Robots ---
        for robot_idx in range(self.num_robots_instance):
            robot_prefix = f"robot{robot_idx}_"
            # --- Extract Dynamic Data ---
            joint_pos_cos = obs_dict.get(f'{robot_prefix}joint_pos_cos')
            joint_pos_sin = obs_dict.get(f'{robot_prefix}joint_pos_sin')
            joint_vel = obs_dict.get(f'{robot_prefix}joint_vel')
            eef_pos = obs_dict.get(f'{robot_prefix}eef_pos')
            eef_quat = obs_dict.get(f'{robot_prefix}eef_quat')
            gripper_qpos = obs_dict.get(f'{robot_prefix}gripper_qpos')
            gripper_qvel = obs_dict.get(f'{robot_prefix}gripper_qvel')
            # --- Placeholders ---
            body_pos_world, body_quat_world, body_velp_world, body_velr_world = None, None, None, None

            # --- Iterate Local Nodes ---
            current_robot_nodes = self.nodes_per_robot_instance[robot_idx]
            gripper_anchor_name_r = self.gripper_anchor_names_per_robot[robot_idx]

            for local_node_idx, (body_name_raw, node_type) in enumerate(current_robot_nodes):
                 global_node_idx = self.node_start_indices[robot_idx] + local_node_idx
                 if global_node_idx >= self.global_max_limbs: continue

                 is_conceptual_gripper = body_name_raw.endswith("_gripper_conceptual")
                 body_name = body_name_raw if not is_conceptual_gripper else gripper_anchor_name_r
                 try: mujoco_body_id = self.model.body_name2id(body_name)
                 except KeyError: continue # Skip if body name somehow invalid

                 proprio_features = []
                 context_features = []
                 config_p = self.proprio_feature_config[node_type]
                 config_c = self.context_feature_config[node_type]
                 arm_joint_local_idx = local_node_idx - 1
                 local_node_idx_to_mujoco_joint_ids = self.local_node_idx_to_mujoco_joint_ids_per_robot[robot_idx]

                 # --- A. Extract Proprio ---
                 for key, expected_dim in config_p:
                     feature = None
                     g_dim = cfg.ROBOSUITE.get('GRIPPER_DIM', 1) # Get gripper dim
                     # (Extraction logic as before, using robot_idx specific keys/data)
                     if key == 'fixed_encoding': feature = np.array([1.0, 0.0, 0.0])
                     elif key == 'joint_pos_cos': feature = joint_pos_cos[arm_joint_local_idx:arm_joint_local_idx+1] if joint_pos_cos is not None and 0 <= arm_joint_local_idx < len(joint_pos_cos) else np.zeros(1)
                     elif key == 'joint_pos_sin': feature = joint_pos_sin[arm_joint_local_idx:arm_joint_local_idx+1] if joint_pos_sin is not None and 0 <= arm_joint_local_idx < len(joint_pos_sin) else np.zeros(1)
                     elif key == 'joint_vel': feature = joint_vel[arm_joint_local_idx:arm_joint_local_idx+1] if joint_vel is not None and 0 <= arm_joint_local_idx < len(joint_vel) else np.zeros(1)
                     elif key == 'eef_pos': feature = eef_pos if eef_pos is not None else np.zeros(3)
                     elif key == 'eef_quat': feature = eef_quat if eef_quat is not None else np.array([1.,0.,0.,0.])
                     elif key == 'gripper_qpos': feature = gripper_qpos if gripper_qpos is not None else np.zeros(g_dim)
                     elif key == 'gripper_qvel': feature = gripper_qvel if gripper_qvel is not None else np.zeros(g_dim)
                     elif key == 'body_pos_world': feature = body_pos_world[mujoco_body_id] if body_pos_world is not None else np.zeros(3) # Requires body_pos_world
                     elif key == 'body_quat_world': feature = body_quat_world[mujoco_body_id] if body_quat_world is not None else np.array([1.,0.,0.,0.]) # Requires body_quat_world
                     elif key == 'body_velp_world': feature = body_velp_world[mujoco_body_id] if body_velp_world is not None else np.zeros(3) # Requires body_velp_world
                     elif key == 'body_velr_world': feature = body_velr_world[mujoco_body_id] if body_velr_world is not None else np.zeros(3) # Requires body_velr_world
                     else: feature = np.zeros(expected_dim)
                     proprio_features.append(np.asarray(feature).flatten())

                 node_proprio_vec = np.concatenate(proprio_features).astype(np.float32)
                 # Pad/truncate to self.limb_obs_size
                 current_prop_dim = node_proprio_vec.size
                 if current_prop_dim != self.limb_obs_size:
                    padded_prop = np.zeros(self.limb_obs_size, dtype=np.float32); copy_len = min(current_prop_dim, self.limb_obs_size)
                    padded_prop[:copy_len] = node_proprio_vec[:copy_len]; node_proprio_vec = padded_prop
                 self.node_proprio_global[global_node_idx, :] = node_proprio_vec


                 type_encoding = {'base': 0, 'arm': 1, 'hand': 2, 'gripper': 3}
                 node_type_one_hot = np.zeros(4); node_type_one_hot[type_encoding[node_type]] = 1.0
                 mujoco_joint_ids_for_node = local_node_idx_to_mujoco_joint_ids.get(local_node_idx, [])

                 for key, expected_dim in config_c:
                     feature = None
                     g_dim = cfg.ROBOSUITE.get('GRIPPER_DIM', 1)
                     if key == 'node_type_encoding': feature = node_type_one_hot
                     elif key == 'is_fixed': feature = np.array([1.0])
                     elif key in ['joint_type', 'joint_limits', 'joint_damping', 'joint_frictionloss', 'joint_armature']:
                         joint_id_for_context = None; is_list = False
                         if node_type == 'gripper':
                             gripper_jnt_names = self.metadata_per_robot_instance[robot_idx]['gripper_joints']
                             if gripper_jnt_names: 
                                try: 
                                    joint_id_for_context = [self.model.joint_name2id(j) for j in gripper_jnt_names];
                                    is_list=True
                                except KeyError: 
                                    pass
                         elif mujoco_joint_ids_for_node: joint_id_for_context = mujoco_joint_ids_for_node[0] # Use first driving joint

                         if joint_id_for_context is not None:
                             j_id0 = joint_id_for_context[0] if is_list else joint_id_for_context
                             if key == 'joint_type': feature = np.array([self.model.jnt_type[j_id0]])
                             elif key == 'joint_limits': feature = np.concatenate([self.model.jnt_range[j_id] for j_id in joint_id_for_context]) if is_list else self.model.jnt_range[j_id0]
                             elif key in ['joint_damping', 'joint_frictionloss', 'joint_armature']:
                                 dof_ids = []
                                 if is_list:
                                     num_gripper_dof = len(joint_id_for_context) # Approx
                                     if num_gripper_dof > 0: dof_start = self.model.jnt_dofadr[j_id0]; dof_ids = list(range(dof_start, dof_start + num_gripper_dof))
                                 else: dof_ids = [self.model.jnt_dofadr[j_id0]]
                                 if dof_ids:
                                     if key == 'joint_damping': feature = self.model.dof_damping[dof_ids]
                                     elif key == 'joint_frictionloss': feature = self.model.dof_frictionloss[dof_ids]
                                     elif key == 'joint_armature': feature = self.model.dof_armature[dof_ids]
                         if feature is None: feature = np.zeros(expected_dim)
                     elif key == 'body_mass': feature = np.array([self.model.body_mass[mujoco_body_id]])
                     elif key == 'body_inertia_diag': feature = self.model.body_inertia[mujoco_body_id]
                     elif key == 'geom_type' or key == 'geom_size':
                         geom_ids = [g_id for g_id, b_id in enumerate(self.model.geom_bodyid) if b_id == mujoco_body_id]
                         if geom_ids:
                              g_id = geom_ids[0]
                              if key == 'geom_type': feature = np.array([self.model.geom_type[g_id]])
                              elif key == 'geom_size': feature = self.model.geom_size[g_id] # Shape (3,)
                         if feature is None: feature = np.zeros(1 if key == 'geom_type' else 3)
                         # Ensure geom_size is always length 3
                         if key == 'geom_size':
                              current_size = np.asarray(feature).flatten(); padded_size = np.zeros(3)
                              copy_len = min(len(current_size), 3); padded_size[:copy_len]=current_size[:copy_len]; feature=padded_size
                     else: feature = np.zeros(expected_dim)
                     context_features.append(np.asarray(feature).flatten())


                 node_context_vec = np.concatenate(context_features).astype(np.float32)
                 # Pad/truncate to self.context_obs_size
                 current_ctx_dim = node_context_vec.size
                 if current_ctx_dim != self.context_obs_size:
                    padded_ctx = np.zeros(self.context_obs_size, dtype=np.float32); copy_len = min(current_ctx_dim, self.context_obs_size)
                    padded_ctx[:copy_len] = node_context_vec[:copy_len]; node_context_vec = padded_ctx
                 self.node_context_global[global_node_idx, :] = node_context_vec

        return self.node_proprio_global, self.node_context_global


    # --- observation() and reset() methods ---
    def observation(self, obs):
        """Processes the raw obs dict into the final node-centric format."""
        if not hasattr(self, '_structure_initialized') or not self._structure_initialized:
            # print("Structure not initialized, defining now...")
            try:
                self._define_structure_and_masks()
            except Exception as e:
                print(f"Error defining structure in observation(): {e}")
                # Fallback: return zeroed observation matching the space
                return {k: np.zeros(s.shape, dtype=s.dtype) for k, s in self.observation_space.spaces.items()}

        # Extract features into global arrays
        try:
             node_proprio, node_context = self._extract_features_per_node(obs)
        except Exception as e:
             print(f"Error during feature extraction: {e}. Returning zero arrays.")
             node_proprio = np.zeros_like(self.node_proprio_global)
             node_context = np.zeros_like(self.node_context_global)

        # Assemble final dict
        final_obs = OrderedDict()
        final_obs['proprioceptive'] = node_proprio.flatten()
        final_obs['context'] = node_context.flatten()
        final_obs['edges'] = self.edges_padded_global
        final_obs['obs_padding_mask'] = self.obs_padding_mask_global
        final_obs['act_padding_mask'] = self.metadata.get('act_padding_mask', np.ones(self.global_max_limbs, dtype=bool)) # Fetch from metadata

        # Pass Through Global Features
        for key in self.observation_space.spaces:
             if key not in final_obs and key in obs:
                 if obs[key].shape == self.observation_space[key].shape: final_obs[key] = obs[key]

        # Ensure all keys present
        for key, space in self.observation_space.spaces.items():
             if key not in final_obs: final_obs[key] = np.zeros(space.shape, dtype=space.dtype)

        return final_obs

    def reset(self, **kwargs):
        """Resets the base environment and re-analyzes the structure."""
        observation_raw = self.env.reset(**kwargs)
        # Structure needs to be re-defined based on the new state/robots
        self._structure_initialized = False # Mark as needing update
        # The actual update happens in the first call to observation()
        return self.observation(observation_raw)



class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Maps the policy's globally padded, node-centric action output to the base
    RoboSuite environment's flat action space for the current morphology instance.
    """
    def __init__(self, env):
        # Initialize base wrapper first
        super().__init__(env)
        print("[Wrapper Init] RobosuiteNodeCentricAction - Phase 3")

        # We need access to the Obs wrapper to get metadata (masks, indices)
        # and the Base wrapper to get the base action space dims.
        self.obs_wrapper_ref = self._find_wrapper_ref(RobosuiteNodeCentricObservation)
        self.base_env_ref = self._find_wrapper_ref(RobosuiteEnvWrapper)

        if not self.obs_wrapper_ref or not self.base_env_ref:
            raise TypeError("RobosuiteNodeCentricAction requires RobosuiteNodeCentricObservation and RobosuiteEnvWrapper in the stack.")

        # Global padding limits from cfg
        self.global_max_limbs = cfg.MODEL.MAX_LIMBS

        # The policy outputs one action value potentially relevant to each *global* node slot.
        # The action mask will determine which ones are actually used.
        self.padded_action_dim_global = self.global_max_limbs
        low = -1.0 * np.ones(self.padded_action_dim_global, dtype=np.float32)
        high = 1.0 * np.ones(self.padded_action_dim_global, dtype=np.float32)
        # This wrapper *redefines* the action space the agent interacts with.
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # These will be updated on reset via _update_mapping_info()
        self.current_act_mask = None
        self.current_per_robot_action_indices = None
        self.current_action_dims_per_robot = None
        self.current_total_action_dim = None
        self.current_num_robots = None
        self.current_base_action_space = self.base_env_ref.action_space # Store initial

        # Initialize mapping info based on the env state *after* super().__init__
        self._update_mapping_info()


    def _find_wrapper_ref(self, wrapper_type):
        """Helper to find a specific wrapper type in the stack below this one."""
        env = self.env # Start from the env this wrapper wraps
        while hasattr(env, 'env'):
            if isinstance(env, wrapper_type):
                return env
            env = env.env
        # Check the final env itself
        if isinstance(env, wrapper_type):
             return env
        print(f"Warning: Could not find {wrapper_type.__name__} in wrapper stack below NodeCentricAction.")
        return None

    def _update_mapping_info(self):
        """Fetches the latest masks, indices, and action dims needed for mapping."""
        # Fetch from Obs wrapper's metadata (which should be updated on its reset)
        self.current_act_mask = self.obs_wrapper_ref.metadata.get('act_padding_mask')
        self.current_per_robot_action_indices = self.obs_wrapper_ref.metadata.get('per_robot_action_indices')

        # Fetch from Base wrapper's properties/metadata
        self.current_action_dims_per_robot = self.base_env_ref.action_dims_per_robot_instance
        self.current_total_action_dim = self.base_env_ref.total_action_dim_instance
        self.current_num_robots = self.base_env_ref.num_robots
        self.current_base_action_space = self.base_env_ref.action_space # Get current base space

        if self.current_act_mask is None or \
           self.current_per_robot_action_indices is None or \
           not isinstance(self.current_per_robot_action_indices, list) or \
           len(self.current_per_robot_action_indices) != self.current_num_robots:
            raise ValueError("Action wrapper could not fetch valid mapping metadata from Observation wrapper.")
        if len(self.current_act_mask) != self.padded_action_dim_global:
             raise ValueError(f"Action mask length ({len(self.current_act_mask)}) doesn't match global padded dim ({self.padded_action_dim_global}).")
        if sum(self.current_action_dims_per_robot) != self.current_total_action_dim:
             print(f"Warning: Sum of per-robot action dims != total action dim in _update_mapping_info.")
        if self.current_total_action_dim != self.current_base_action_space.shape[0]:
            print(f"Warning: Total action dim != base action space shape in _update_mapping_info.")


        # Inside RobosuiteNodeCentricAction
    def action(self, action):
        """Maps the policy's padded action to the base environment's action space."""
        action = np.asarray(action)

        if action.shape != self.action_space.shape:
             raise ValueError(f"Input action shape {action.shape} != expected wrapper action space {self.action_space.shape}")

        if self.current_act_mask is None:
            # print("Warning: Action mapping info not initialized. Attempting update.") # Optional
            self._update_mapping_info()

        try:
            # Mask is True for padded/non-actuated slots
            valid_action_values = action[~self.current_act_mask]
        except IndexError as e:
             print(f"Error applying action mask: {e}. Mask shape: {self.current_act_mask.shape}, Action shape: {action.shape}")
             return np.zeros_like(self.current_base_action_space.low)

        per_robot_final_actions = []
        current_index_in_valid = 0

        for robot_idx in range(self.current_num_robots):
            # Get expected action dims for *this* robot's controller
            expected_arm_dim = self.base_env_ref.metadata_per_robot_instance[robot_idx]['num_arm_joints'] # This metadata key might need adjustment if based on DoF not joints
            expected_gripper_dim = self.base_env_ref.metadata_per_robot_instance[robot_idx]['num_gripper_joints'] # This metadata key might need adjustment
            # *** Get the TRUE action dim from the base wrapper ***
            expected_total_dim_r = self.current_action_dims_per_robot[robot_idx]
            # *** Calculate true arm dim based on total and gripper ***
            true_expected_arm_dim = expected_total_dim_r - expected_gripper_dim


            # Determine how many values to *read* from valid_action_values for this robot
            num_actuated_arm_nodes = len(self.current_per_robot_action_indices[robot_idx]['arm'])
            num_actuated_gripper_nodes = len(self.current_per_robot_action_indices[robot_idx]['gripper']) # Should be 0 or 1
            num_values_to_read = num_actuated_arm_nodes + num_actuated_gripper_nodes

            end_index = current_index_in_valid + num_values_to_read
            if end_index > len(valid_action_values):
                 print(f"Error: Not enough values in valid_action_values ({len(valid_action_values)}) to reconstruct action for robot {robot_idx} (needs {num_values_to_read} starting from {current_index_in_valid}).")
                 per_robot_final_actions.append(np.zeros(expected_total_dim_r, dtype=np.float32))
                 current_index_in_valid = len(valid_action_values)
                 continue

            values_for_robot = valid_action_values[current_index_in_valid : end_index]
            current_index_in_valid = end_index

            arm_actions_raw = values_for_robot[:num_actuated_arm_nodes]
            gripper_action_raw = values_for_robot[num_actuated_arm_nodes:]

            if len(arm_actions_raw) < true_expected_arm_dim:
                 # This shouldn't happen if node unmasking is correct, but handle defensively
                 print(f"Warning: Robot {robot_idx} read {len(arm_actions_raw)} arm values, but expected {true_expected_arm_dim}. Padding with zeros.")
                 arm_actions_final = np.zeros(true_expected_arm_dim, dtype=np.float32)
                 copy_len = len(arm_actions_raw)
                 arm_actions_final[:copy_len] = arm_actions_raw
            elif len(arm_actions_raw) > true_expected_arm_dim:
                 # Take only the first needed values if policy outputs more than needed (e.g., for 7 nodes but 6 OSC DoF)
                 # print(f"Info: Robot {robot_idx} taking first {true_expected_arm_dim} of {len(arm_actions_raw)} arm values for OSC.")
                 arm_actions_final = arm_actions_raw[:true_expected_arm_dim]
            else: # Length matches exactly
                 arm_actions_final = arm_actions_raw

            # (Keep the logic from previous step - repeat single value if needed)
            if len(gripper_action_raw) == 1 and expected_gripper_dim >= 1:
                 gripper_action_final = np.repeat(gripper_action_raw[0], expected_gripper_dim)
            elif len(gripper_action_raw) == expected_gripper_dim:
                 gripper_action_final = gripper_action_raw
            elif expected_gripper_dim == 0 and len(gripper_action_raw) == 0:
                 gripper_action_final = np.array([], dtype=np.float32)
            else: # Fallback
                 # print(f"Warning: Robot {robot_idx} gripper action size mismatch ({len(gripper_action_raw)} vs {expected_gripper_dim}). Using default -1.")
                 gripper_action_final = -1.0 * np.ones(expected_gripper_dim, dtype=np.float32)


            final_action_r = np.concatenate([arm_actions_final, gripper_action_final])

            if final_action_r.shape[0] != expected_total_dim_r:
                 print(f"ERROR: Final action dim for robot {robot_idx} ({final_action_r.shape[0]}) != expected ({expected_total_dim_r}). Correcting.") # Corrected print
                 corrected_action_r = np.zeros(expected_total_dim_r, dtype=np.float32)
                 copy_len = min(final_action_r.shape[0], expected_total_dim_r)
                 corrected_action_r[:copy_len] = final_action_r[:copy_len]
                 final_action_r = corrected_action_r

            per_robot_final_actions.append(final_action_r)

        try:
            final_action_global = np.concatenate(per_robot_final_actions)
        except ValueError as e:
             print(f"Error concatenating actions: {e}"); return np.zeros_like(self.current_base_action_space.low)

        if final_action_global.shape[0] != self.current_total_action_dim:
             print(f"ERROR: Final combined action dim ({final_action_global.shape[0]}) != base env dim ({self.current_total_action_dim}). Using zero action.")
             return np.zeros_like(self.current_base_action_space.low)

        clipped_action = np.clip(final_action_global,
                                 self.current_base_action_space.low,
                                 self.current_base_action_space.high)

        return clipped_action.astype(self.current_base_action_space.dtype)

    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        self._update_mapping_info()
        return ret

class RobosuiteMLPFlattener(gym.ObservationWrapper):
    """
    Flattens selected observations (robot proprioceptive, object state)
    from the *currently active* morphology instance into a single vector.
    Pads this vector to a pre-calculated maximum dimension
    (cfg.MODEL.MAX_FLAT_OBS_DIM) across all training morphologies.
    Adds dummy masks/edges using global padding limits.
    Designed for use with MLP policies.
    """
    def __init__(self, env):
        super().__init__(env)
        print("[Wrapper Init] RobosuiteMLPFlattener - Phase 4")

        if not isinstance(self.env, RobosuiteEnvWrapper):
             raise TypeError("RobosuiteMLPFlattener must wrap RobosuiteEnvWrapper.")
        self.base_env_ref = self.env

        self.max_flat_obs_dim = cfg.MODEL.get('MAX_FLAT_OBS_DIM')
        if self.max_flat_obs_dim is None or self.max_flat_obs_dim <= 0:
            print("ERROR: cfg.MODEL.MAX_FLAT_OBS_DIM not set or invalid! This must be pre-calculated.")
            print("       Attempting estimation based on initial env (may be incorrect).")
            try:
                initial_obs_spec = self.base_env_ref.env.observation_spec()
                keys_to_flatten_init = self._get_keys_to_flatten() # Keys for *this* instance
                self.max_flat_obs_dim = sum(
                    np.prod(initial_obs_spec[k].shape)
                    for k in keys_to_flatten_init if k in initial_obs_spec and hasattr(initial_obs_spec[k], 'shape')
                )
                if self.max_flat_obs_dim == 0: raise ValueError("Zero estimate")
                print(f"  Fallback Estimated MAX_FLAT_OBS_DIM: {self.max_flat_obs_dim}")
            except Exception as e:
                 print(f"  Fallback estimation failed: {e}. Using hardcoded 100.")
                 self.max_flat_obs_dim = 100 # Hardcoded fallback

        self.global_max_limbs = cfg.MODEL.MAX_LIMBS
        self.global_max_joints = cfg.MODEL.MAX_JOINTS

        inf = np.float32(np.inf)
        dmsk = (self.global_max_limbs,)
        dedg = (2 * self.global_max_joints,)
        flat_prop_shape = (self.max_flat_obs_dim,) # Use pre-calculated max dim

        self.observation_space = spaces.Dict({
            "proprioceptive": Box(-inf, inf, flat_prop_shape, np.float32),
            "obs_padding_mask": Box(False, True, dmsk, dtype=bool),
            "act_padding_mask": Box(False, True, dmsk, dtype=bool),
            "edges": Box(-inf, inf, dedg, dtype=np.float32),
            "context": Box(-inf, inf, flat_prop_shape, dtype=np.float32), # Context mirrors proprio
        })
        self.action_space = env.action_space # Action space is passthrough

        # No need for cached obs, reset handles fetching

    def _get_keys_to_flatten(self):
        """Dynamically determines which keys to flatten based on num_robots in current instance."""
        # Relies on the base wrapper having the correct num_robots for the *current* env instance
        num_robots = self.base_env_ref.num_robots
        keys = [f"robot{i}_proprio-state" for i in range(num_robots)]
        # Always attempt to include object state
        keys.append("object-state")
        # Add other potential keys if needed
        # keys.extend(cfg.ROBOSUITE.get("MLP_EXTRA_FLATTEN_KEYS", []))
        return keys

    def _flatten_and_pad(self, obs_dict):
        """Flattens relevant keys from obs_dict and pads to self.max_flat_obs_dim."""
        pieces = []
        keys_to_use = self._get_keys_to_flatten() # Get keys relevant to *this* instance

        for k in keys_to_use:
            if k in obs_dict and isinstance(obs_dict[k], np.ndarray) and obs_dict[k].size > 0:
                pieces.append(obs_dict[k].ravel())
            # else: # Skip keys not present in this instance's obs_dict

        if not pieces:
            # If no relevant keys found (e.g., error in obs_dict), return zeros
            return np.zeros(self.max_flat_obs_dim, dtype=np.float32)

        flat_vec = np.concatenate(pieces).astype(np.float32)
        current_dim = flat_vec.shape[0]

        # Pad or truncate to the pre-calculated global max dimension
        if current_dim < self.max_flat_obs_dim:
            padded_vec = np.zeros(self.max_flat_obs_dim, dtype=np.float32)
            padded_vec[:current_dim] = flat_vec
            return padded_vec
        elif current_dim > self.max_flat_obs_dim:
            print(f"Warning: Flattened dim {current_dim} > pre-calculated MAX_FLAT_OBS_DIM {self.max_flat_obs_dim}. Truncating.")
            return flat_vec[:self.max_flat_obs_dim]
        else:
            return flat_vec # Dimensions match

    def observation(self, obs_dict):
        """Processes the raw observation dict from the base wrapper."""
        flat_padded_prop = self._flatten_and_pad(obs_dict)

        # Create dummy elements using global padding limits
        # For MLP, masks are often ignored, so simple zeros/False might suffice.
        # Using zeros for masks to maintain type consistency if needed elsewhere.
        dummy_obs_mask = np.zeros(self.global_max_limbs, dtype=bool)
        dummy_act_mask = np.zeros(self.global_max_limbs, dtype=bool)
        dummy_edges = np.zeros(2 * self.global_max_joints, dtype=np.float32)
        # Context mirrors the flattened proprioceptive state
        dummy_ctx = flat_padded_prop.copy()

        # Ensure context has the correct shape defined in the space
        if dummy_ctx.shape != self.observation_space['context'].shape:
             # This should only happen if max_flat_obs_dim calculation was wrong
             print(f"Warning: MLP Context shape mismatch ({dummy_ctx.shape} vs {self.observation_space['context'].shape}). Adjusting.")
             ctx_dim = self.observation_space['context'].shape[0]
             adjusted_ctx = np.zeros(ctx_dim, dtype=np.float32)
             copy_len = min(dummy_ctx.shape[0], ctx_dim)
             adjusted_ctx[:copy_len] = dummy_ctx[:copy_len]
             dummy_ctx = adjusted_ctx


        return OrderedDict({
            "proprioceptive": flat_padded_prop,
            "obs_padding_mask": dummy_obs_mask,
            "act_padding_mask": dummy_act_mask,
            "edges": dummy_edges,
            "context": dummy_ctx, # Padded context mirroring proprio
        })

    def reset(self, **kwargs):
        # Reset base env to get potentially new obs structure/keys
        obs_dict_raw = self.env.reset(**kwargs)
        # Process the raw observation using the updated instance info
        return self.observation(obs_dict_raw)

    # Action is passthrough
    def action(self, action):
        return action

# --- Test Code ---
# if __name__ == "__main__":
#     # ... (SR/TR tests for NodeCentric wrappers as before) ...

#     print("\n--- Testing MLPFlattener (Phase 4) ---")

#     # --- Config for MLP SR Test ---
#     cfg.defrost()
#     cfg.ENV_NAME = "Robosuite-v0"; cfg.ROBOSUITE.ENV_NAME = "Lift"
#     cfg.ROBOSUITE.ROBOTS = ["Panda"]; cfg.ROBOSUITE.CONTROLLERS = ["JOINT_VELOCITY"]
#     cfg.ROBOSUITE.ENV_ARGS.horizon = 50
#     # ** Set a pre-calculated MAX_FLAT_OBS_DIM (replace with actual calculated value later) **
#     # Panda Lift JOINT_VEL proprio is 30, object-state is 14 -> Flat = 44
#     # TwoArm Lift OSC_POSE proprio is (19*2), object-state is 19 -> Flat = 57 (approx)
#     # Let's set max high enough for testing both scenarios below
#     cfg.MODEL.MAX_FLAT_OBS_DIM = 80
#     cfg.MODEL.MAX_LIMBS = 22 # Use global padding from TR test
#     cfg.MODEL.MAX_JOINTS = 20 # Use global padding from TR test
#     cfg.freeze()

#     print("\n--- Testing MLP Wrapper SR (Panda Lift) ---")
#     base_env_mlp_sr = RobosuiteEnvWrapper(
#         env_name=cfg.ROBOSUITE.ENV_NAME, robot_names=cfg.ROBOSUITE.ROBOTS,
#         controller_names=cfg.ROBOSUITE.CONTROLLERS, horizon=cfg.ROBOSUITE.ENV_ARGS.horizon,
#         robosuite_args=dict(cfg.ROBOSUITE.ENV_ARGS)
#     )
#     mlp_wrapped_env_sr = RobosuiteMLPFlattener(base_env_mlp_sr)

#     print("[MLP Wrapped Env SR] Action Space:", mlp_wrapped_env_sr.action_space)
#     print("[MLP Wrapped Env SR] Obs Space:", mlp_wrapped_env_sr.observation_space)

#     obs_mlp_sr = mlp_wrapped_env_sr.reset()
#     print("\n[MLP Wrapped Env SR Reset] Obs Keys:", list(obs_mlp_sr.keys()))
#     print(f"  Proprioceptive shape: {obs_mlp_sr['proprioceptive'].shape} (Expected ({cfg.MODEL.MAX_FLAT_OBS_DIM},))")
#     print(f"  Context shape: {obs_mlp_sr['context'].shape} (Expected ({cfg.MODEL.MAX_FLAT_OBS_DIM},))")
#     print(f"  Edges shape: {obs_mlp_sr['edges'].shape} (Expected ({2 * cfg.MODEL.MAX_JOINTS},))")
#     print(f"  Masks shape: {obs_mlp_sr['obs_padding_mask'].shape} (Expected ({cfg.MODEL.MAX_LIMBS},))")
#     assert obs_mlp_sr['proprioceptive'].shape[0] == cfg.MODEL.MAX_FLAT_OBS_DIM
#     assert obs_mlp_sr['context'].shape[0] == cfg.MODEL.MAX_FLAT_OBS_DIM
#     assert obs_mlp_sr['edges'].shape[0] == 2 * cfg.MODEL.MAX_JOINTS
#     assert obs_mlp_sr['obs_padding_mask'].shape[0] == cfg.MODEL.MAX_LIMBS

#     mlp_wrapped_env_sr.close()


#     # --- Config for MLP TR Test ---
#     cfg.defrost()
#     cfg.ROBOSUITE.ENV_NAME = "TwoArmLift"
#     cfg.ROBOSUITE.ROBOTS = ["Panda", "Panda"]
#     cfg.ROBOSUITE.CONTROLLERS = ["OSC_POSE", "OSC_POSE"]
#     # Use the same pre-calculated MAX values
#     cfg.MODEL.MAX_FLAT_OBS_DIM = 80
#     cfg.MODEL.MAX_LIMBS = 22
#     cfg.MODEL.MAX_JOINTS = 20
#     cfg.freeze()

#     print("\n--- Testing MLP Wrapper TR (TwoArmLift, Panda, Panda) ---")
#     try:
#         base_env_mlp_tr = RobosuiteEnvWrapper(
#             env_name=cfg.ROBOSUITE.ENV_NAME, robot_names=cfg.ROBOSUITE.ROBOTS,
#             controller_names=cfg.ROBOSUITE.CONTROLLERS, horizon=cfg.ROBOSUITE.ENV_ARGS.horizon,
#             robosuite_args=dict(cfg.ROBOSUITE.ENV_ARGS)
#         )
#         mlp_wrapped_env_tr = RobosuiteMLPFlattener(base_env_mlp_tr)

#         print("[MLP Wrapped Env TR] Action Space:", mlp_wrapped_env_tr.action_space)
#         print("[MLP Wrapped Env TR] Obs Space:", mlp_wrapped_env_tr.observation_space)

#         obs_mlp_tr = mlp_wrapped_env_tr.reset()
#         print("\n[MLP Wrapped Env TR Reset] Obs Keys:", list(obs_mlp_tr.keys()))
#         print(f"  Proprioceptive shape: {obs_mlp_tr['proprioceptive'].shape} (Expected ({cfg.MODEL.MAX_FLAT_OBS_DIM},))")
#         assert obs_mlp_tr['proprioceptive'].shape[0] == cfg.MODEL.MAX_FLAT_OBS_DIM

#         mlp_wrapped_env_tr.close()
#     except Exception as e:
#         print(f"Error during TR MLP test: {e}")


#     print("\n--- Wrapper Test (Phase 4) Complete ---")