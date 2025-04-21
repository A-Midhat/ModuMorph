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
        super().__init__()
        self.robosuite_env_name = robosuite_env_name
        self.robot_name = robot_name
        self.robosuite_cfg = robosuite_cfg if robosuite_cfg is not None else {}

        # --- Controller Setup --- 
        controller_name = controller_name 
        if controller_name not in ALL_CONTROLLERS:
            print("Warning: controller_name not in ALL_CONTROLLERS. Using default controller. (JOINT_POSITION)")
            self.controller_name = "JOINT_POSITION"
        try: 
            self.controller_config = load_controller_config(default_controller=controller_name)
        except Exception as e:
            print(f"Error loading controller_config: {e}")
            self.controller_config = load_controller_config(default_controller="JOINT_POSITION")

        # --- Env Setup --- 
        # TODO: 1.check if more/less args needed, 2.Add to config file
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
        print("Robosuite init args: ", robosuite_init_args)

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
                print(f"Error processing obs spec: {e}")
                continue
        self.observation_space = spaces.Dict(gym_obs_spaces)
        # ---Metadata--- 
        self._robot_metadata = self._extract_robot_metadata()
        self.metadata["robot_metadata"] = self._robot_metadata 
        # for multi-robot envs, we will use robot_name 
        #TODO: add some kind of unique Id (when using same robot but differen kinamtics)
        self.metadata["robot_name"] = self.robot_name
        self.ep_step_count = 0 # optional 

    def _extract_robot_metadata(self):
        """
        Extracts static metadata from the robot.
        # TODO: add more metadata if needed 
        """

        metadata = {}
        if hasattr(self.env, "robots") and len(self.env.robots)>0:
            robot = self.env.robots[0]
            metadata["robot_name"] = robot.name 
            metadata['num_arm_joints'] = len(robot._ref_joint_indexes) # use _ref_joint_indexes instead of arm_joint_names, returns list of len (num of joints) without gripper
            metadata['num_gripper_joints'] = robot.gripper.dof # for 1 gripper it returns 1, if i want the real joints we can use i.e. robot._gripper_joint_pos_indexes (2)
            
            # Number of nodes (DoF + Gripper)
            metadata['num_nodes'] = metadata['num_arm_joints'] + metadata['num_gripper_joints']
            # TODO: add the following (lower, upper, damping, armature, friction) physical properties (check dir(robot))
            # metadata['joint_names'] = robot.robot_joints 
            # metadata['g_joint_name'] = robot.gripper_joints 
            # alt 
            # metadata['joint_names'] = robot.robot_joints + robot.gripper_joints  ? 

        else: 
            print("Warning: No robot found in the environment. Metadata will set to None.")
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
            print(f"ERROR during Robosuite step: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Action=`{action}`, Error={e}")
            raise e # for debugging 
            # TODO: add a flag to skip the step and return None for training stablity (UNCOMMENT)
            #obs_dict = self.observation_space.sample() # dummy obs
            #reward = -10 # heavily penelized 
            # done = True 
            #info = {'error': str(e), 'name': self.robot_name}
        
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
            print(f"ERROR during Robosuite reset: Robot=`{self.robot_name}`, Env=`{self.robosuite_env_name}`, Error={e}")
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
        robot_proprio_key = f'robot0_proprio-state' # change the '0' to the robot index if we have multiple robots (guess)
        if robot_proprio_key in obs_dict:
            prelim_obs[robot_proprio_key] = obs_dict[robot_proprio_key]
        else:
            print(f"Warning: {robot_proprio_key} state not found for robot {self.robot_name} in the observation dictionary.")
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
                    print(f"Missing key: {key}")
                    # skip for now 
                    pass 
            if not parts:
                # create a dummy observation (zeros)
                # Problem we need to know the shape of the observation
                print(f"ERROR: Could not assemble robot-state for {self.robot_name}.")
                prelim_obs['robot_state'] = np.zeros(1, dtype=np.float32) # leads to problem better to raise an error
            else :
                prelim_obs['robot_state'] = np.concatenate(parts)
        
        # ---Object State---
        if 'object-state' in obs_dict:
            prelim_obs['object_state'] = obs_dict['object-state'].astype(np.float32)
        else:
            # Either Env doesnt have object or is it disabled
            print(f"Warning: object-state not found in the observation dictionary.")
            # create a dummy observation (zeros)
            prelim_obs['object_state'] = np.zeros([], dtype=np.float32)
        
        # TODO: Combined states (proprio + obj) for MLP (single robot)
        #combined_state = np.concatenate([obs_dict[robot_proprio_key], obs_dict['object-state']])
        # --------------------------------------------------------------------------------------
        # ---Gripper to Object Distance---
        # TODO: alt we can use the eef_pos and gripper_to_object directly
        obj_pos_key = 'object-state' # gripper to obj is the last 3 elemtns in object-state 
        eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
        if obj_pos_key in obs_dict and eef_pos_key in obs_dict:
            obj_pos = obs_dict[obj_pos_key][-3:]
            eef_pos = obs_dict[eef_pos_key]
            prelim_obs['gripper_to_object'] = (obj_pos - eef_pos).astype(np.float32)
        else:
            print(f"Warning: {obj_pos_key} or {eef_pos_key} not found in the observation dictionary.")
            # create a dummy observation (zeros)
            prelim_obs['gripper_to_object'] = np.zeros(3, dtype=np.float32) # TODO: better to set high value 

        # ---EEF State--- (TODO: add this Ad a direct input to EXT_MIX)
        eef_pos_key = f'robot0_eef_pos' # change `0` to the robot index if we have multiple robots (guess)
        if eef_pos_key in obs_dict:
            prelim_obs['eef_pos'] = obs_dict[eef_pos_key].astype(np.float32)
        else:
            print(f"Warning: {eef_pos_key} not found in the observation dictionary.")
            # create a dummy observation (zeros)
            prelim_obs['eef_pos'] = np.zeros(3, dtype=np.float32)
        
        eef_quat_key = f'robot0_eef_quat' # change `0` to the robot index if we have multiple robots (guess)
        if eef_quat_key in obs_dict:
            prelim_obs['eef_quat'] = obs_dict[eef_quat_key].astype(np.float32)
        else:
            print(f"Warning: {eef_quat_key} not found in the observation dictionary.")
            prelim_obs['eef_quat'] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32) # Default Identity 
        
        # --- Robot Context (Static Info) --- 
        # features extracted from the _robot_metadata so node-centric wrapper can easily distribute it to all nodes
        prelim_obs['robot_context_dict'] = self.metadata['robot_metadata']

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
        # Normally robosuite handles this when robosuite.make is called TODO:(check if this is needed )

        if hasattr(self.env, "seed") and callable(self.env.seed):
            self.env.seed(seed)
        self.action_space.seed(seed)

class RobosuiteNodeCentricObservation(gym.ObservationWrapper):
    """
    Wrapper to convert Robosuite observations into a padded, node-centric format
    suitable for the ModuMorph transformer model, following the simplified plan.

    Input: Observation dictionary from RobosuiteEnvWrapper (prelim_obs).
    Output: Padded, node-centric observation dictionary for the policy.
    """
    def __init__(self, env):
        super().__init__(env)

        # Ensure the underlying env is our base wrapper
        if not isinstance(self.env, RobosuiteEnvWrapper):
             raise TypeError("RobosuiteNodeCentricObservation must wrap a RobosuiteEnvWrapper instance.")

        self.robot_metadata = self.env.metadata.get('robot_metadata', {})
        if not self.robot_metadata:
            raise ValueError("Robot metadata not found in underlying env.")

        # --- Define Node Structure (Simplified Plan) ---
        # Node 0: Base (Placeholder features)
        # Node 1-N: Arm Link/Joint (N=num_arm_joints)
        self.num_arm_joints = self.robot_metadata.get('num_arm_joints', 0)
        self.num_nodes = self.num_arm_joints + 1 # Base node + one node per arm joint/link
        self.num_gripper_joints = self.robot_metadata.get('num_gripper_joints', 0) # Needed for state extraction

        # --- Configured Maximums ---
        self.max_limbs = cfg.MODEL.MAX_LIMBS # Max sequence length (nodes)
        self.max_joints = cfg.MODEL.MAX_JOINTS # Max controllable joints (for edges)

        if self.num_nodes > self.max_limbs:
             raise ValueError(f"Robot {self.env.robot_name} needs {self.num_nodes} nodes, but config MODEL.MAX_LIMBS is only {self.max_limbs}.")
        if self.num_arm_joints > self.max_joints: # Check arm joints against max_joints for edges
             raise ValueError(f"Robot {self.env.robot_name} has {self.num_arm_joints} arm joints, but config MODEL.MAX_JOINTS is only {self.max_joints}.")


        # Calculate padding required
        self.num_node_pads = self.max_limbs - self.num_nodes

        # --- Determine Per-Node Feature Sizes (Simplified Plan) ---
        self.limb_obs_size = self._get_proprio_feature_dim() # Proprioceptive features per node
        self.context_obs_size = self._get_context_feature_dim() # Context features per node
        print(f"[Wrapper {self.env.robot_name}] Nodes: {self.num_nodes}, ProprioDim/Node: {self.limb_obs_size}, ContextDim/Node: {self.context_obs_size}")

        # --- Create Padding Masks and Edge List ---
        self._create_padding_masks_and_edges()
        # Store masks in metadata for the action wrapper and potentially the model
        self.metadata['act_padding_mask'] = self.act_padding_mask # Used by RobosuiteNodeCentricAction
        self.metadata['num_nodes'] = self.num_nodes # Pass actual node count

        # --- Define the NEW Observation Space ---
        self.observation_space = self._define_observation_space()
        print(f"[Wrapper {self.env.robot_name}] Final Observation Space Defined.")


    def _get_proprio_feature_dim(self):
        """Calculate the dimension of proprioceptive features per node (Simplified Plan)."""
        # Node 0 (Base): Placeholder, same size as others for simplicity for now.
        # Node i (1-N): Joint i state [sin(pos), cos(pos), vel] = 3 dims.
        return 3

    def _get_context_feature_dim(self):
        """Calculate the dimension of context features per node (Simplified Plan)."""
        # Node 0 (Base): Placeholder, same size as others.
        # Node i (1-N): Joint i static props [limit_low, limit_high, damping, armature, frictionloss] = 5 dims.
        return 5

    def _create_padding_masks_and_edges(self):
        """Generates padding masks and the padded edge list based on the simplified plan."""
        # --- Obs Padding Mask (for Transformer input) ---
        self.obs_padding_mask = np.asarray([False] * self.num_nodes + [True] * self.num_node_pads, dtype=bool)

        # --- Edge List (Kinematic Chain 0->1->...->N) ---
        # Edges represent parent-child relationships between nodes.
        # Node 0 is base, Node 1 is link 1 connected via joint 1, etc.
        num_real_edges = self.num_nodes - 1 # Base doesn't have a parent *within the robot*
        real_edges = []
        if num_real_edges > 0:
            # Edges: [child1, parent1, child2, parent2, ...]
            # Joint 1 connects base (node 0) to link 1 (node 1) -> edge (1, 0)
            # Joint 2 connects link 1 (node 1) to link 2 (node 2) -> edge (2, 1)
            # ...
            # Joint N connects link N-1 (node N-1) to link N (node N) -> edge (N, N-1)
            parents = list(range(self.num_nodes - 1)) # Nodes 0 to N-1 are parents
            children = list(range(1, self.num_nodes)) # Nodes 1 to N are children
            real_edges = np.array(list(zip(children, parents)), dtype=np.int32).flatten()

        # Pad edges up to max_joints length
        num_real_edge_pairs = len(real_edges) // 2
        self.num_joint_pads = self.max_joints - num_real_edge_pairs

        # Pad with dummy edges (e.g., self-loops on the dummy padding node index)
        pad_value = self.max_limbs - 1 # Index of a potential dummy node beyond real nodes
        padded_edges = np.full(2 * self.max_joints, pad_value, dtype=np.int32)
        if len(real_edges) > 0:
             if len(real_edges) > len(padded_edges):
                 print(f"Warning: Robot {self.env.robot_name} has more edges ({num_real_edge_pairs}) than MAX_JOINTS ({self.max_joints}). Truncating edges.")
                 padded_edges[:] = real_edges[:2 * self.max_joints]
             else:
                 padded_edges[:len(real_edges)] = real_edges
        self.edges = padded_edges.astype(np.float32) # Convert to float for obs space compatibility

        # --- Action Padding Mask ---
        # Determines which outputs of the policy correspond to valid actions.
        # Simplified Plan: Assumes main decoder outputs only ARM actions.
        # Gripper action comes from a separate head.
        # Policy output size is assumed to be `max_limbs * actions_per_node` (e.g., max_limbs * 1 if 1 DoF per node).
        # Let's assume the decoder outputs 1 action per *arm* node (nodes 1 to num_arm_joints).
        # Total required arm actions = self.num_arm_joints.
        # Assumed policy ARM output dimension = self.max_limbs (needs coordination with model decoder).
        # *CRITICAL*: This mask MUST match the structure of the policy's output layer.
        # If the policy outputs `max_limbs` actions intended for nodes 1..max_limbs:
        is_valid_action = [False] # Action for base node (node 0) is invalid
        is_valid_action.extend([True] * self.num_arm_joints) # Actions for nodes 1 to N_arm are valid arm joints
        is_valid_action.extend([False] * (self.max_limbs - (self.num_arm_joints + 1))) # Pad remaining nodes
        # Convert to mask (True = padding/invalid)
        self.act_padding_mask = np.array(~np.array(is_valid_action), dtype=bool)

        # Sanity check
        if len(self.act_padding_mask) != self.max_limbs:
             raise ValueError(f"Action padding mask length ({len(self.act_padding_mask)}) doesn't match MAX_LIMBS ({self.max_limbs}). Check logic.")
        if (~self.act_padding_mask).sum() != self.num_arm_joints:
             print(f"Warning: Action mask expects {self.num_arm_joints} valid arm actions, but mask has {(~self.act_padding_mask).sum()}.")


    def _define_observation_space(self):
        """Defines the final, padded observation space dictionary."""
        inf = np.float32(np.inf)
        obs_shape_dict = OrderedDict()

        # Padded Node-Centric Observations
        obs_shape_dict['proprioceptive'] = Box(-inf, inf, (self.limb_obs_size * self.max_limbs,), np.float32)
        obs_shape_dict['context'] = Box(-inf, inf, (self.context_obs_size * self.max_limbs,), np.float32)

        # Padded Structural Information
        obs_shape_dict['edges'] = Box(-inf, inf, (2 * self.max_joints,), np.float32)

        # Padding Masks
        obs_shape_dict['obs_padding_mask'] = Box(0, 1, (self.max_limbs,), dtype=bool) # For transformer input
        # Action mask needs to match the assumed policy output structure
        # If policy outputs max_limbs actions (1 per potential node):
        obs_shape_dict['act_padding_mask'] = Box(0, 1, (self.max_limbs,), dtype=bool)

        # Extroceptive Information (passed through) - Get shapes from underlying env's *processed* obs
        # Need to access the space defined in RobosuiteEnvWrapper before modification
        base_obs_space = self.env.observation_space
        if 'object_state' in base_obs_space.spaces:
             obj_state_shape = base_obs_space['object_state'].shape
             # Ensure shape is valid (not zero size if key exists)
             if obj_state_shape[0] > 0:
                 obs_shape_dict['object_state'] = Box(-inf, inf, obj_state_shape, np.float32)
             else:
                 print("Warning: 'object_state' has size 0, omitting from final obs space.")

        if 'gripper_to_object' in base_obs_space.spaces:
             g2o_shape = base_obs_space['gripper_to_object'].shape
             if g2o_shape[0] > 0:
                 obs_shape_dict['gripper_to_object'] = Box(-inf, inf, g2o_shape, np.float32)
             else:
                 print("Warning: 'gripper_to_object' has size 0, omitting from final obs space.")

        # Add EEF state keys if they exist in the base observation
        if 'eef_pos' in base_obs_space.spaces:
             eef_pos_shape = base_obs_space['eef_pos'].shape
             obs_shape_dict['eef_pos'] = Box(-inf, inf, eef_pos_shape, np.float32)
        if 'eef_quat' in base_obs_space.spaces:
             eef_quat_shape = base_obs_space['eef_quat'].shape
             obs_shape_dict['eef_quat'] = Box(-inf, inf, eef_quat_shape, np.float32)
        # Add gripper state keys if they exist (will be used by extroceptive encoder/gripper head)
        if 'gripper_qpos' in base_obs_space.spaces:
             grip_qpos_shape = base_obs_space['gripper_qpos'].shape
             obs_shape_dict['gripper_qpos'] = Box(-inf, inf, grip_qpos_shape, np.float32)
        if 'gripper_qvel' in base_obs_space.spaces:
             grip_qvel_shape = base_obs_space['gripper_qvel'].shape
             obs_shape_dict['gripper_qvel'] = Box(-inf, inf, grip_qvel_shape, np.float32)

        return Dict(obs_shape_dict)


    def _extract_features_per_node(self, obs_dict):
        """
        Distributes features from the preliminary observation dictionary
        into per-node feature vectors according to the simplified plan.

        Args:
            obs_dict (dict): The observation dictionary from RobosuiteEnvWrapper.

        Returns:
            tuple: (node_proprio, node_context)
                node_proprio (np.ndarray): Shape (max_limbs, limb_obs_size=3)
                node_context (np.ndarray): Shape (max_limbs, context_obs_size=5)
        """
        node_proprio = np.zeros((self.max_limbs, self.limb_obs_size), dtype=np.float32)
        node_context = np.zeros((self.max_limbs, self.context_obs_size), dtype=np.float32)

        # --- Extract Features from Robosuite Obs Dict ---
        # These keys are expected based on RobosuiteEnvWrapper._convert_observation
        # Note: Using .get() with defaults for robustness
        joint_pos_cos = obs_dict.get(f'{self.env.robot_name}0_joint_pos_cos', np.zeros(self.num_arm_joints))
        joint_pos_sin = obs_dict.get(f'{self.env.robot_name}0_joint_pos_sin', np.zeros(self.num_arm_joints))
        joint_vel = obs_dict.get(f'{self.env.robot_name}0_joint_vel', np.zeros(self.num_arm_joints))

        # --- Extract Context (Static Properties) ---
        # This requires accessing the *underlying* robosuite env.sim.model
        sim = self.env.env.sim # Access base robosuite env's sim object
        robot_model = self.env.env.robots[0] # Access base robot object

        # Get joint indices reference by the robot model (excluding gripper)
        ref_joint_indexes = robot_model._ref_joint_indexes
        ref_joint_vel_indexes = robot_model._ref_joint_vel_indexes # Often  from 6 (after root)
        num_arm_joints_to_use = self.num_arm_joints # Assume metadata is correct for now

        if len(ref_joint_indexes) != self.num_arm_joints:
             print(f"Warning: Metadata arm joints ({self.num_arm_joints}) != reference joint indexes ({len(ref_joint_indexes)}). Context might be misaligned.")
             # TODO: Adjust num_arm_joints based on reference if mismatch detected? Risky.
             # num_arm_joints_to_use = min(self.num_arm_joints, len(ref_joint_indexes))
             num_arm_joints_to_use = self.num_arm_joints # Assume metadata is correct for now

        joint_limits = np.zeros((num_arm_joints_to_use, 2), dtype=np.float32)
        joint_damping = np.zeros(num_arm_joints_to_use, dtype=np.float32)
        joint_armature = np.zeros(num_arm_joints_to_use, dtype=np.float32)
        joint_friction = np.zeros(num_arm_joints_to_use, dtype=np.float32)

        for i in range(num_arm_joints_to_use):
            try:
                joint_id_model = ref_joint_indexes[i] # MuJoCo ID for the joint in the full model
                # Use the velocity index (usually starting from 6) for damping/armature/frictionloss
                # MuJoCo DoF arrays (damping, armature, frictionloss) are indexed by velocity coord (qvel)
                dof_id_model = ref_joint_vel_indexes[i]

                joint_limits[i] = sim.model.jnt_range[joint_id_model]
                joint_damping[i] = sim.model.dof_damping[dof_id_model]
                joint_armature[i] = sim.model.dof_armature[dof_id_model]
                joint_friction[i] = sim.model.dof_frictionloss[dof_id_model]
            except IndexError as e:
                 print(f"Error accessing MuJoCo model properties for joint index {i} (Model ID {joint_id_model}, DoF ID {dof_id_model}): {e}")
                 # Handle error, e.g., by keeping zeros or raising


        # --- Populate Node Features (Nodes 1 to N_arm) ---
        for i in range(num_arm_joints_to_use):
            node_idx = i + 1 # Node index (1-based for arm joints)
            if node_idx < self.max_limbs:
                # Proprioceptive: [sin(pos_i), cos(pos_i), vel_i]
                proprio_feat = [joint_pos_sin[i], joint_pos_cos[i], joint_vel[i]]
                node_proprio[node_idx, :3] = np.array(proprio_feat, dtype=np.float32)

                # Context: [limit_low, limit_high, damping, armature, frictionloss]
                context_feat = [joint_limits[i, 0], joint_limits[i, 1], joint_damping[i], joint_armature[i], joint_friction[i]]
                node_context[node_idx, :5] = np.array(context_feat, dtype=np.float32)

        # --- Populate Node 0 (Base) ---
        # Currently filled with zeros by initialization.
        # TODO: Add base pose/velocity if needed later.

        # --- Padding ---
        # Remaining entries in node_proprio and node_context are already zeros.

        return node_proprio, node_context


    def observation(self, obs):
        """
        Processes the observation dictionary from the underlying wrapper
        into the final format for the policy.
        """
        start_time = time.time()
        # Extract raw data passed from RobosuiteEnvWrapper
        # Use .get() to handle potentially missing keys gracefully
        object_state = obs.get('object_state', np.array([], dtype=np.float32))
        gripper_to_object = obs.get('gripper_to_object', np.array([], dtype=np.float32))
        eef_pos = obs.get('eef_pos', np.array([0.,0.,0.], dtype=np.float32))
        eef_quat = obs.get('eef_quat', np.array([1.,0.,0.,0.], dtype=np.float32))
        gripper_qpos = obs.get('gripper_qpos', np.zeros(self.num_gripper_joints, dtype=np.float32))
        gripper_qvel = obs.get('gripper_qvel', np.zeros(self.num_gripper_joints, dtype=np.float32))

        # Distribute proprioceptive and context features into padded, per-node arrays
        # This now relies on the more detailed extraction logic using sim.model
        node_proprio, node_context = self._extract_features_per_node(obs)

        # --- Assemble the Final Observation Dictionary ---
        final_obs = OrderedDict()
        final_obs['proprioceptive'] = node_proprio.flatten()
        final_obs['context'] = node_context.flatten()
        final_obs['edges'] = self.edges # Use pre-computed padded edges
        final_obs['obs_padding_mask'] = self.obs_padding_mask
        final_obs['act_padding_mask'] = self.act_padding_mask # Use the arm-only mask

        # --- Pass Through Extroceptive & Other Global Features ---
        # These will be processed by the ExtroceptiveEncoder in the model
        # Ensure keys match those defined in self._define_observation_space()
        if 'object_state' in self.observation_space.spaces:
             final_obs['object_state'] = object_state
        if 'gripper_to_object' in self.observation_space.spaces:
            final_obs['gripper_to_object'] = gripper_to_object
        if 'eef_pos' in self.observation_space.spaces:
            final_obs['eef_pos'] = eef_pos
        if 'eef_quat' in self.observation_space.spaces:
             final_obs['eef_quat'] = eef_quat
        # Include gripper state for the extroceptive encoder / separate gripper head
        if 'gripper_qpos' in self.observation_space.spaces:
             final_obs['gripper_qpos'] = gripper_qpos
        if 'gripper_qvel' in self.observation_space.spaces:
             final_obs['gripper_qvel'] = gripper_qvel


        # --- Verification (Optional Debugging) ---
        # for key, val in final_obs.items():
        #     if key in self.observation_space.spaces:
        #         if not self.observation_space[key].contains(val):
        #             print(f"Observation Verification ERROR for key '{key}':")
        #             print(f"  Space: {self.observation_space[key]}")
        #             print(f"  Value Shape: {val.shape}, Value Dtype: {val.dtype}")
        #             # print(f"  Value: {val}") # Can be large
        #             # raise ValueError(f"Observation value for {key} is outside defined space!")
        #     else:
        #          print(f"Warning: Observation key '{key}' generated but not in defined final observation space.")


        end_time = time.time()
        # print(f"Obs wrapper time: {end_time - start_time:.6f}s")
        return final_obs


    def reset(self, **kwargs):
        """Resets the environment and processes the initial observation."""
        observation = self.env.reset(**kwargs)
        # Regenerate masks/edges in case robot metadata changed
        # (unlikely for fixed robots but good practice)
        # Ensure metadata is fetched from the *correct* underlying env instance
        current_metadata = self.env.metadata.get('robot_metadata', {})
        if not current_metadata:
             print("Warning: Robot metadata missing during reset in NodeCentricObservation wrapper.")
        else:
            self.robot_metadata = current_metadata
            self.num_arm_joints = self.robot_metadata.get('num_arm_joints', 0)
            self.num_nodes = self.num_arm_joints + 1
            self.num_gripper_joints = self.robot_metadata.get('num_gripper_joints', 0)
            self.num_node_pads = self.max_limbs - self.num_nodes # Recalculate padding
            # Regenerate masks and edges based on potentially updated metadata
            self._create_padding_masks_and_edges()
            self.metadata['act_padding_mask'] = self.act_padding_mask # Update metadata for action wrapper
            self.metadata['num_nodes'] = self.num_nodes

        return self.observation(observation)


class RobosuiteNodeCentricAction(gym.ActionWrapper):
    """
    Wrapper to convert the padded action vector from the policy's main decoder
    (assumed to be arm actions) into the correct format for the underlying
    Robosuite environment. Gripper actions are assumed to be handled separately.
    """
    def __init__(self, env):
        super().__init__(env)

        # Get the original action space of the *base* Robosuite environment
        unwrapped_env = env.unwrapped
        while hasattr(unwrapped_env, 'env'): # Iterate down
            if isinstance(unwrapped_env, RobosuiteEnvWrapper):
                break
            unwrapped_env = unwrapped_env.env
        if not isinstance(unwrapped_env, RobosuiteEnvWrapper):
            raise TypeError("Could not find RobosuiteEnvWrapper in the wrapper stack for ActionWrapper.")

        self.base_action_space = unwrapped_env.action_space
        # Determine dimensions for arm and gripper actions from the base space
        # This assumes the base action space concatenates arm and gripper actions.
        self.num_arm_actions = unwrapped_env.metadata['robot_metadata'].get('num_arm_joints', 0)
        self.num_gripper_actions = unwrapped_env.metadata['robot_metadata'].get('num_gripper_joints', 0) # Assumes 1 DoF per gripper joint? Check robosuite spec.
        self.real_action_dim = self.base_action_space.shape[0]

        # Verify consistency
        if self.num_arm_actions + self.num_gripper_actions != self.real_action_dim:
             print(f"Warning: Sum of arm ({self.num_arm_actions}) and gripper ({self.num_gripper_actions}) joints doesn't match base action dim ({self.real_action_dim}). Check metadata extraction.")
             # Fallback: assume all actions are arm actions if gripper count is wrong
             if self.num_arm_actions != self.real_action_dim and self.num_gripper_actions == 0:
                  print(f"  Adjusting num_arm_actions to {self.real_action_dim}")
                  self.num_arm_actions = self.real_action_dim


        # Get max dims from config
        self.max_limbs = cfg.MODEL.MAX_LIMBS

        # --- Define the PADDED Action Space the Policy Outputs (Main Decoder Arm Actions) ---
        # Policy's main decoder outputs one action per node (up to max_limbs)
        # Based on our obs wrapper, only actions corresponding to nodes 1..N_arm are valid.
        self.padded_action_dim = self.max_limbs # Policy outputs max_limbs potential arm actions
        low = -1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        high = 1.0 * np.ones(self.padded_action_dim, dtype=np.float32)
        self.action_space = Box(low=low, high=high, dtype=np.float32)

        # Get the action padding mask generated by the observation wrapper
        if 'act_padding_mask' not in self.env.metadata:
            raise ValueError("Action padding mask not found in env metadata for ActionWrapper.")
        self.act_padding_mask = self.env.metadata['act_padding_mask']

        # Verify mask length matches the padded action space dimension
        if len(self.act_padding_mask) != self.padded_action_dim:
            raise ValueError(f"Action padding mask length ({len(self.act_padding_mask)}) in metadata does not match expected padded action dimension ({self.padded_action_dim}).")


    def action(self, action):
        """
        Takes the padded action vector (arm actions) from the policy's main decoder,
        un-pads it, and combines it with a placeholder/default gripper action
        to match the underlying Robosuite environment's expected format.
        """
        # action is the raw, padded output from the policy's ARM decoder (shape: self.padded_action_dim)
        if action.shape[-1] != self.padded_action_dim:
             raise ValueError(f"Received action shape {action.shape} doesn't match expected padded dimension {self.padded_action_dim}")

        # Select the valid ARM action dimensions using the mask
        unpadded_arm_action = action[~self.act_padding_mask]

        # Ensure the resulting shape matches the number of arm joints
        if unpadded_arm_action.shape[0] != self.num_arm_actions:
            num_valid_mask = (~self.act_padding_mask).sum()
            raise ValueError(
                f"Arm action shape mismatch after unpadding: "
                f"Expected {self.num_arm_actions} (from metadata), "
                f"but got {unpadded_arm_action.shape[0]} (based on mask with {num_valid_mask} valid entries). "
                f"Check mask generation in observation wrapper."
            )

        # --- Gripper Action ---
        # Gripper action is handled separately (e.g., by another policy head).
        # Here, we need to provide *some* value for the gripper dimensions
        # expected by the underlying environment.
        # Common choice: Action '1' usually means 'close gripper'. Let's use that as default.
        # Check Robosuite controller spec for exact meaning (-1 to 1).
        default_gripper_action = np.ones(self.num_gripper_actions, dtype=np.float32) # Default: Close

        # Combine arm and gripper actions
        full_action = np.concatenate([unpadded_arm_action, default_gripper_action])

        # Ensure the final action matches the base environment's dimension
        if full_action.shape[0] != self.real_action_dim:
             raise ValueError(f"Final combined action dimension ({full_action.shape[0]}) does not match base environment ({self.real_action_dim}).")


        # Clip the final combined action to the valid range of the base environment
        clipped_action = np.clip(full_action, self.base_action_space.low, self.base_action_space.high)

        return clipped_action