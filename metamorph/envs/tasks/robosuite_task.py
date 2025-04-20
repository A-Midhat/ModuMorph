# TODO:CHeck the obs and act spaces here 

import gym 
from gym import spaces
import numpy as np 
import robosuite
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS 

from metamorph.config import cfg



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
        
# --- Factory function for making env and applying the wrappers --- 
#TODO: MOVE IT OTHER FILE TO ESCAPE THE CIRCULAR IMPORT ERROR
def make_env_robosuite(robot_name):
    base_env = RobosuiteEnvWrapper(
        robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
        robot_name=robot_name,
        controller_name=cfg.ROBOSUITE.CONTROLLER,
        robosuite_cfg=cfg.ROBOSUITE # pass the config to the env 
    ) 
    try:
        from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteNodeCentricObservation, RobosuiteNodeCentricAction
        print("Success")
    except Exception as e:
        print("ERROR: ", e)
        print("Warning: metamorph.envs.wrappers.robosuite_wrappers not found yet. Node-centric wrappers will not be applied by default.")
        # Will cause problems 
        RobosuiteNodeCentricObservation = None
        RobosuiteNodeCentricAction = None
    
    if RobosuiteNodeCentricObservation is not None and RobosuiteNodeCentricAction is not None:
        env = RobosuiteNodeCentricObservation(base_env)
        env = RobosuiteNodeCentricAction(env)
        return env
    else:
        print("Warning: RobosuiteNodeCentricObservation or RobosuiteNodeCentricAction is not defined. Skipping the wrappers.")
        env = base_env 
    # TODO: Uncomment latter 
    # for wrapper_name in cfg.ENV.WRAPPERS:
    #     env = globals()[wrapper_name](env) 
    
    return env
make_env_robosuite("Panda")