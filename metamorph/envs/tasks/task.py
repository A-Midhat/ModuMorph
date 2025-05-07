import os
from metamorph.config import cfg, get_list_cfg


# from metamorph.envs.modules.agent import create_agent_xml
# from metamorph.envs.tasks.escape_bowl import make_env_escape_bowl
# from metamorph.envs.tasks.locomotion import make_env_locomotion
# from metamorph.envs.tasks.obstacle import make_env_obstacle
# from metamorph.envs.tasks.incline import make_env_incline
# from metamorph.envs.tasks.exploration import make_env_exploration
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper
from metamorph.utils import file as fu

from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper


def make_base_env(env_id, morphology_spec, env_name_spec=None, controller_spec=None):
    """
    Creates the base environment instance based on env_id and specifications.

    Args:
        env_id (str): The main environment ID (e.g., "Robosuite-v0", "Unimal-v0").
        morphology_spec (str or list[str]):
            - For Robosuite: List/string of robot names (e.g., "Panda", ["Panda", "Panda"]).
            - For Unimal/Modular: The agent/walker name (string).
        env_name_spec (str, optional): Robosuite task name (e.g., "Lift"). Required if env_id="Robosuite-v0".
        controller_spec (str or list[str], optional): Robosuite controller(s). Required if env_id="Robosuite-v0".

    Returns:
        gym.Env: The created base environment instance (before common wrappers like VecNormalize).
    """
    env = None 
    if env_id =="Robosuite-v0":
        if env_name_spec is None or controller_spec is None: 
            raise ValueError("env_name_spec and controller_spec must be provided for Robosuite-v0.")
        env = RobosuiteEnvWrapper(
            env_name = env_name_spec, 
            robot_names = get_list_cfg(morphology_spec),
            controller_names= get_list_cfg(controller_spec),
            horizon=cfg.ROBOSUITE.ENV_ARGS.get('horizon', 100),
            robosuite_args=dict(cfg.ROBOSUITE.ENV_ARGS)

        )
    elif env_id in ["Unimal-v0","Modular-v0"]:
        # for those two morphology_spec is the agent_name
        agent_name = morphology_spec
        xml_path = os.path.join(
        cfg.ENV.WALKER_DIR, "xml", "{}.xml".format(agent_name)
    )
        if not os.path.exists(xml_path):
            raise FileNotFoundError("No agent xml file found at {}".format(xml_path))
        
        xml = create_agent_xml(xml_path)
        env_func = "make_env_{}".format(cfg.ENV.TASK)
        env = globals()[env_func](xml, agent_name)

    else :
        raise NotImplementedError
    
    if cfg.ENV_NAME != "Robosuite-v0":
        keys_to_keep = cfg.ENV.KEYS_TO_KEEP + cfg.MODEL.OBS_TYPES
        env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
    
    return env
