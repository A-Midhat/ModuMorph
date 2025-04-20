import os
from metamorph.config import cfg

from metamorph.envs.tasks.robosuite_task import make_env_robosuite

from metamorph.envs.modules.agent import create_agent_xml
from metamorph.envs.tasks.escape_bowl import make_env_escape_bowl
from metamorph.envs.tasks.locomotion import make_env_locomotion
from metamorph.envs.tasks.obstacle import make_env_obstacle
from metamorph.envs.tasks.incline import make_env_incline
from metamorph.envs.tasks.exploration import make_env_exploration
from metamorph.envs.wrappers.select_keys import SelectKeysWrapper
from metamorph.utils import file as fu


def make_env(agent_name):
    xml_path = os.path.join(
        cfg.ENV.WALKER_DIR, "xml", "{}.xml".format(agent_name)
    )

    if cfg.ENV_NAME =="Robosuite-v0":
        # agent name here is teh robot type (i.e. "Sawyer")
        env = make_env_robosuite(robot_name=agent_name)
    elif cfg.ENV_NAME == "Unimal-v0" or cfg.ENV_NAME == "Modular-v0":
        xml = create_agent_xml(xml_path)
        env_func = "make_env_{}".format(cfg.ENV.TASK)
        env = globals()[env_func](xml, agent_name)

    else :
        raise NotImplementedError
    
    if cfg.ENV_NAME == "Robosuite-v0":
        # for single robots 
        if cfg.MODEL.TYPE=="mlp":
            keys_to_keep = ["proprioceptive"] # try add object state
        else:
            # TODO: add this as sperate keeys to keep for robosuite' or similar
            keys_to_keep = ["proprioceptive", "context", "edges", \
                            "obs_padding_mask", "act_padding_mask", \
                                
                                "object_state", "gripper_to_object",\
                                "eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel"]
            final_obs_spcae_keys = env.observation_space.spaces.keys()
            keys_to_keep = [k for k in keys_to_keep if k in final_obs_spcae_keys]
    else :
        keys_to_keep = cfg.ENV.KEYS_TO_KEEP + cfg.MODEL.OBS_TYPES
    env = SelectKeysWrapper(env, keys_to_keep=keys_to_keep)
    return env
