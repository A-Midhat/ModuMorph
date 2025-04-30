from metamorph.config import cfg
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper

def make_env_robosuite(robot_name): # TODO: for future add task name as will  (MTRL)
    base_env = RobosuiteEnvWrapper(
        robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
        robot_name=robot_name,
        controller_name=cfg.ROBOSUITE.CONTROLLER,
        robosuite_cfg=cfg.ROBOSUITE
    )
    return base_env
