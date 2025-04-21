from metamorph.config import cfg
from metamorph.envs.wrappers.robosuite_wrappers import RobosuiteEnvWrapper, RobosuiteNodeCentricObservation, RobosuiteNodeCentricAction        
# --- Factory function for making env and applying the wrappers --- 
#TODO: MOVE IT OTHER FILE TO ESCAPE THE CIRCULAR IMPORT ERROR
def make_env_robosuite(robot_name):
    base_env = RobosuiteEnvWrapper(
        robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
        robot_name=robot_name,
        controller_name=cfg.ROBOSUITE.CONTROLLER,
        robosuite_cfg=cfg.ROBOSUITE
    )
    # TODO: Uncomment latter 
    # for wrapper_name in cfg.ENV.WRAPPERS:
    #     env = globals()[wrapper_name](env) 
    env = RobosuiteNodeCentricObservation(base_env)
    env = RobosuiteNodeCentricAction(env)
    return env
env = make_env_robosuite("Panda")
print(env.observation_space)