from metamorph.config import cfg
from metamorph.envs.wrappers.robosuite_wrappers import ( \
    RobosuiteEnvWrapper, RobosuiteNodeCentricObservation, \
    RobosuiteNodeCentricAction, RobosuiteMLPFlattener )       
# --- Factory function for making env and applying the wrappers --- 
#TODO: MOVE IT OTHER FILE TO ESCAPE THE CIRCULAR IMPORT ERROR
cfg.MODEL.TYPE = "mlp"
def make_env_robosuite(robot_name):
    base_env = RobosuiteEnvWrapper(
        robosuite_env_name=cfg.ROBOSUITE.ENV_NAME,
        robot_name=robot_name,
        controller_name=cfg.ROBOSUITE.CONTROLLER,
        robosuite_cfg=cfg.ROBOSUITE
    )
    if cfg.MODEL.TYPE == "mlp":
        print("Applying [RobosuiteMLPFlattener] wrapper")
        env = RobosuiteMLPFlattener(base_env)
        return env
    elif cfg.MODEL.TYPE == "transformer":
        print("Applying [RobosuiteNodeCentricObservation/Action] wrappers")
        env = RobosuiteNodeCentricObservation(base_env)
        env = RobosuiteNodeCentricAction(env)
        return env
    else:
        print("Warning: No wrapper applied. Using base environment.")
        return base_env
    # TODO: Uncomment latter 
    # for wrapper_name in cfg.ENV.WRAPPERS:
    #     env = globals()[wrapper_name](env) 
    # env = RobosuiteNodeCentricObservation(base_env)
    # env = RobosuiteNodeCentricAction(env)
    # return env
env = make_env_robosuite("Panda")
print(env.observation_space)