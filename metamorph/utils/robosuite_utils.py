import numpy as np
import robosuite
from robosuite.controllers import load_controller_config
from metamorph.config import cfg, get_list_cfg

# from metamorph.utils import mjpy as ju 

def count_limbs_and_joints(env):
    """
    Given a RobosuiteEnvWrapper.env (the robosuite.Env instance),
    return (num_nodes_total, num_edges_total) across all robots in that env.
    Nodes = base + one per link + one conceptual gripper.
    Edges = nodes - 1 per robot.
    """
    limbs = 0
    joints = 0
    for r in env.robots:
        # build node list for this robot
        nodes = []
        root = r.robot_model.root_body
        nodes.append(root)
        seen = {root}
        for jn in r.robot_joints:
            joint_id = env.sim.model.joint_name2id(jn)
            body_id = env.sim.model.jnt_bodyid[joint_id]
            body_name = env.sim.model.body_id2name(body_id)
            if body_name not in seen:
                nodes.append(body_name)
                seen.add(body_name)
        # add one conceptual gripper node
        nodes.append(nodes[-1] + "_gripper_conceptual")

        n_nodes = len(nodes)
        n_edges = n_nodes - 1

        limbs += n_nodes
        joints += n_edges

    return limbs, joints

def max_limbs_joints(morphs, env_names, controllers, common_args):
    """
    Instantiate each (morph,env,controller) combo, count limbs/joints,
    and return the maxima over all.
    """
    max_limbs, max_joints = 0, 0
    for i, m in enumerate(morphs):
        robots = get_list_cfg(m)
        ctrl = controllers[i]
        # pick first controller name if list
        ctrl_name = ctrl if isinstance(ctrl, str) else ctrl[0]
        cfg_ctrl = load_controller_config(default_controller=ctrl_name)
        # build environment
        env = robosuite.make(
            env_name=env_names[i],
            robots=robots,
            controller_configs=[cfg_ctrl]*len(robots),
            **common_args
        )
        l, j = count_limbs_and_joints(env)
        max_limbs = max(max_limbs, l)
        max_joints = max(max_joints, j)
        env.close()

    return max_limbs, max_joints



###########################################################
###################WRAPPERS UTILS##########################


# TODO: add robosuite NodeCentric wrappers functionality here


###########################################################
###################WRAPPERS UTILS##########################