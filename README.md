# Investigating Morphology-Aware Policies for Robotic Manipulation

This repository contains the code for the Master's thesis, "Investigating Morphology-Aware Policies for Robotic Manipulation." This work adapts and extends the state-of-the-art universal control framework, originally developed for locomotion tasks in papers like [MetaMorph](https://arxiv.org/abs/2203.11931) and [ModuMorph](https://arxiv.org/abs/2302.11070), to the more precise and challenging domain of robotic manipulation.

Our core contribution is the integration and evaluation of these powerful, Transformer-based, morphology-aware policies within the **Robosuite** simulation environment. We demonstrate their effectiveness in learning complex manipulation tasks, such as lifting objects and opening doors, across a diverse set of commercially available robot arms.

## Key Features of This Fork

*   **Robosuite Integration:** The entire codebase has been adapted to work seamlessly with the Robosuite environment, including custom wrappers for observation and action spaces.
*   **Semantic Graph Representation for Manipulators:** We introduce a node-centric graph representation specifically for manipulator arms, consisting of `Base`, `Arm Link`, `Hand`, and `Gripper` nodes.
*   **JNT and OSC Control Support:** The framework supports training policies with both low-level `JOINT_VELOCITY` control and high-level `OSC_POSE` (Operational Space Control).
*   **Multi-Robot Training for Manipulation:** The training pipeline is configured for multi-robot, single-task (MR-ST) learning in a manipulation context, enabling a single policy to learn to control multiple different robot arms simultaneously.


## Installation

Run the following script:

```bash
./docker/build_files/requirements.txt
```

This will set up the environment with all necessary dependencies, including Robosuite and the required Python packages.

## Running the Experiments

All experiment configurations are located in the `configs/` directory. The main training script is `tools/train_ppo.py`.

### 1. Training a Universal Policy (Multi-Robot, Single-Task)

To train a universal policy on a set of robot arms for a specific manipulation task, you can use the provided configuration files. The following examples demonstrate how to train the **ModuMorph** and **MetaMorph** architectures.

**Example: Train ModuMorph with OSC on the 'Lift' Task**

This command trains the primary model from the thesis: ModuMorph with `OSC_POSE` control on six different robot arms for the Lift task.

```bash
python tools/train_ppo.py \
    --cfg configs/robosuite_modumorph.yaml \
    OUT_DIR ./output/modumorph_osc_lift/ \
    ROBOSUITE.ENV_NAMES '["Lift","Lift","Lift","Lift","Lift","Lift"]' \
    ROBOSUITE.CONTROLLERS '["OSC_POSE","OSC_POSE","OSC_POSE","OSC_POSE","OSC_POSE","OSC_POSE"]' \
    MODEL.DECODER_OUT_DIM 6
```

**Example: Train MetaMorph with JNT on the 'Door' Task**

This command trains the MetaMorph baseline with `JOINT_VELOCITY` control on the Door task.

```bash
python tools/train_ppo.py \
    --cfg configs/robosuite_metamorph.yaml \
    OUT_DIR ./output/metamorph_jnt_door/ \
    ROBOSUITE.ENV_NAMES '["Door","Door","Door","Door","Door","Door"]' \
    ROBOSUITE.CONTROLLERS '["JOINT_VELOCITY","JOINT_VELOCITY","JOINT_VELOCITY","JOINT_VELOCITY","JOINT_VELOCITY","JOINT_VELOCITY"]' \
    MODEL.DECODER_OUT_DIM 1
```

### 2. Training a Single-Robot Baseline

To train a specialized policy on a single robot (e.g., for the SR-10M or SR-fair baselines), you can use the single-robot configuration files.

**Example: Train a Panda on the 'Lift' task with OSC**

```bash
python tools/train_ppo.py \
    --cfg configs/robosuite_baseline/panda.yaml \
    OUT_DIR ./output/baselines/panda_lift_osc/ \
    ROBOSUITE.CONTROLLERS '["OSC_POSE"]' \
    MODEL.DECODER_OUT_DIM 6
```
*Note: The `panda.yaml` is pre-configured for JNT control. We override the controller and `DECODER_OUT_DIM` for OSC.*

## Acknowledgements

This codebase is a fork and significant adaptation of the original [MetaMorph](https://github.com/agrimgupta92/metamorph) and [ModuMorph](https://github.com/MasterXiong/ModuMorph) repositories. We are deeply grateful to the original authors for making their excellent work public.
