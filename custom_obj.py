import numpy as np
import robosuite as suite
from collections import OrderedDict
import os 
# Imports needed for our custom class
from robosuite.models.tasks import ManipulationTask
from robosuite.environments.manipulation.lift import Lift
from robosuite.models.objects import BallObject
from robosuite.models.arenas import TableArena
from robosuite.utils.placement_samplers import UniformRandomSampler
# New imports needed for the overridden observables method
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.transform_utils import convert_quat


# --- VIDEO RECORDING SOLUTIONS ---

class VideoRecorder:
    """Simple video recorder for robosuite environments"""
    
    def __init__(self, save_path="./videos", filename="recording", fps=20):
        self.save_path = save_path
        self.filename = filename
        self.fps = fps
        self.frames = []
        os.makedirs(save_path, exist_ok=True)
    
    def capture_frame(self, env, camera_name="agentview"):
        """Capture a frame from the environment"""
        frame = env.sim.render(width=512, height=512, camera_name=camera_name)
        if frame is not None:
            # MuJoCo renders frames upside down, so flip them
            frame = np.flipud(frame)
            self.frames.append(frame)
    
    def save_video(self):
        """Save collected frames as video"""
        if not self.frames:
            print("No frames to save!")
            return False
        
        # Try imageio first (best quality)
        if self._save_with_imageio():
            return True
        
        # Fallback to PIL for GIF
        if self._save_with_pil():
            return True
            
        # Fallback to OpenCV
        if self._save_with_opencv():
            return True
            
        print("Failed to save video with any method")
        return False
    
    def _save_with_imageio(self):
        """Save as MP4 using imageio"""
        try:
            import imageio
            filepath = os.path.join(self.save_path, f"{self.filename}.mp4")
            writer = imageio.get_writer(filepath, fps=self.fps)
            for frame in self.frames:
                writer.append_data(frame)
            writer.close()
            print(f"✅ Video saved as {filepath}")
            return True
        except ImportError:
            print("⚠️  imageio not available")
            return False
        except Exception as e:
            print(f"❌ imageio failed: {e}")
            return False
    
    def _save_with_pil(self):
        """Save as GIF using PIL"""
        try:
            from PIL import Image
            filepath = os.path.join(self.save_path, f"{self.filename}.gif")
            images = [Image.fromarray(frame) for frame in self.frames]
            images[0].save(
                filepath,
                save_all=True,
                append_images=images[1:],
                duration=int(1000/self.fps),
                loop=0
            )
            print(f"✅ Video saved as {filepath}")
            return True
        except ImportError:
            print("⚠️  PIL not available")
            return False
        except Exception as e:
            print(f"❌ PIL failed: {e}")
            return False
    
    def _save_with_opencv(self):
        """Save as AVI using OpenCV"""
        try:
            import cv2
            filepath = os.path.join(self.save_path, f"{self.filename}.avi")
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(filepath, fourcc, self.fps, (width, height))
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print(f"✅ Video saved as {filepath}")
            return True
        except ImportError:
            print("⚠️  OpenCV not available")
            return False
        except Exception as e:
            print(f"❌ OpenCV failed: {e}")
            return False



# --- 1. The definitive custom class with all overrides ---
class LiftCustomObject(Lift):
    """
    A custom Lift environment that replaces the default cube with a ball
    by taking full control of model creation, references, and observables.
    """
    def _load_model(self):
        # Call grandparent's _load_model to set up the robot, skipping Lift's method
        super(Lift, self)._load_model()

        # Adjust base pose
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # Load workspace
        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        # Create our custom ball object
        self.ball = BallObject(
            name="ball",
            size=[0.04],
            rgba=[0, 0, 1, 1],
        )

        # Create the placement initializer for OUR object
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler", mujoco_objects=self.ball,
            x_range=[-0.03, 0.03], y_range=[-0.03, 0.03],
            rotation=None, ensure_object_boundary_in_range=False,
            ensure_valid_placement=True, reference_pos=self.table_offset,
            z_offset=0.01, #rng=self.rng,
        )

        # Create the task model with the arena, robot, and OUR ball
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=self.ball,
        )

    def _setup_references(self):
        # Call grandparent's _setup_references, skipping Lift's method
        super(Lift, self)._setup_references()
        # Create a reference ONLY to our new object
        self.ball_body_id = self.sim.model.body_name2id(self.ball.root_body)

    # THIS IS THE NEW METHOD THAT FIXES THE CRASH
    def _setup_observables(self):
        # Get robot observables from the grandparent class
        observables = super(Lift, self)._setup_observables()

        # Get robot-specific prefixes
        pf = self.robots[0].robot_model.naming_prefix

        # Add object-related observables
        if self.use_object_obs:
            modality = "object"
            
            # Ball position sensor
            @sensor(modality=modality)
            def ball_pos(obs_cache):
                return np.array(self.sim.data.body_xpos[self.ball_body_id])

            # Ball quaternion sensor
            @sensor(modality=modality)
            def ball_quat(obs_cache):
                return convert_quat(np.array(self.sim.data.body_xquat[self.ball_body_id]), to="xyzw")
            
            # Gripper-to-ball position sensor
            @sensor(modality=modality)
            def gripper_to_ball_pos(obs_cache):
                return (
                    obs_cache[f"{pf}eef_pos"] - obs_cache["ball_pos"]
                    if f"{pf}eef_pos" in obs_cache and "ball_pos" in obs_cache
                    else np.zeros(3)
                )

            sensors = [ball_pos, ball_quat, gripper_to_ball_pos]
            names = [s.__name__ for s in sensors]

            for name, s in zip(names, sensors):
                observables[name] = Observable(
                    name=name,
                    sensor=s,
                    sampling_rate=self.control_freq,
                )
        return observables

    def _check_success(self):
        # Override to check the height of our ball
        ball_height = self.sim.data.body_xpos[self.ball_body_id][2]
        table_height = self.model.mujoco_arena.table_offset[2]
        return ball_height > table_height + 0.04

    def reward(self, action=None):
        """
        Reward function for the task.

        Sparse un-normalized reward:

            - a discrete reward of 2.25 is provided if the cube is lifted

        Un-normalized summed components if using reward shaping:

            - Reaching: in [0, 1], to encourage the arm to reach the cube
            - Grasping: in {0, 0.25}, non-zero if arm is grasping the cube
            - Lifting: in {0, 1}, non-zero if arm has lifted the cube

        The sparse reward only consists of the lifting component.

        Note that the final reward is normalized and scaled by
        reward_scale / 2.25 as well so that the max score is equal to reward_scale

        Args:
            action (np array): [NOT USED]

        Returns:
            float: reward value
        """
        reward = 0.0

        # sparse completion reward
        if self._check_success():
            reward = 2.25

        # use a shaping reward
        elif self.reward_shaping:

            # reaching reward
            dist = self._gripper_to_target(
                gripper=self.robots[0].gripper, target=self.ball.root_body, target_type="body", return_distance=True
            )
            reaching_reward = 1 - np.tanh(10.0 * dist)
            reward += reaching_reward

            # grasping reward
            if self._check_grasp(gripper=self.robots[0].gripper, object_geoms=self.ball):
                reward += 0.25

        # Scale reward if requested
        if self.reward_scale is not None:
            reward *= self.reward_scale / 2.25

        return reward




def run_with_video_recording():
    """Run simulation with video recording"""
    
    # Create environment - MUST have offscreen rendering enabled
    env = LiftCustomObject(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,  # This is crucial!
        use_camera_obs=False,  # We'll use sim.render directly
        camera_names="agentview",
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
        horizon=500,
        reward_scale=1.0,
        reward_shaping=True,
    )
    
    # Initialize video recorder
    recorder = VideoRecorder(save_path="./videos", filename="custom_ball_lift")
    
    print("🎬 Starting simulation with video recording...")
    obs = env.reset()
    total_reward = 0
    step_rewards = []
    
    print("Initial observation keys:", obs.keys() if isinstance(obs, dict) else "N/A")
    
    # Run simulation
    for i in range(500):
        # Simple policy: move toward ball and grasp
        if isinstance(obs, dict) and 'gripper_to_ball_pos' in obs:
            gripper_to_ball = obs['gripper_to_ball_pos']
            action = np.zeros(env.action_dim)
            
            # Move toward ball
            action[:3] = np.clip(gripper_to_ball * 3.0, -1, 1)
            
            # Close gripper when close
            if np.linalg.norm(gripper_to_ball) < 0.06:
                action[-1] = 1.0  # Close gripper
            else:
                action[-1] = -1.0  # Open gripper
        else:
            action = np.random.rand(env.action_dim) * 2 - 1
        
        obs, reward, terminated, info = env.step(action)
        total_reward += reward
        step_rewards.append(reward)
        
        # Capture frame every few steps to reduce file size
        if i % 3 == 0:
            recorder.capture_frame(env, camera_name="agentview")
        
        if terminated:
            print(f"Episode terminated at step {i}")
            break
    
    print(f"Episode finished. Total reward: {total_reward:.3f}")
    print(f"Average reward per step: {np.mean(step_rewards):.4f}")
    print(f"Collected {len(recorder.frames)} frames")
    
    # Save video
    recorder.save_video()
    
    env.close()
    print("🎬 Simulation complete!")


def alternative_camera_recording():
    """Alternative method using camera observations"""
    
    env = LiftCustomObject(
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,  # Enable camera observations
        camera_names=["frontview"],  # Multiple cameras
        camera_heights=512,
        camera_widths=512,
        control_freq=20,
        horizon=500,
        reward_scale=1.0,
        reward_shaping=True,
    )
    
    print("📹 Starting alternative camera recording method...")
    obs = env.reset()
    
    frames = []
    total_reward = 0
    
    for i in range(200):  # Shorter episode for demo
        action = np.random.rand(env.action_dim) * 2 - 1
        obs, reward, terminated, info = env.step(action)
        total_reward += reward
        
        # Collect frames from camera observations
        if 'agentview_image' in obs:
            frame = obs['agentview_image']
            frames.append(frame)
        
        if terminated:
            break
    
    print(f"Total reward: {total_reward:.3f}, Frames: {len(frames)}")
    
    # Save frames
    if frames:
        try:
            import imageio
            imageio.mimsave('./videos/camera_obs_recording.mp4', frames, fps=20)
            print("✅ Camera observation video saved!")
        except ImportError:
            print("❌ imageio required for camera observation method")
    
    env.close()


if __name__ == "__main__":
    print("Choose recording method:")
    print("1. Direct rendering method (recommended)")
    print("2. Camera observation method")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == "1":
        run_with_video_recording()
    elif choice == "2":
        alternative_camera_recording()
    else:
        print("Running default method...")
        run_with_video_recording()








# if __name__ == "__main__":
#     env = LiftCustomObject(
#         robots="Panda",
#         has_renderer=False,
#         has_offscreen_renderer=True,
#         use_camera_obs=False,
#         camera_names="agentview",
#         camera_heights=512,
#         camera_widths=512,
#         control_freq=20,
#         horizon=500,
#         reward_scale=1.0,
#         reward_shaping=True,
#     )
#     print("Starting simulation... this should work now! ✅")
#     obs = env.reset()
#     r = 0
#     print("Initial observation keys:", obs.keys() if isinstance(obs, dict) else "N/A")
#     for i in range(500):
#         action = np.random.rand(env.action_dim)
#         obs, reward, terminated, info = env.step(action)
#         r += reward
#         if terminated:
#             obs= env.reset()
#     print("Episode finished. Total reward:", r)
#     env.close()