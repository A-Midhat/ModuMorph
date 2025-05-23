import os
import time
import pickle
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from metamorph.config import cfg
from metamorph.envs.vec_env.vec_video_recorder import VecVideoRecorder
from metamorph.utils import file as fu
from metamorph.utils import model as mu
from metamorph.utils import optimizer as ou
from metamorph.utils.meter import TrainMeter

import wandb

from .buffer import Buffer
from .envs import get_ob_rms
from .envs import make_vec_envs
from .envs import set_ob_rms
from .inherit_weight import restore_from_checkpoint
from .model import ActorCritic
from .model import Agent

class PPO:
    def __init__(self, print_model=True):
        # Create vectorized envs
        self.envs = make_vec_envs()
        self.file_prefix = cfg.ENV_NAME

        self.device = torch.device(cfg.DEVICE)

        self.actor_critic = globals()[cfg.MODEL.ACTOR_CRITIC](
            self.envs.observation_space, self.envs.action_space
        )

        # Used while using train_ppo.py
        if cfg.PPO.CHECKPOINT_PATH:
            ob_rms = restore_from_checkpoint(self.actor_critic)
            set_ob_rms(self.envs, ob_rms)

        if print_model:
            print("Num params: {}".format(mu.num_params(self.actor_critic)))

        self.actor_critic.to(self.device)
        self.agent = Agent(self.actor_critic)

        # Setup experience buffer
        self.buffer = Buffer(self.envs.observation_space, self.envs.action_space.shape)
        # Optimizer for both actor and critic
        self.optimizer = optim.Adam(
            self.actor_critic.parameters(), lr=cfg.PPO.BASE_LR, eps=cfg.PPO.EPS, weight_decay=cfg.PPO.WEIGHT_DECAY
        )
        self.lr_scale = [1. for _ in self.optimizer.param_groups]

        self.train_meter = TrainMeter()

        # ----------------------
        # Setup logging backend:
        # Use wandb if cfg.LOGGING.USE_WANDB is True,
        # otherwise fallback to TensorBoard.
        # ----------------------
        if hasattr(cfg, "LOGGING") and cfg.LOGGING.get("USE_WANDB", False):
            self.logger_backend = "wandb"
            self.logger = wandb.init(
                entity=cfg.LOGGING.get("ENTITY", "default_entity"),
                project=cfg.LOGGING.get("PROJECT", "default_project"),
                name=cfg.OUT_DIR.split("/")[-1],
                config=dict(cfg)
            )
            wandb.config.update({"seed": cfg.RNG_SEED})
        else:
            self.logger_backend = "tensorboard"
            #from torch.utils import tensorboard
            from torch.utils.tensorboard import SummaryWriter
            self.logger = SummaryWriter(log_dir=os.path.join(cfg.OUT_DIR, "tensorboard"))
        # ----------------------

        # Get the param name for log_std term, can vary depending on arch
        for name, param in self.actor_critic.state_dict().items():
            if "log_std" in name:
                self.log_std_param = name
                break

        self.fps = 0

    def log_metric(self, metrics, step):
        """
        Helper function to log metrics based on selected logging backend.
        """
        if self.logger_backend == "wandb":
            wandb.log(metrics, step=step)
        else:
            # For TensorBoard, add each scalar
            for key, value in metrics.items():
                self.logger.add_scalar(key, value, step)

    def train(self):
        self.save_sampled_agent_seq(0)
        obs = self.envs.reset()
        self.buffer.to(self.device)
        self.start = time.time()

        print('obs')
        print(type(obs), len(obs))
        for key in obs:
            print(key, obs[key].size())

        if cfg.PPO.MAX_ITERS > 1000:
            self.stat_save_freq = 100
        else:
            self.stat_save_freq = 10

        for cur_iter in range(cfg.PPO.MAX_ITERS):

            if cfg.PPO.EARLY_EXIT and cur_iter >= cfg.PPO.EARLY_EXIT_MAX_ITERS:
                break

            lr = ou.get_iter_lr(cur_iter)
            ou.set_lr(self.optimizer, lr, self.lr_scale)

            for step in range(cfg.PPO.TIMESTEPS):
                if cfg.MODEL.TRANSFORMER.PER_NODE_EMBED:
                    unimal_ids = self.envs.get_unimal_idx()
                else:
                    unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
                # Sample actions
                val, act, logp, dropout_mask_v, dropout_mask_mu = self.agent.act(obs, unimal_ids=unimal_ids)

                next_obs, reward, done, infos = self.envs.step(act)

                self.train_meter.add_ep_info(infos)
                # Optionally log additional info per step if needed.
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device,
                )
                timeouts = torch.tensor(
                    [[0.0] if "timeout" in info.keys() else [1.0] for info in infos],
                    dtype=torch.float32,
                    device=self.device,
                )

                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts,
                                   dropout_mask_v, dropout_mask_mu, unimal_ids)
                obs = next_obs

            if cfg.MODEL.TRANSFORMER.PER_NODE_EMBED:
                unimal_ids = self.envs.get_unimal_idx()
            else:
                unimal_ids = [0 for _ in range(cfg.PPO.NUM_ENVS)]
            next_val = self.agent.get_value(obs, unimal_ids=unimal_ids)
            self.buffer.compute_returns(next_val)
            self.train_on_batch(cur_iter)
            self.save_sampled_agent_seq(cur_iter)

            self.train_meter.update_mean()
            # ----------Log each morphs stats --------------
            env_step = self.env_steps_done(cur_iter)

            for agent_name, agent_meter in self.train_meter.agent_meters.items():
                if agent_meter.ep_count > 0 and agent_meter.mean_ep_rews.get('reward'):
                    # mean reward per morph(agent)
                    self.log_metric({f"Agent/{agent_name}/Reward":agent_meter.mean_ep_rews["reward"][-1]}, env_step)
                    # mean success rate per morph(agent)
                    if agent_meter.mean_success:
                        self.log_metric({f"Agent/{agent_name}/Success":agent_meter.mean_success[-1]}, env_step)
                    # min/max/median rewards per morph 
                    # No need for this now (console log is enough)

            # ----------Log each morphs stats --------------

            if len(self.train_meter.overall_mean_ep_rews["reward"]):
                cur_rew = self.train_meter.overall_mean_ep_rews["reward"][-1]
                self.log_metric({"Reward": cur_rew}, self.env_steps_done(cur_iter))
            
            if self.train_meter.overall_mean_success_rate:
                    self.log_metric({"Overall/Success": self.train_meter.overall_mean_success_rate[-1]}, env_step)

            if cur_iter >= 0 and cur_iter % cfg.LOG_PERIOD == 0 and cfg.LOG_PERIOD > 0:
                self._log_stats(cur_iter)

                file_name = "{}_results.json".format(self.file_prefix)
                path = os.path.join(cfg.OUT_DIR, file_name)
                self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
                stats = self.train_meter.get_stats()
                stats["fps"] = self.fps
                fu.save_json(stats, path)
                print(cfg.OUT_DIR)

            if cur_iter % 100 == 0:
                self.save_model(cur_iter)

        print("Finished Training: {}".format(self.file_prefix))

    def train_on_batch(self, cur_iter):
        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for i in range(cfg.PPO.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)

            for j, batch in enumerate(batch_sampler):
                val, _, logp, ent, _, _ = self.actor_critic(
                    batch["obs"],
                    batch["act"],
                    dropout_mask_v=batch['dropout_mask_v'],
                    dropout_mask_mu=batch['dropout_mask_mu'],
                    unimal_ids=batch['unimal_ids']
                )
                clip_ratio = cfg.PPO.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])
                approx_kl = (batch["logp_old"] - logp).mean().item()

                if cfg.PPO.KL_TARGET_COEF is not None and approx_kl > cfg.PPO.KL_TARGET_COEF * 0.01:
                    self.train_meter.add_train_stat("approx_kl", approx_kl)
                    print(f'early stop iter {cur_iter} at epoch {i + 1}/{cfg.PPO.EPOCHS}, batch {j + 1} with approx_kl {approx_kl}')
                    return

                surr1 = ratio * batch["adv"]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
                clip_frac = (ratio != surr2).float().mean().item()
                surr2 *= batch["adv"]

                pi_loss = -torch.min(surr1, surr2).mean()

                if cfg.PPO.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(-clip_ratio, clip_ratio)
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()
                loss = val_loss * cfg.PPO.VALUE_COEF + pi_loss - ent * cfg.PPO.ENTROPY_COEF
                loss.backward()

                norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), cfg.PPO.MAX_GRAD_NORM)
                log_std = self.actor_critic.state_dict()[self.log_std_param].cpu().numpy()[0]
                std = np.mean(np.exp(log_std))

                self.log_metric({
                    "approx_kl": approx_kl,
                    "pi_loss": pi_loss.item(),
                    "val_loss": val_loss.item(),
                    "clip_frac": clip_frac,
                    "std": float(std),
                    "surr1": surr1.mean().item(),
                    "surr2": surr2.mean().item(),
                    "ratio": ratio.mean().item(),
                    "grad_norm": norm.item()
                }, self.env_steps_done(cur_iter))
                self.optimizer.step()

                # Optionally, you can log weight and gradient histograms
                if cfg.SAVE_HIST_WEIGHTS:
                    for name, weight in self.actor_critic.named_parameters():
                        self.log_metric({f"weights/{name}": wandb.Histogram(weight.cpu().detach().numpy())
                                         } if self.logger_backend == "wandb" else {f"weights/{name}": weight.cpu().detach().numpy()}, self.env_steps_done(cur_iter))
                        if weight.grad is not None:
                            self.log_metric({f"gradients/{name}": wandb.Histogram(weight.grad.cpu().detach().numpy())
                                             } if self.logger_backend == "wandb" else {f"gradients/{name}": weight.grad.cpu().detach().numpy()}, self.env_steps_done(cur_iter))

    def save_model(self, cur_iter, path=None):
        if not path:
            path = os.path.join(cfg.OUT_DIR, self.file_prefix + ".pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], path)
        checkpoint_path = os.path.join(cfg.OUT_DIR, f"checkpoint_{cur_iter}.pt")
        torch.save([self.actor_critic, get_ob_rms(self.envs)], checkpoint_path)
        if self.logger_backend == "wandb":
            artifact = wandb.Artifact(
                f"{cfg.ENV_NAME}-{cfg.ROBOSUITE.TASK_TYPE}-ST-{os.path.basename(cfg.OUT_DIR)}-run",
                type="model"
            )
            artifact.add_dir(cfg.OUT_DIR)
            wandb.log_artifact(artifact)

    def _log_stats(self, cur_iter):
        self._log_fps(cur_iter)
        self.train_meter.log_stats()
        # Success histograms (per-morphology and overall)
        overall = []
        for name, m in self.train_meter.agent_meters.items():
            if m.ep_success:
                data = np.array(list(m.ep_success), dtype=int)
                overall.extend(data.tolist())
                if self.logger_backend == "wandb":
                    wandb.log(
                        {f"{name}/Success_Histogram": wandb.Histogram(data)},
                        step=self.env_steps_done(cur_iter),
                        )
                else:
                    try:
                        # TensorBoard
                        self.logger.add_histogram(
                            f"Agent/{name}/Success_Histogram",
                            data,
                            global_step=self.env_steps_done(cur_iter),
                            )
                    except TypeError:
                        # it passes a dtype= kwarg into np.greater, which newer NumPy versions no longer accept
                        print("[PPO] Failed to log histogram to TensorBoard due to TypeError.")
                        pass
        if overall:
            overall = np.array(overall, dtype=int)
            if self.logger_backend == "wandb":
                wandb.log(
                    {"Overall/Success_Histogram": wandb.Histogram(overall)},
                    step=self.env_steps_done(cur_iter),
                    )
            else:
                try:
                    self.logger.add_histogram(
                        "Overall/Success_Histogram",
                        overall,
                        global_step=self.env_steps_done(cur_iter),
                    )
                except TypeError:
                    # it passes a dtype= kwarg into np.greater, which newer NumPy versions no longer accept
                    print("[PPO] Failed to log histogram to TensorBoard due to TypeError.")
                    pass

    def _log_fps(self, cur_iter, log=True):
        env_steps = self.env_steps_done(cur_iter)
        end = time.time()
        self.fps = int(env_steps / (end - self.start))
        if log:
            print("Updates {}, num timesteps {}, FPS {}".format(cur_iter, env_steps, self.fps))

    def env_steps_done(self, cur_iter):
        return (cur_iter + 1) * cfg.PPO.NUM_ENVS * cfg.PPO.TIMESTEPS

    def save_rewards(self, path=None, hparams=None):
        if not path:
            file_name = "{}_results.json".format(self.file_prefix)
            path = os.path.join(cfg.OUT_DIR, file_name)

        self._log_fps(cfg.PPO.MAX_ITERS - 1, log=False)
        stats = self.train_meter.get_stats()
        stats["fps"] = self.fps
        fu.save_json(stats, path)

        if hparams:
            hparams_to_save = {k: v for k, v in hparams.items() if not isinstance(v, list)}
            final_env_reward = np.mean(stats["__env__"]["reward"]["reward"][-100:])
            wandb.config.update(hparams_to_save) if self.logger_backend == "wandb" else None
            self.log_metric({"final_reward": final_env_reward}, self.env_steps_done(cfg.PPO.MAX_ITERS - 1))

        if self.logger_backend == "wandb":
            wandb.finish()

    def save_video(self, save_dir, xml=None):
        env = make_vec_envs(training=False, norm_rew=False, save_video=True, xml_file=xml)
        set_ob_rms(env, get_ob_rms(self.envs))
        env = VecVideoRecorder(
            env,
            save_dir,
            record_video_trigger=lambda x: x == 0,
            video_length=cfg.PPO.VIDEO_LENGTH,
            file_prefix=xml,
        )
        obs = env.reset()
        returns = []
        episode_count = 0
        for t in range(cfg.PPO.VIDEO_LENGTH + 1):
            _, act, _, _, _ = self.agent.act(obs)
            obs, _, _, infos = env.step(act)
            if 'episode' in infos[0]:
                print(infos[0]['episode']['r'])
                returns.append(infos[0]['episode']['r'])
                episode_count += 1
                if episode_count == 5:
                    break
        env.close()
        avg_return = int(np.array(returns).mean())
        os.remove(os.path.join(save_dir, f"{xml}_video.meta.json"))
        os.rename(os.path.join(save_dir, f"{xml}_video.mp4"),
                  os.path.join(save_dir, f"{xml}_video_{avg_return}.mp4"))
        video_path = os.path.join(save_dir, f"{xml}_video_{avg_return}.mp4")
        # If needed, log the video using the appropriate backend.
        # self.log_metric({"video": wandb.Video(video_path, fps=cfg.VIDEO.FPS, format="mp4")},
        #                 step=self.env_steps_done(0))
        return returns

    def save_sampled_agent_seq(self, cur_iter):
        num_agents = len(cfg.ENV.WALKERS)
        if num_agents <= 1:
            return
        if cfg.ENV.TASK_SAMPLING == "uniform_random_strategy":
            ep_lens = [1000] * num_agents
        elif cfg.ENV.TASK_SAMPLING == "balanced_replay_buffer":
            if cur_iter < 30:
                ep_lens = [1000] * num_agents
            else:
                if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
                    ep_lens = [np.mean(self.train_meter.agent_meters[agent].ep_len_ema) for agent in cfg.ENV.WALKERS]
                elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
                    ep_lens = [np.mean(self.train_meter.agent_meters[agent].ep_len) for agent in cfg.ENV.WALKERS]
        probs = [1000.0 / l for l in ep_lens]
        probs = np.power(probs, cfg.TASK_SAMPLING.PROB_ALPHA)
        probs = [p / sum(probs) for p in probs]
        avg_ep_len = np.mean([np.mean(self.train_meter.agent_meters[agent].ep_len) for agent in cfg.ENV.WALKERS])
        if np.isnan(avg_ep_len):
            avg_ep_len = 100
        ep_per_env = cfg.PPO.TIMESTEPS / avg_ep_len
        size = int(ep_per_env * cfg.PPO.NUM_ENVS * 50)
        task_list = np.random.choice(range(0, num_agents), size=size, p=probs)
        task_list = [int(_) for _ in task_list]
        path = os.path.join(cfg.OUT_DIR, "sampling.json")
        fu.save_json(task_list, path)
