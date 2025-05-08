from collections import defaultdict
from collections import deque

import numpy as np

from metamorph.config import cfg


class AgentMeter:
    """
    Tracks statistics for a single morphology encountered during training.
    Uses deques to maintain sliding windows of recent episode stats and
    deques of means to track performance over batches of episodes.
    """
    def __init__(self, name):
        self.name = name
        
        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_success = deque(maxlen=10)
        self.ep_len = deque(maxlen=10)

       
        self.mean_ep_rews_queue = defaultdict(lambda: deque(maxlen=100))
        self.mean_pos_queue = deque(maxlen=100)
        self.mean_vel_queue = deque(maxlen=100)
        self.mean_metric_queue = deque(maxlen=100)
        self.mean_ep_len_queue = deque(maxlen=100)
        self.mean_ep_success_rate_queue = deque(maxlen=100)

        # Total episode count for this morphology
        self.ep_count = 0
        # Exponential moving average of episode length (optional, based on config)
        self.ep_len_ema = -1 # Initialize EMA

    def add_single_ep_info(self, ep_info: dict):
        """
        Adds statistics from a *single* completed episode.
        These stats are added to the ephemeral deques.
        """
        # Add basic episode stats
        self.ep_rew["reward"].append(ep_info["r"])
        self.ep_len.append(ep_info["l"])
        self.ep_count += 1

        # Add optional stats if present in ep_info
        if "success" in ep_info: self.ep_success.append(ep_info["success"])
        if "x_pos" in ep_info: self.ep_pos.append(ep_info["x_pos"])
        if "x_vel" in ep_info: self.ep_vel.append(ep_info["x_vel"])
        if "metric" in ep_info: self.ep_metric.append(ep_info["metric"])

        # Add reward components if present (keys starting with __reward__)
        for key, value in ep_info.items():
            if "__reward__" in key and key not in self.ep_rew: # Avoid double-adding '__reward__reward'
                 self.ep_rew[key].append(value)

        # Update EMA for episode length (This logic updates with every episode)
        if cfg.TASK_SAMPLING.AVG_TYPE == "ema":
             if self.ep_len_ema == -1: # First episode
                  self.ep_len_ema = ep_info["l"]
             else:
                  alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                  self.ep_len_ema = alpha * ep_info["l"] + (1 - alpha) * self.ep_len_ema
        elif cfg.TASK_SAMPLING.AVG_TYPE == "moving_window":
             # EMA is not used, mean is over the deque window handled by maxlen
             pass # ep_len deque already updated


    def calculate_and_store_mean(self) -> bool:
        """
        Calculates the mean of the current ephemeral deque stats and appends
        them to the historical mean queues.
        Returns True if any episodes were processed in this batch, False otherwise.
        """
        # Check if any episodes were added since the last update
        if not self.ep_rew["reward"]:
            return False # No new episodes to process

        for rew_type, rews_list in self.ep_rew.items():
             if rews_list: # Only calculate if deque is not empty
                 self.mean_ep_rews_queue[rew_type].append(round(np.mean(list(rews_list)), 2)) # Convert deque to list for mean

        if self.ep_pos: self.mean_pos_queue.append(round(np.mean(list(self.ep_pos)), 2))
        if self.ep_vel: self.mean_vel_queue.append(round(np.mean(list(self.ep_vel)), 2))
        if self.ep_metric: self.mean_metric_queue.append(round(np.mean(list(self.ep_metric)), 2))
        if self.ep_len: self.mean_ep_len_queue.append(round(np.mean(list(self.ep_len)), 2))
        if self.ep_success: self.mean_ep_success_rate_queue.append(round(np.mean(list(self.ep_success)), 3))


       
        return True 


    def log_stats(self, max_name_len):
        """Logs the most recent average stats for this specific morphology."""
        # Check if there are any historical mean stats to log
        if not self.mean_ep_rews_queue.get("reward"):
            return # No stats collected yet

        # Get the latest calculated mean stats from the historical mean queues
        latest_mean_rew = self.mean_ep_rews_queue["reward"][-1]
        latest_mean_success = self.mean_ep_success_rate_queue[-1] if self.mean_ep_success_rate_queue else "N/A"
        latest_mean_ep_len = self.mean_ep_len_queue[-1] if self.mean_ep_len_queue else "N/A"

        print(
             "Agent {:>{size}}: mean reward {:>8.2f}, success: {}, avg Ep len: {:>5.1f}".format(
                 self.name,
                 latest_mean_rew,
                 latest_mean_success,
                 latest_mean_ep_len,
                 size=max_name_len,
             )
        )

    def get_stats(self):
        """
        Returns the history of calculated mean statistics for this morphology.
        These are the values stored in the mean_*_queue deques.
        """
        # Convert deques to lists for static output
        stats = {
            "reward": list(self.mean_ep_rews_queue["reward"]),
            "reward_components": {k: list(v) for k, v in self.mean_ep_rews_queue.items()},
            "pos": list(self.mean_pos_queue),
            "vel": list(self.mean_vel_queue),
            "metric": list(self.mean_metric_queue),
            "ep_len": list(self.mean_ep_len_queue),
            "success_rate": list(self.mean_ep_success_rate_queue),
            "total_episodes": self.ep_count 
        }
        return stats


class TrainMeter:
    """
    Aggregates statistics across all active AgentMeters for different morphologies.
    Dynamically creates AgentMeters as new morphology names are encountered.
    Calculates overall average statistics across all tracked morphologies.
    """
    def __init__(self):
        # AgentMeters are now dynamically added as morphologies are encountered
        self.agent_meters = {} # Dictionary to store AgentMeter instances, keyed by morphology name

        # Maximum length of morphology names for formatted printing
        self.max_name_len = 0

        self.overall_mean_ep_rews = defaultdict(list)
        self.overall_mean_pos = []
        self.overall_mean_vel = []
        self.overall_mean_metric = []
        self.overall_mean_ep_len = []
        self.overall_mean_ep_success_rate = []

        # Other training stats not specific to a morphology
        self.train_stats = defaultdict(list)


    def add_train_stat(self, stat_type: str, stat_value: float):
        """Adds a training statistic (e.g., loss, KL divergence)."""
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos: list):
        """
        Processes a list of info dictionaries from a VecEnv step.
        Adds episode info to the appropriate AgentMeter.
        Dynamically creates AgentMeters for new morphology names.
        """
        for info in infos:
            if "episode" in info:
                # Episode finished in one of the environments
                morphology_name = info.get("name", "unknown_morphology") # Get morphology name

                # Create AgentMeter if this is the first time seeing this morphology
                if morphology_name not in self.agent_meters:
                    self.agent_meters[morphology_name] = AgentMeter(morphology_name)
                    self.max_name_len = max(self.max_name_len, len(morphology_name))
                    # print(f"TrainMeter: Tracking new morphology: {morphology_name}") 

                self.agent_meters[morphology_name].add_single_ep_info(info["episode"])


    def update_mean(self):
        """
        Updates all individual AgentMeters and then calculates the overall
        mean statistics across all morphologies encountered so far.
        This is typically called after collecting a batch of experience.
        """
        
        updated_meters_count = 0
        for _, agent_meter in self.agent_meters.items():
            if agent_meter.calculate_and_store_mean(): # If the meter had new episodes
                updated_meters_count += 1

        # Only calculate overall means if at least one meter had new episodes processed
        if updated_meters_count == 0:
             return # No new stats to average

        # Calculate overall mean statistics by averaging the latest mean from each meter
        # We only average over meters that have collected at least one episode in total (ep_count > 0)
        # and had new episodes processed in this batch (indicated by calculate_and_store_mean returning True and having new mean in queue)

        active_meters = [m for m in self.agent_meters.values() if m.ep_count > 0 and m.mean_ep_rews_queue.get("reward")] # Only include meters that have logged at least one mean batch

        if not active_meters:
            return # No active meters with logged means

        # Collect the latest mean values from each active meter
        latest_means = {
            "reward": [], "pos": [], "vel": [], "metric": [], "ep_len": [], "success_rate": []
        }
        reward_component_types = set()
        for meter in active_meters:
            latest_means["reward"].append(meter.mean_ep_rews_queue["reward"][-1])
            if meter.mean_pos_queue: latest_means["pos"].append(meter.mean_pos_queue[-1])
            if meter.mean_vel_queue: latest_means["vel"].append(meter.mean_vel_queue[-1])
            if meter.mean_metric_queue: latest_means["metric"].append(meter.mean_metric_queue[-1])
            if meter.mean_ep_len_queue: latest_means["ep_len"].append(meter.mean_ep_len_queue[-1])
            if meter.mean_ep_success_rate_queue: latest_means["success_rate"].append(meter.mean_ep_success_rate_queue[-1])

            # Collect all reward component types across meters
            reward_component_types.update(meter.mean_ep_rews_queue.keys())


        # Calculate the overall average and append to overall history lists
        if latest_means["reward"]: self.overall_mean_ep_rews["reward"].append(round(np.mean(latest_means["reward"]), 2))
        if latest_means["pos"]: self.overall_mean_pos.append(round(np.mean(latest_means["pos"]), 2))
        if latest_means["vel"]: self.overall_mean_vel.append(round(np.mean(latest_means["vel"]), 2))
        if latest_means["metric"]: self.overall_mean_metric.append(round(np.mean(latest_means["metric"]), 2))
        if latest_means["ep_len"]: self.overall_mean_ep_len.append(round(np.mean(latest_means["ep_len"]), 2))
        if latest_means["success_rate"]: self.overall_mean_ep_success_rate.append(round(np.mean(latest_means["success_rate"]), 3))

        # Calculate overall mean for reward components
        for rew_type in reward_component_types:
            rew_comp_means = [m.mean_ep_rews_queue[rew_type][-1] for m in active_meters if m.mean_ep_rews_queue.get(rew_type)]
            if rew_comp_means:
                 self.overall_mean_ep_rews[rew_type].append(round(np.mean(rew_comp_means), 2))


    def log_stats(self):
        """
        Logs the most recent average stats for each individual morphology
        and the overall average stats across all morphologies.
        """
        # Log stats for each individual morphology that has collected any episodes
        for name, agent_meter in self.agent_meters.items():
            if agent_meter.ep_count > 0: # Only log if meter is active
                 agent_meter.log_stats(self.max_name_len)

        # Log overall environment stats if any overall stats have been calculated
        if self.overall_mean_ep_rews.get("reward"):
            latest_overall_reward = self.overall_mean_ep_rews["reward"][-1]
            latest_overall_success = self.overall_mean_ep_success_rate[-1] if self.overall_mean_ep_success_rate else "N/A"

            print(
                 "Agent {:>{size}}: mean reward {:>8.2f}, mean success: {}".format(
                     "__env__", # Special name for overall stats
                     latest_overall_reward,
                     latest_overall_success,
                     size=self.max_name_len
                 )
            )
        else:
             print(f"Agent {'__env__':>{self.max_name_len}}: No overall stats yet.")


    def get_stats(self):
        """
        Returns a dictionary containing historical mean stats for each individual
        morphology and the overall historical mean stats.
        """
        stats = {}
        # Get stats for each individual morphology that has collected any episodes
        for agent_name, agent_meter in self.agent_meters.items():
            if agent_meter.ep_count > 0: # Only include active meters
                 stats[agent_name] = agent_meter.get_stats()

        stats["__env__"] = {
                "reward": list(self.overall_mean_ep_rews.get("reward", [])),
                "reward_components": {k: list(v) for k, v in self.overall_mean_ep_rews.items()},
                "pos": list(self.overall_mean_pos),
                "vel": list(self.overall_mean_vel),
                "metric": list(self.overall_mean_metric),
                "ep_len": list(self.overall_mean_ep_len),
                "success_rate": list(self.overall_mean_ep_success_rate),
        }
        stats["__env__"].update(dict(self.train_stats))

        return stats