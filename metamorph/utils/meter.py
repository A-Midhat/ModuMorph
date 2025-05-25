from collections import defaultdict
from collections import deque

import numpy as np

from metamorph.config import cfg


class AgentMeter:
    def __init__(self, name):
        self.name = name
        self.mean_ep_rews = defaultdict(list)
        self.mean_pos = []
        self.mean_vel = []
        self.mean_metric = []
        self.mean_ep_len = []
        # list to store historical windowed mean success rate
        self.mean_success = []
        # lists to store historical min/max/median rewards (windowed)
        self.mean_min_rew = []
        self.mean_max_rew = []
        self.mean_median_rew = []

        self.ep_rew = defaultdict(lambda: deque(maxlen=10))
        self.ep_pos = deque(maxlen=10)
        self.ep_vel = deque(maxlen=10)
        self.ep_metric = deque(maxlen=10)
        self.ep_count = 0
        self.ep_len = deque(maxlen=10)
        self.ep_len_ema = -1
        # for robosuite envs
        self.ep_success = deque(maxlen=10)
        self.success_history = []
        self.total_success = 0
        self.first_success_iter = None 
        # stores original config index (Populated by TrainMeter)
        self.morph_config_index = None


    def add_ep_info(self, infos, cur_iter=None):
        # Process list of infos received from VecEnv.step().
        # We only process episode info that matches this agent's name.
        for info in infos:
            if info.get("name") != self.name or "episode" not in info.keys():
                continue

            # Episode finished in this environment
            episode_info = info["episode"]

            # Add base reward and length
            self.ep_rew["reward"].append(episode_info["r"])
            self.ep_count += 1
            self.ep_len.append(episode_info["l"])

            # Update episode length EMA
            if self.ep_count == 10:
                self.ep_len_ema = np.mean(self.ep_len)
            elif self.ep_count >= 10:
                alpha = cfg.TASK_SAMPLING.EMA_ALPHA
                self.ep_len_ema = (
                    alpha * self.ep_len[-1] + (1 - alpha) * self.ep_len_ema
                )

            # Add success flag if present in episode info
            if "success" in episode_info:
                   succ = bool(episode_info["success"])
                   self.ep_success.append(succ) # moving avg 
                   self.success_history.append(succ) # full 
                   if succ:
                      self.total_success += 1
                      if self.first_success_iter is None:
                          self.first_success_iter = cur_iter
            # Add other reward components
            for rew_type, rew_value in episode_info.items():
                if rew_type.startswith("__reward__") and rew_type != "reward":
                    self.ep_rew[rew_type].append(rew_value)

            # Add other optional metrics if present
            if "x_pos" in episode_info:
                self.ep_pos.append(episode_info["x_pos"])
            if "x_vel" in episode_info:
                self.ep_vel.append(episode_info["x_vel"])
            if "metric" in episode_info:
                self.ep_metric.append(episode_info["metric"])


    def update_mean(self):
        # Check if any episodes were added since the last update
        if len(self.ep_rew["reward"]) == 0:
            return False # No new episodes to process

        # Calculate mean reward components and append to mean queues
        for rew_type, rews_list in self.ep_rew.items():
             if rews_list:
                 self.mean_ep_rews[rew_type].append(round(np.mean(list(rews_list)), 2))

        # Calculate mean for other metrics and append if deque is not empty
        if self.ep_pos: self.mean_pos.append(round(np.mean(list(self.ep_pos)), 2))
        if self.ep_vel: self.mean_vel.append(round(np.mean(list(self.ep_vel)), 2))
        if self.ep_metric: self.mean_metric.append(round(np.mean(list(self.ep_metric)), 2))
        if self.ep_len: self.mean_ep_len.append(round(np.mean(list(self.ep_len)), 2))

        # Calculate mean success rate and append to its list
        if self.ep_success:
            self.mean_success.append(round(float(np.mean(self.ep_success)), 3))
        # Calculate min/max/median rewards and append to their lists
        if self.ep_rew["reward"]: # Ensure there are rewards to calculate stats on
            self.mean_min_rew.append(round(np.min(list(self.ep_rew["reward"])), 2))
            self.mean_max_rew.append(round(np.max(list(self.ep_rew["reward"])), 2))
            self.mean_median_rew.append(round(np.median(list(self.ep_rew["reward"])), 2))


        # Note: Ephemeral deques (self.ep_*) are not cleared here, relying on maxlen

        return True # Indicates that stats for this morphology were updated

    def log_stats(self, max_name_len):
        # Check if there are any historical mean stats to log (using main reward queue)
        if not self.mean_ep_rews.get("reward"):
            return # No stats collected yet

        # Get the latest calculated mean reward and min/max/median from history lists
        latest_mean_rew = self.mean_ep_rews["reward"][-1]
        latest_min_rew = self.mean_min_rew[-1] if self.mean_min_rew else "N/A"
        latest_max_rew = self.mean_max_rew[-1] if self.mean_max_rew else "N/A"
        latest_median_rew = self.mean_median_rew[-1] if self.mean_median_rew else "N/A"


        print_str = "Agent {:>{size}}: mean/median reward {:>8.2f}/{:<8.2f}, min/max reward {:>8.2f}/{:<8.2f}".format(
            self.name,
            latest_mean_rew, latest_median_rew, latest_min_rew, latest_max_rew,
            size=max_name_len
        )

        # Get latest mean success rate and episode length if available
        latest_mean_success = self.mean_success[-1] if self.mean_success else "N/A"
        latest_mean_ep_len = self.mean_ep_len[-1] if self.mean_ep_len else "N/A"


        print_str += ", #Ep: {:>7.0f}, avg/ema Ep len: {:>5.1f}/{:>5.1f}".format(
            self.ep_count, latest_mean_ep_len, round(self.ep_len_ema, 1) # Use latest calculated mean ep len and EMA
        )

        if latest_mean_success != "N/A":
            extra = f"{latest_mean_success}"
            if self.total_success:
                extra += f" (total {self.total_success}"
                if self.first_success_iter is not None:
                    extra += f", first at {self.first_success_iter} iter"
                extra += ")"
            print_str += ", success: " + extra
        # Print a newline character to finish the line
        print(print_str)


    def get_stats(self):
        # Returns a dictionary containing historical mean stats for this morphology.
        stats = {
            "reward": list(self.mean_ep_rews.get("reward", [])), # Get main reward history
            # Get reward components history
            "reward_components": {k: list(v) for k, v in self.mean_ep_rews.items() if k != "reward"},
            "pos": list(self.mean_pos),
            "vel": list(self.mean_vel),
            "metric": list(self.mean_metric),
            "ep_len": list(self.mean_ep_len),
            # Include success rate history
            "success_rate": list(self.mean_success),
            "min_reward": list(self.mean_min_rew),
            "max_reward": list(self.mean_max_rew),
            "median_reward": list(self.mean_median_rew),
            "total_episodes": self.ep_count # Total episodes collected over training for this morph
        }
        # Return the stats dictionary
        return stats


class TrainMeter:
    def __init__(self):
        # Removed cfg.ENV.WALKERS dependency; agents are dynamically added
        self.agent_meters = {} # Dictionary to store AgentMeter instances, keyed by morphology name (e.g. "Panda_Lift")

        # Maximum length of morphology names for formatted printing
        self.max_name_len = 0 # Initialized to 0

        # Overall statistics, averaged across all active AgentMeters
        # These store the history of overall means (e.g., mean reward across all morphs for each logging period)
        self.overall_mean_ep_rews = defaultdict(list)
        self.overall_mean_pos = []
        self.overall_mean_vel = []
        self.overall_mean_metric = []
        self.overall_mean_ep_len = []
        # list for overall mean success rate history
        self.overall_mean_success_rate = []
        # lists for overall min/max/median rewards history
        self.overall_mean_min_rew = []
        self.overall_mean_max_rew = []
        self.overall_mean_median_rew = []


        # Other training stats not specific to a morphology 
        self.train_stats = defaultdict(list)

        # mapping from unique morphology name to its original config index
        # This map is needed by save_sampled_agent_seq in ppo.py
        # It's populated when add_ep_info sees 'morph_config_index' in info
        self.morph_name2cfg_idx_map = {}


    def add_train_stat(self, stat_type, stat_value):
        self.train_stats[stat_type].append(stat_value)

    def add_ep_info(self, infos: list, cur_iter = None):
        # Process infos to discover new agents and update name-to-index map
        # Also pass info to each agent meter.
        for info in infos:
            if "episode" in info:
                morphology_name = info.get("name")
                morph_config_index = info.get("morph_config_index") # Get original config index

                if morphology_name and morphology_name not in self.agent_meters:
                    self.agent_meters[morphology_name] = AgentMeter(morphology_name)
                    self.max_name_len = max(self.max_name_len, len(morphology_name)) # Update max name length
                if morphology_name and morph_config_index is not None:
                    self.morph_name2cfg_idx_map[morphology_name] = morph_config_index # Store the map

        # Pass the list of infos to each agent meter's add_ep_info method
        for _, agent_meter in self.agent_meters.items():
            agent_meter.add_ep_info(infos, cur_iter)


    def update_mean(self):
        # First, trigger all individual AgentMeters to calculate and store their means
        updated_meters_count = 0
        for _, agent_meter in self.agent_meters.items():
            if agent_meter.update_mean(): # If the meter had new episodes processed
                updated_meters_count += 1

        # Only calculate overall means if at least one meter had new episodes processed
        if updated_meters_count == 0 or not self.agent_meters:
             return # No new stats to average or no meters exist

        # Collect the latest mean values from each *active* meter
        active_meters = [m for m in self.agent_meters.values() if m.mean_ep_rews.get("reward")] # Only include meters that have logged reward means

        if not active_meters:
            return # No active meters with logged means this round

        # Calculate overall average for each metric and append to overall history lists
        latest_reward_means = [m.mean_ep_rews["reward"][-1] for m in active_meters]
        latest_pos_means = [m.mean_pos[-1] for m in active_meters if m.mean_pos]
        latest_vel_means = [m.mean_vel[-1] for m in active_meters if m.mean_vel]
        latest_metric_means = [m.mean_metric[-1] for m in active_meters if m.mean_metric]
        latest_ep_len_means = [m.mean_ep_len[-1] for m in active_meters if m.mean_ep_len]
        latest_success_rate_means = [m.mean_success[-1] for m in active_meters if m.mean_success] # Collect latest mean success rate from each agent that has data
        latest_min_rew_means = [m.mean_min_rew[-1] for m in active_meters if m.mean_min_rew]
        latest_max_rew_means = [m.mean_max_rew[-1] for m in active_meters if m.mean_max_rew]
        latest_median_rew_means = [m.mean_median_rew[-1] for m in active_meters if m.mean_median_rew]


        if latest_reward_means:
            self.overall_mean_ep_rews["reward"].append(round(np.mean(latest_reward_means), 2))
        if latest_pos_means:
            self.overall_mean_pos.append(round(np.mean(latest_pos_means), 2))
        if latest_vel_means:
            self.overall_mean_vel.append(round(np.mean(latest_vel_means), 2))
        if latest_metric_means:
            self.overall_mean_metric.append(round(np.mean(latest_metric_means), 2))
        if latest_ep_len_means:
            self.overall_mean_ep_len.append(round(np.mean(latest_ep_len_means), 2))
        if latest_success_rate_means:
            self.overall_mean_success_rate.append(round(np.mean(latest_success_rate_means), 3)) 
        if latest_min_rew_means:
            self.overall_mean_min_rew.append(round(np.mean(latest_min_rew_means), 2)) 
        if latest_max_rew_means:
            self.overall_mean_max_rew.append(round(np.mean(latest_max_rew_means), 2)) 
        if latest_median_rew_means:
            self.overall_mean_median_rew.append(round(np.mean(latest_median_rew_means), 2)) 


        # Calculate overall mean for reward components across active meters
        reward_component_types = set()
        for meter in active_meters:
             reward_component_types.update(meter.mean_ep_rews.keys()) # Collect all reward component types

        for rew_type in reward_component_types:
            if rew_type != "reward": # Skip the main reward which is handled above
                # Collect the latest mean for this reward component from all active meters
                rew_comp_means = [m.mean_ep_rews[rew_type][-1] for m in active_meters if m.mean_ep_rews.get(rew_type)]
                if rew_comp_means: # Calculate and append only if there are means to average
                     self.overall_mean_ep_rews[rew_type].append(round(np.mean(rew_comp_means), 2))


    def log_stats(self):
        # Log stats for each individual morphology that has collected any episodes
        # and has logged at least one batch of means.
        for name, agent_meter in self.agent_meters.items():
            if agent_meter.ep_count > 0 and agent_meter.mean_ep_rews.get('reward'):
                 agent_meter.log_stats(self.max_name_len)

        # Log overall environment stats if any overall stats have been calculated
        if self.overall_mean_ep_rews.get("reward"):
            latest_overall_reward = self.overall_mean_ep_rews["reward"][-1]
            # Get latest overall success rate if available
            latest_overall_success = self.overall_mean_success_rate[-1] if self.overall_mean_success_rate else "N/A"
            # Get latest overall min/max/median rewards
            latest_overall_min_rew = self.overall_mean_min_rew[-1] if self.overall_mean_min_rew else "N/A"
            latest_overall_max_rew = self.overall_mean_max_rew[-1] if self.overall_mean_max_rew else "N/A"
            latest_overall_median_rew = self.overall_mean_median_rew[-1] if self.overall_mean_median_rew else "N/A"


            print_str = "Agent {:>{size}}: mean/median reward {:>8.2f}/{:<8.2f}, min/max reward {:>8.2f}/{:<8.2f}".format(
                "__env__", # Special name for overall stats
                latest_overall_reward, latest_overall_median_rew, latest_overall_min_rew, latest_overall_max_rew,
                size=self.max_name_len
            )

            if latest_overall_success != "N/A":
                 print_str += ", mean success: {}".format(latest_overall_success)

            print(print_str)

    def get_stats(self):
        # Returns a dictionary containing historical mean stats for this morphology.
        stats = {}
        # Get stats for each individual morphology that has collected any episodes
        for agent_name, agent_meter in self.agent_meters.items():
            if agent_meter.ep_count > 0: # Only include active meters
                 stats[agent_name] = agent_meter.get_stats()

        # Include overall environment stats history
        stats["__env__"] = {
            "reward": list(self.overall_mean_ep_rews.get("reward", [])),
            # Get reward components history
            "reward_components": {k: list(v) for k, v in self.overall_mean_ep_rews.items() if k != "reward"},
            "pos": list(self.overall_mean_pos),
            "vel": list(self.overall_mean_vel),
            "metric": list(self.overall_mean_metric),
            "ep_len": list(self.overall_mean_ep_len),
            # Include overall success rate history
            "success_rate": list(self.overall_mean_success_rate),
            "min_reward": list(self.overall_mean_min_rew),
            "max_reward": list(self.overall_mean_max_rew),
            "median_reward": list(self.overall_mean_median_rew),
        }
        # Include other training stats not specific to a morphology (e.g., loss values)
        stats["__env__"].update(dict(self.train_stats))

        # Store the mapping from morphology name to its original config index in the stats output
        stats["morph_name2cfg_idx_map"] = dict(self.morph_name2cfg_idx_map)

        return stats

