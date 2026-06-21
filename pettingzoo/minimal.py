from pettingzoo import ParallelEnv
from gymnasium import spaces


class MinimalPettingZooEnv(ParallelEnv):
    metadata = {
        "name": "minimal_pettingzoo_v0",
        "render_modes": ["human"],
    }

    def __init__(self):
        self.possible_agents = ["player_0", "player_1"]
        self.agents = []

        self._action_spaces = {
            "player_0": spaces.Discrete(2),
            "player_1": spaces.Discrete(2),
        }

        self._observation_spaces = {
            "player_0": spaces.Discrete(1),
            "player_1": spaces.Discrete(1),
        }

    def observation_space(self, agent):
        return self._observation_spaces[agent]

    def action_space(self, agent):
        return self._action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]

        observations = {
            "player_0": 0,
            "player_1": 0,
        }

        infos = {
            "player_0": {},
            "player_1": {},
        }

        return observations, infos

    def step(self, actions):
        action_0 = actions["player_0"]
        action_1 = actions["player_1"]

        same_action = action_0 == action_1

        if same_action:
            rewards = {
                "player_0": 1,
                "player_1": -1,
            }
        else:
            rewards = {
                "player_0": -1,
                "player_1": 1,
            }

        observations = {
            "player_0": 0,
            "player_1": 0,
        }

        terminations = {
            "player_0": True,
            "player_1": True,
        }

        truncations = {
            "player_0": False,
            "player_1": False,
        }

        infos = {
            "player_0": {},
            "player_1": {},
        }

        self.agents = []

        return observations, rewards, terminations, truncations, infos