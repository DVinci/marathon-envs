import logging
import itertools
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
from gymnasium import error, spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple, DecisionSteps, TerminalSteps
import os
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from sys import platform


DEFAULT_EDITOR_PORT = 5004

class MarathonEnvsException(error.Error):
    """
    Any error related to the gym wrapper of ml-agents.
    """

    pass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("marathon_envs")


GymSingleStepResult = Tuple[np.ndarray, float, bool, bool, Dict]
GymMultiStepResult = Tuple[List[np.ndarray], List[float], List[bool], List[bool], Dict]
GymStepResult = Union[GymSingleStepResult, GymMultiStepResult]


class MarathonEnvs(gym.Env):
    """
    Provides Gymnasium wrapper for Unity Learning Environments.
    Multi-agent environments use lists for object types, as done here:
    https://github.com/openai/multiagent-particle-envs
    """

    def __init__(
        self,
        environment_name: str,
        num_spawn_envs: int = 1,
        worker_id: int = 0,
        marathon_envs_path: str = None,
        no_graphics: bool = False,
        use_editor: bool = False,
        inference: bool = False,
        render_mode: Optional[str] = None,
    ):
        """
        Environment initialization
        :param environment_name: The Marathon Environment
        :param num_spawn_envs: The number of environments to spawn per instance
        :param worker_id: Worker number for environment.
        :param marathon_envs_path: alternative path for environment
        :param no_graphics: Whether to run the Unity simulator in no-graphics mode
        :param use_editor: If True, assume Unity Editor is the environment (use for debugging)
        :param inference: If True, run in inference mode (normal framerate)
        :param render_mode: Gymnasium render mode (None or "rgb_array")
        """
        self.render_mode = render_mode
        multiagent: bool = True  # force multiagent

        base_port = 5005
        if use_editor:
            base_port = DEFAULT_EDITOR_PORT
            marathon_envs_path = None
        elif marathon_envs_path is None:
            marathon_envs_path = os.path.join('envs', 'MarathonEnvs')
            if platform == "win32":
                marathon_envs_path = os.path.join(marathon_envs_path, 'Unity Environment.exe')
        args = ['--spawn-env=' + environment_name]
        args.append('--num-spawn-envs=' + str(num_spawn_envs))

        engine_configuration_channel = EngineConfigurationChannel()
        channels = [engine_configuration_channel]

        self._env = UnityEnvironment(
            marathon_envs_path,
            worker_id=worker_id,
            base_port=base_port,
            side_channels=channels,
            no_graphics=no_graphics,
            args=args,
        )
        if not inference:
            engine_configuration_channel.set_configuration_parameters(
                width=160, height=160, quality_level=0,
                time_scale=20., target_frame_rate=-1)

        self._env.reset()
        behavior_names = list(self._env.behavior_specs.keys())
        if len(behavior_names) != 1:
            raise MarathonEnvsException(
                "There can only be one behavior in a UnityEnvironment "
                "if it is wrapped in a gym."
            )

        self.brain_name = behavior_names[0]
        self.name = self.brain_name
        self.group_spec = self._env.behavior_specs[self.brain_name]

        self._multiagent = multiagent
        self._n_agents = -1
        self.visual_obs = None
        self.game_over = False

        decision_steps, _ = self._env.get_steps(self.brain_name)
        self._check_agents(len(decision_steps))
        self._previous_decision_steps = decision_steps
        self.agent_mapper = AgentIdIndexMapper()
        self.agent_mapper.set_initial_agents(list(decision_steps.agent_id))

        # Set observation and action spaces
        action_spec = self.group_spec.action_spec
        if action_spec.is_discrete():
            branches = action_spec.discrete_branches
            if len(branches) == 1:
                self._action_space = spaces.Discrete(branches[0])
            else:
                self._action_space = spaces.MultiDiscrete(branches)
        else:
            high = np.array([1] * action_spec.continuous_size)
            self._action_space = spaces.Box(-high, high, dtype=np.float32)

        obs_size = self._get_vec_obs_size()
        high = np.array([np.inf] * obs_size)
        self._observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict]:
        """Resets the state of the environment and returns an initial observation.
        Returns: (observation, info)
        """
        super().reset(seed=seed)
        self._env.reset()
        decision_steps, _ = self._env.get_steps(self.brain_name)
        self._check_agents(len(decision_steps))
        self.game_over = False
        self._previous_decision_steps = decision_steps

        if not self._multiagent:
            obs = self._get_vector_obs(decision_steps)[0, :]
        else:
            obs = self._get_vector_obs(decision_steps)
        return obs, {}

    def step(self, action: Any) -> GymStepResult:
        """Run one timestep of the environment's dynamics.
        Returns: (observation, reward, terminated, truncated, info)
        """
        if self._multiagent:
            if isinstance(action, list):
                action = np.array(action)
            if isinstance(action, np.ndarray):
                if action.shape[0] != self._n_agents:
                    raise MarathonEnvsException(
                        "The environment was expecting {} actions.".format(self._n_agents)
                    )
        else:
            action = np.array(action).reshape(1, -1)

        action_spec = self.group_spec.action_spec
        action = np.array(action).reshape((self._n_agents, action_spec.continuous_size))
        action = self._sanitize_action(action)

        action_tuple = ActionTuple(continuous=action)
        self._env.set_actions(self.brain_name, action_tuple)
        self._env.step()

        decision_steps, terminal_steps = self._env.get_steps(self.brain_name)

        # Build per-agent results merging decision and terminal steps
        obs_list, rewards, terminated, truncated = self._merge_steps(
            decision_steps, terminal_steps
        )
        self._previous_decision_steps = decision_steps

        info = {"decision_steps": decision_steps, "terminal_steps": terminal_steps}

        if not self._multiagent:
            self.game_over = terminated[0]
            return obs_list[0], rewards[0], terminated[0], truncated[0], info
        else:
            self.game_over = all(terminated)
            return obs_list, rewards, terminated, truncated, info

    def _merge_steps(
        self,
        decision_steps: DecisionSteps,
        terminal_steps: TerminalSteps,
    ) -> Tuple[List[np.ndarray], List[float], List[bool], List[bool]]:
        """Combine decision and terminal steps into per-gym-index arrays."""
        obs_size = self._get_vec_obs_size()
        obs_list = [np.zeros(obs_size, dtype=np.float32)] * self._n_agents
        rewards = [0.0] * self._n_agents
        terminated = [False] * self._n_agents
        truncated = [False] * self._n_agents

        for i, agent_id in enumerate(decision_steps.agent_id):
            gym_idx = self.agent_mapper.get_gym_index(agent_id)
            obs_list[gym_idx] = self._get_agent_obs(decision_steps, i)
            rewards[gym_idx] = float(decision_steps.reward[i])

        for i, agent_id in enumerate(terminal_steps.agent_id):
            gym_idx = self.agent_mapper.get_gym_index(agent_id)
            obs_list[gym_idx] = self._get_agent_obs(terminal_steps, i)
            rewards[gym_idx] = float(terminal_steps.reward[i])
            if terminal_steps.interrupted[i]:
                truncated[gym_idx] = True
            else:
                terminated[gym_idx] = True

        return obs_list, rewards, terminated, truncated

    def _get_agent_obs(self, steps, agent_index: int) -> np.ndarray:
        """Extract flat vector observation for one agent from steps."""
        parts = []
        for obs in steps.obs:
            if len(obs.shape) == 2:
                parts.append(obs[agent_index])
        if parts:
            return np.concatenate(parts, axis=0)
        return np.zeros(self._get_vec_obs_size(), dtype=np.float32)

    def _get_vector_obs(self, steps) -> np.ndarray:
        """Extract vector observations for all agents as a 2D array."""
        result: List[np.ndarray] = []
        for obs in steps.obs:
            if len(obs.shape) == 2:
                result.append(obs)
        return np.concatenate(result, axis=1)

    def _get_vec_obs_size(self) -> int:
        result = 0
        for obs_spec in self.group_spec.observation_specs:
            if len(obs_spec.shape) == 1:
                result += obs_spec.shape[0]
        return result

    def render(self):
        return self.visual_obs

    def close(self) -> None:
        self._env.close()

    def seed(self, seed: Any = None) -> None:
        logger.warning("Could not seed environment %s", self.name)
        return

    def _check_agents(self, n_agents: int) -> None:
        if not self._multiagent and n_agents > 1:
            raise MarathonEnvsException(
                "The environment was launched as a single-agent environment, however "
                "there is more than one agent in the scene."
            )
        if self._n_agents == -1:
            self._n_agents = n_agents
            logger.info("{} agents within environment.".format(n_agents))
        elif self._n_agents != n_agents:
            raise MarathonEnvsException(
                "The number of agents in the environment has changed since "
                "initialization. This is not supported."
            )

    def _sanitize_action(self, action: np.ndarray) -> np.ndarray:
        sanitized = np.zeros_like(action)
        for index, agent_id in enumerate(self._previous_decision_steps.agent_id):
            gym_idx = self.agent_mapper.get_gym_index(agent_id)
            sanitized[index, :] = action[gym_idx, :]
        return sanitized

    @property
    def metadata(self):
        return {"render_modes": ["rgb_array"]}

    @property
    def reward_range(self) -> Tuple[float, float]:
        return -float("inf"), float("inf")

    @property
    def spec(self):
        return None

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def number_agents(self):
        return self._n_agents


class ActionFlattener:
    """
    Flattens branched discrete action spaces into single-branch discrete action spaces.
    """

    def __init__(self, branched_action_space):
        self._action_shape = branched_action_space
        self.action_lookup = self._create_lookup(self._action_shape)
        self.action_space = spaces.Discrete(len(self.action_lookup))

    @classmethod
    def _create_lookup(self, branched_action_space):
        possible_vals = [range(_num) for _num in branched_action_space]
        all_actions = [list(_action) for _action in itertools.product(*possible_vals)]
        action_lookup = {
            _scalar: _action for (_scalar, _action) in enumerate(all_actions)
        }
        return action_lookup

    def lookup_action(self, action):
        return self.action_lookup[action]


class AgentIdIndexMapper:
    def __init__(self) -> None:
        self._agent_id_to_gym_index: Dict[int, int] = {}
        self._done_agents_index_to_last_reward: Dict[int, float] = {}

    def set_initial_agents(self, agent_ids: List[int]) -> None:
        for idx, agent_id in enumerate(agent_ids):
            self._agent_id_to_gym_index[agent_id] = idx

    def mark_agent_done(self, agent_id: int, reward: float) -> None:
        gym_index = self._agent_id_to_gym_index.pop(agent_id)
        self._done_agents_index_to_last_reward[gym_index] = reward

    def register_new_agent_id(self, agent_id: int) -> float:
        free_index, last_reward = self._done_agents_index_to_last_reward.popitem()
        self._agent_id_to_gym_index[agent_id] = free_index
        return last_reward

    def get_id_permutation(self, agent_ids: List[int]) -> List[int]:
        new_agent_ids_to_index = {
            agent_id: idx for idx, agent_id in enumerate(agent_ids)
        }
        new_permutation = [-1] * len(self._agent_id_to_gym_index)
        for agent_id, original_index in self._agent_id_to_gym_index.items():
            new_permutation[original_index] = new_agent_ids_to_index[agent_id]
        return new_permutation

    def get_gym_index(self, agent_id: int) -> int:
        return self._agent_id_to_gym_index[agent_id]
