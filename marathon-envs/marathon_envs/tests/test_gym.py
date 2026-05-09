from unittest import mock
import pytest
import numpy as np

from gymnasium import spaces
from marathon_envs.envs import MarathonEnvs, MarathonEnvsException, AgentIdIndexMapper


def make_behavior_spec(obs_size=3, continuous_actions=2):
    """Build a minimal mock BehaviorSpec."""
    action_spec = mock.MagicMock()
    action_spec.is_discrete.return_value = False
    action_spec.continuous_size = continuous_actions

    obs_spec = mock.MagicMock()
    obs_spec.shape = (obs_size,)

    spec = mock.MagicMock()
    spec.action_spec = action_spec
    spec.observation_specs = [obs_spec]
    return spec


def make_decision_steps(num_agents=1, obs_size=3, agent_ids=None):
    """Build a mock DecisionSteps."""
    if agent_ids is None:
        agent_ids = list(range(num_agents))
    steps = mock.MagicMock()
    steps.obs = [np.ones((num_agents, obs_size), dtype=np.float32)]
    steps.reward = np.ones(num_agents, dtype=np.float32)
    steps.agent_id = np.array(agent_ids)
    steps.__len__ = mock.MagicMock(return_value=num_agents)
    return steps


def make_terminal_steps(num_agents=0, obs_size=3, interrupted=None):
    """Build a mock TerminalSteps."""
    steps = mock.MagicMock()
    steps.obs = [np.zeros((num_agents, obs_size), dtype=np.float32)]
    steps.reward = np.zeros(num_agents, dtype=np.float32)
    steps.agent_id = np.array([], dtype=np.int64)
    steps.interrupted = np.array([], dtype=bool) if interrupted is None else interrupted
    steps.__len__ = mock.MagicMock(return_value=num_agents)
    return steps


def setup_mock_env(mock_unity_env, spec, decision_steps, terminal_steps=None):
    """Wire a mock UnityEnvironment for MarathonEnvs use."""
    if terminal_steps is None:
        terminal_steps = make_terminal_steps()
    instance = mock_unity_env.return_value
    instance.behavior_specs = {"MockBrain": spec}
    instance.get_steps.return_value = (decision_steps, terminal_steps)
    instance.reset.return_value = None
    instance.step.return_value = None
    instance.set_actions.return_value = None
    instance.close.return_value = None
    return instance


@mock.patch("marathon_envs.envs.UnityEnvironment")
def test_gym_wrapper_reset(mock_unity_env):
    spec = make_behavior_spec()
    decision_steps = make_decision_steps(num_agents=1)
    setup_mock_env(mock_unity_env, spec, decision_steps)

    env = MarathonEnvs.__new__(MarathonEnvs)
    env._env = mock_unity_env.return_value
    env.brain_name = "MockBrain"
    env.group_spec = spec
    env._multiagent = False
    env._n_agents = 1
    env.visual_obs = None
    env.game_over = False
    env._previous_decision_steps = decision_steps
    env.agent_mapper = AgentIdIndexMapper()
    env.agent_mapper.set_initial_agents([0])
    env.render_mode = None

    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert isinstance(info, dict)


@mock.patch("marathon_envs.envs.UnityEnvironment")
def test_gym_wrapper_step(mock_unity_env):
    spec = make_behavior_spec()
    decision_steps = make_decision_steps(num_agents=1)
    terminal_steps = make_terminal_steps()
    setup_mock_env(mock_unity_env, spec, decision_steps, terminal_steps)

    env = MarathonEnvs.__new__(MarathonEnvs)
    env._env = mock_unity_env.return_value
    env.brain_name = "MockBrain"
    env.group_spec = spec
    env._multiagent = False
    env._n_agents = 1
    env.visual_obs = None
    env.game_over = False
    env._previous_decision_steps = decision_steps
    env.agent_mapper = AgentIdIndexMapper()
    env.agent_mapper.set_initial_agents([0])
    env.render_mode = None
    env._action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
    env._observation_space = spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float32)

    action = env._action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


@mock.patch("marathon_envs.envs.UnityEnvironment")
def test_multi_agent_step(mock_unity_env):
    n = 2
    spec = make_behavior_spec()
    decision_steps = make_decision_steps(num_agents=n)
    terminal_steps = make_terminal_steps()
    setup_mock_env(mock_unity_env, spec, decision_steps, terminal_steps)

    env = MarathonEnvs.__new__(MarathonEnvs)
    env._env = mock_unity_env.return_value
    env.brain_name = "MockBrain"
    env.group_spec = spec
    env._multiagent = True
    env._n_agents = n
    env.visual_obs = None
    env.game_over = False
    env._previous_decision_steps = decision_steps
    env.agent_mapper = AgentIdIndexMapper()
    env.agent_mapper.set_initial_agents(list(range(n)))
    env.render_mode = None
    env._action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)
    env._observation_space = spaces.Box(-np.full(3, np.inf), np.full(3, np.inf), dtype=np.float32)

    actions = np.ones((n, 2), dtype=np.float32)
    obs_list, rewards, terminated, truncated, info = env.step(actions)
    assert isinstance(obs_list, list)
    assert isinstance(rewards, list)
    assert isinstance(terminated, list)
    assert isinstance(truncated, list)
    assert isinstance(info, dict)
    assert len(obs_list) == n
    assert len(rewards) == n


def test_agent_id_index_mapper():
    mapper = AgentIdIndexMapper()
    initial_agent_ids = [1001, 1002, 1003, 1004]
    mapper.set_initial_agents(initial_agent_ids)

    # Mark some agents as done
    mapper.mark_agent_done(1001, 42.0)
    mapper.mark_agent_done(1004, 1337.0)

    # Register new agents
    old_reward1 = mapper.register_new_agent_id(2001)
    old_reward2 = mapper.register_new_agent_id(2002)

    assert {old_reward1, old_reward2} == {42.0, 1337.0}

    new_agent_ids = [1002, 1003, 2001, 2002]
    permutation = mapper.get_id_permutation(new_agent_ids)
    assert set(permutation) == set(range(4))

    permuted_ids = [new_agent_ids[i] for i in permutation]
    for idx, agent_id in enumerate(initial_agent_ids):
        if agent_id in permuted_ids:
            assert permuted_ids[idx] == agent_id


def test_get_vec_obs_size():
    env = MarathonEnvs.__new__(MarathonEnvs)
    env.group_spec = make_behavior_spec(obs_size=5)
    assert env._get_vec_obs_size() == 5


def test_vec_obs_size_ignores_visual():
    """Visual observation specs (shape len > 1) should not count toward vec obs size."""
    env = MarathonEnvs.__new__(MarathonEnvs)
    spec = mock.MagicMock()
    obs_spec_vec = mock.MagicMock()
    obs_spec_vec.shape = (8,)
    obs_spec_vis = mock.MagicMock()
    obs_spec_vis.shape = (84, 84, 3)
    spec.observation_specs = [obs_spec_vec, obs_spec_vis]
    env.group_spec = spec
    assert env._get_vec_obs_size() == 8
