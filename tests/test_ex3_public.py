"""
Those are tests that will be shared with students
They should test that the code structure/return values
are of correct type/shape
"""

import pytest
import gym
import os.path
import numpy as np

def test_imports_0():
    from rl2023.exercise3 import DQN, Reinforce, ReplayBuffer
    from rl2023.exercise3.train_dqn import CARTPOLE_CONFIG as DQN_CARTPOLE_CONFIG
    from rl2023.exercise3.train_reinforce import ACROBOT_CONFIG as REINF_ACROBOT_CONFIG

def test_config_0():
    from rl2023.exercise3.train_dqn import CARTPOLE_CONFIG
    assert "eval_freq" in CARTPOLE_CONFIG
    assert "eval_episodes" in CARTPOLE_CONFIG
    assert "episode_length" in CARTPOLE_CONFIG
    assert "max_timesteps" in CARTPOLE_CONFIG

    assert "batch_size" in CARTPOLE_CONFIG
    assert "buffer_capacity" in CARTPOLE_CONFIG

def test_config_2():
    from rl2023.exercise3.train_reinforce import ACROBOT_CONFIG
    assert "eval_freq" in ACROBOT_CONFIG
    assert "eval_episodes" in ACROBOT_CONFIG
    assert "episode_length" in ACROBOT_CONFIG
    assert "max_timesteps" in ACROBOT_CONFIG


