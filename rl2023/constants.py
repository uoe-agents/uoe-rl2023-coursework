EX1_CONSTANTS = {
    "gamma": 0.8,
}

EX2_CONSTANTS = {
    "env": "Taxi-v3",
    "eps_max_steps": 50,
    "eval_episodes": 500,
    "eval_eps_max_steps": 100,
}

EX2_MC_CONSTANTS = EX2_CONSTANTS.copy()
EX2_MC_CONSTANTS["total_eps"] = 100000

EX2_QL_CONSTANTS = EX2_CONSTANTS.copy()
EX2_QL_CONSTANTS["total_eps"] = 10000

EX3_CARTPOLE_CONSTANTS = {
    "env": "CartPole-v1",
    "gamma": 0.99,
    "episode_length": 400,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

EX3_DQN_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_DQN_CARTPOLE_CONSTANTS["max_timesteps"] = 25000
EX3_DQN_CARTPOLE_CONSTANTS["algo"] = "DQN"

EX3_REINFORCE_CARTPOLE_CONSTANTS = EX3_CARTPOLE_CONSTANTS.copy()
EX3_REINFORCE_CARTPOLE_CONSTANTS["max_timesteps"] = 200000
EX3_REINFORCE_CARTPOLE_CONSTANTS["algo"] = "Reinforce"

EX3_ACROBOT_CONSTANTS = {
    "env": "Acrobot-v1",
    "gamma": 1.0,
    "episode_length": 1000,
    "max_time": 30 * 60,
    "save_filename": None,
    "algo": None,
}

EX3_DQN_ACROBOT_CONSTANTS = EX3_ACROBOT_CONSTANTS.copy()
EX3_DQN_ACROBOT_CONSTANTS["max_timesteps"] = 100000
EX3_DQN_ACROBOT_CONSTANTS["algo"] = "DQN"

EX3_REINFORCE_ACROBOT_CONSTANTS = EX3_ACROBOT_CONSTANTS.copy()
EX3_REINFORCE_ACROBOT_CONSTANTS["max_timesteps"] = 700000
EX3_REINFORCE_ACROBOT_CONSTANTS["algo"] = "Reinforce"

EX4_PENDULUM_CONSTANTS = {
    "env": "Pendulum-v1",
    "target_return": -300.0,
    "episode_length": 200,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "save_filename": "pendulum_latest.pt",
    "algo": "DDPG",
}

EX4_BIPEDAL_CONSTANTS = {
    "env": "BipedalWalker-v3",
    "eval_freq": 20000,
    "eval_episodes": 100,
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "save_filename": "bipedal_q4_latest.pt",
    "algo": "DDPG",
}

EX5_BIPEDAL_CONSTANTS = {
    "env": "BipedalWalker-v3",
    "eval_freq": 20000,
    "eval_episodes": 100,
    "target_return": 300.0,
    "episode_length": 1600,
    "max_timesteps": 400000,
    "max_time": 120 * 60,
    "save_filename": "bipedal_q5_latest.pt",
    "algo": "DDPG",
}
