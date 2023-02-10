import copy
import pickle
from collections import defaultdict

import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2023.constants import EX5_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from rl2023.exercise4.agents import DDPG
from rl2023.exercise4.train_ddpg import train
from rl2023.exercise3.replay import ReplayBuffer
from rl2023.util.hparam_sweeping import generate_hparam_configs
from rl2023.util.result_processing import Run

RENDER = False
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "BIPEDAL" # "ACROBOT" is also possible if you uncomment the corresponding code, but is not assessed for DQN.

# IN EXERCISE 5 YOU SHOULD TUNE PARAMETERS IN THIS CONFIG ONLY
BIPEDAL_CONFIG = {
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "gamma": 0.99,
    "tau": 0.2,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
}
BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

### INCLUDE YOUR CHOICE OF HYPERPARAMETERS HERE ###
BIPEDAL_HPARAMS = {...}

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Bipedal-sweep-results-ex5.pkl"

if __name__ == "__main__":
    if ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
        HPARAMS_SWEEP = BIPEDAL_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise (ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])

    if SWEEP and HPARAMS_SWEEP is not None:
        config_list, swept_params = generate_hparam_configs(CONFIG, HPARAMS_SWEEP)
        results = []
        for config in config_list:
            run = Run(config)
            hparams_values = '_'.join([':'.join([key, str(config[key])]) for key in swept_params])
            run.run_name = hparams_values
            print(f"\nStarting new run...")
            for i in range(NUM_SEEDS_SWEEP):
                print(f"\nTraining iteration: {i + 1}/{NUM_SEEDS_SWEEP}")
                run_save_filename = '--'.join([run.config["algo"], run.config["env"], hparams_values, str(i)])
                if SWEEP_SAVE_ALL_WEIGTHS:
                    run.set_save_filename(run_save_filename)
                eval_returns, eval_timesteps, times, run_data = train(env, run.config, output=False)
                run.update(eval_returns, eval_timesteps, times, run_data)
            results.append(copy.deepcopy(run))
            print(f"Finished run with hyperparameters {hparams_values}. "
                  f"Mean final score: {run.final_return_mean} +- {run.final_return_ste}")

        if SWEEP_SAVE_RESULTS:
            print(f"Saving results to {SWEEP_RESULTS_FILE}")
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)

    else:
        _ = train(env, CONFIG)

    env.close()
