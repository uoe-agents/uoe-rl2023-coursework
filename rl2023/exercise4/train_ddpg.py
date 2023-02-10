import copy
import pickle
from collections import defaultdict

import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

from rl2023.constants import EX4_PENDULUM_CONSTANTS as PENDULUM_CONSTANTS
from rl2023.constants import EX4_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from rl2023.exercise4.agents import DDPG
from rl2023.exercise3.replay import ReplayBuffer
from rl2023.util.hparam_sweeping import generate_hparam_configs
from rl2023.util.result_processing import Run

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "PENDULUM" #"PENDULUM" OR "BIPEDAL"

PENDULUM_CONFIG = {
    "eval_freq": 2000,
    "eval_episodes": 3,
    "policy_learning_rate": 1e-3,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
    "tau": 0.01,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
}
PENDULUM_CONFIG.update(PENDULUM_CONSTANTS)

BIPEDAL_CONFIG = {
    "critic_hidden_size": [64, 64],
    "policy_hidden_size": [64, 64],
}
BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

### INCLUDE YOUR CHOICE OF HYPERPARAMETERS HERE ###
BIPEDAL_HPARAMS = {
    "critic_hidden_size": ...,
    "policy_hidden_size": ...,
    }

SWEEP_RESULTS_FILE_BIPEDAL = "DDPG-Bipedal-sweep-results-ex4.pkl"


def play_episode(
        env,
        agent,
        replay_buffer,
        train=True,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):

    ep_data = defaultdict(list)
    obs = env.reset()
    done = False
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                new_data = agent.update(batch)
                for k, v in new_data.items():
                    ep_data[k].append(v)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return, ep_data


def train(env: gym.Env, config, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DDPG(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break

            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, ep_return, ep_data = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    # pbar.write(f"Epsilon = {agent.epsilon}")
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)
                if eval_returns >= config["target_return"]:
                    pbar.write(
                        f"Reached return {eval_returns} >= target return of {config['target_return']}"
                    )
                    break

    if config["save_filename"]:
        print("Saving to: ", agent.save(config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":
    if ENV == "PENDULUM":
        CONFIG = PENDULUM_CONFIG
        HPARAMS_SWEEP = None # Not required for assignment
        SWEEP_RESULTS_FILE = None # Not required for assignment
    elif ENV == "BIPEDAL":
        CONFIG = BIPEDAL_CONFIG
        HPARAMS_SWEEP = BIPEDAL_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_BIPEDAL
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

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
                print(f"\nTraining iteration: {i+1}/{NUM_SEEDS_SWEEP}")
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
