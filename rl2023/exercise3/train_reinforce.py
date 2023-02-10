import copy
import pickle
from collections import defaultdict

import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict

from rl2023.constants import EX3_REINFORCE_ACROBOT_CONSTANTS as ACROBOT_CONSTANTS
from rl2023.exercise3.agents import Reinforce
from rl2023.util.hparam_sweeping import generate_hparam_configs
from rl2023.util.result_processing import Run

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "ACROBOT"

ACROBOT_CONFIG = {
    "eval_freq": 70000,
    "eval_episodes": 100,
    "hidden_size": (32,),
    "learning_rate": 1e-3,
}

ACROBOT_CONFIG.update(ACROBOT_CONSTANTS)

ACROBOT_HPARAMS = {
    "learning_rate": [6e-1, 6e-2, 6e-3],
    }

SWEEP_RESULTS_FILE_ACROBOT = "Reinforce-Acrobot-sweep-results.pkl"


def play_episode(
    env: gym.Env,
    agent: Reinforce,
    train: bool = True,
    explore=True,
    render=False,
    max_steps=200,
) -> Tuple[int, float, Dict]:
    """
    Play one episode and train reinforce algorithm

    :param env (gym.Env): gym environment
    :param agent (Reinforce): Reinforce agent
    :param train (bool): flag whether training should be executed
    :param explore (bool): flag whether exploration is used
    :param render (bool): flag whether environment should be visualised
    :param max_steps (int): max number of timesteps for the episode
    :return (Tuple[int, float]): total number of executed steps and received reward
    """

    ep_data = defaultdict(list)
    obs = env.reset()

    if render:
        env.render()

    done = False
    num_steps = 0
    episode_return = 0

    observations = []
    actions = []
    rewards = []

    while not done and num_steps < max_steps:
        action = agent.act(np.array(obs), explore=explore)
        nobs, rew, done, _ = env.step(action)

        observations.append(obs)
        actions.append(action)
        rewards.append(rew)

        if render:
            env.render()

        num_steps += 1
        episode_return += rew

        obs = nobs

    if train:
        new_data = agent.update(rewards, observations, actions)
        for k, v in new_data.items():
            ep_data[k].append(v)

    return num_steps, episode_return, ep_data


def train(env: gym.Env, config, output: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Execute training of REINFORCE on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[np.ndarray, np.ndarray, np.ndarray]): average eval returns during training, evaluation
            timesteps and compute times at evaluation
    """
    timesteps_elapsed = 0

    agent = Reinforce(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )

    total_steps = config["max_timesteps"]
    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []
    run_data = defaultdict(list)

    start_time = time.time()
    with tqdm(total=total_steps) as pbar:
        while timesteps_elapsed < total_steps:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, total_steps)
            num_steps, ep_return, ep_data = play_episode(
                env,
                agent,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
            )
            timesteps_elapsed += num_steps
            pbar.update(num_steps)
            for k, v in ep_data.items():
                run_data[k].extend(v)
            run_data["train_ep_returns"].append(ep_return)

            if timesteps_elapsed % config["eval_freq"] < num_steps:
                eval_return = 0
                if config["env"] == "CartPole-v1" or config["env"] == "Acrobot-v1":
                    max_steps = config["episode_length"]
                else:
                    raise ValueError(f"Unknown environment {config['env']}")

                for _ in range(config["eval_episodes"]):
                    _, total_reward, _ = play_episode(
                        env,
                        agent,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                    )
                    eval_return += total_reward / (config["eval_episodes"])
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean return of {eval_return}"
                    )
                eval_returns_all.append(eval_return)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

    if config["save_filename"]:
        print("\nSaving to: ", agent.save(config["save_filename"]))

    # you may add logging of additional metrics here
    run_data["train_episodes"] = np.arange(1, len(run_data["train_ep_returns"]) + 1).tolist()

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":

    if ENV == "ACROBOT":
        CONFIG = ACROBOT_CONFIG
        HPARAMS_SWEEP = ACROBOT_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_ACROBOT
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])

    if SWEEP:
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
            with open(SWEEP_RESULTS_FILE, 'wb') as f:
                pickle.dump(results, f)
    else:
        _ = train(env, CONFIG)
    env.close()
