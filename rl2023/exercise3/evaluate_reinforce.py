import gym
import pickle
from typing import List, Tuple

from rl2023.exercise3.agents import Reinforce
from rl2023.exercise3.train_reinforce import play_episode, ACROBOT_CONFIG, SWEEP_RESULTS_FILE_ACROBOT
from rl2023.util.result_processing import Run, get_best_saved_run

ENV = "ACROBOT"
RENDER = True
SWEEP_DIR = "/home/..." #Path to sweep results directory
SWEEP_RESULTS_FILE = SWEEP_DIR + SWEEP_RESULTS_FILE_ACROBOT


def evaluate(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = Reinforce(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    agent.restore(config['save_filename'])

    eval_returns_all = []
    eval_times_all = []


    eval_returns = 0
    for _ in range(config["eval_episodes"]):
        _, episode_return, _ = play_episode(
            env,
            agent,
            train=False,
            explore=False,
            render=RENDER,
            max_steps=config["episode_length"],
        )
        eval_returns += episode_return / config["eval_episodes"]

    return eval_returns


if __name__ == "__main__":
    if ENV == "ACROBOT":
        CONFIG = ACROBOT_CONFIG
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_ACROBOT
    else:
        raise(ValueError(f"Unknown environment {ENV}"))

    env = gym.make(CONFIG["env"])
    results = pickle.load(open(SWEEP_RESULTS_FILE, 'rb'))
    best_run, best_run_filename = get_best_saved_run(results)
    print(f"Best run was {best_run_filename}")
    CONFIG.update(best_run.config)
    CONFIG['save_filename'] = SWEEP_DIR + best_run_filename
    returns = evaluate(env, CONFIG)
    print(returns)
    env.close()
