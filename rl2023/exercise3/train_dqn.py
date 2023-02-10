import copy
import pickle

import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple, Dict
from collections import defaultdict
import matplotlib.pyplot as plt

from rl2023.constants import EX3_DQN_CARTPOLE_CONSTANTS as CARTPOLE_CONSTANTS
from rl2023.constants import EX3_DQN_ACROBOT_CONSTANTS as ACROBOT_CONSTANTS
from rl2023.exercise3.agents import DQN
from rl2023.exercise3.replay import ReplayBuffer
from rl2023.util.hparam_sweeping import generate_hparam_configs
from rl2023.util.result_processing import Run

RENDER = False # FALSE FOR FASTER TRAINING / TRUE TO VISUALIZE ENVIRONMENT DURING EVALUATION
SWEEP = False # TRUE TO SWEEP OVER POSSIBLE HYPERPARAMETER CONFIGURATIONS
NUM_SEEDS_SWEEP = 10 # NUMBER OF SEEDS TO USE FOR EACH HYPERPARAMETER CONFIGURATION
SWEEP_SAVE_RESULTS = True # TRUE TO SAVE SWEEP RESULTS TO A FILE
SWEEP_SAVE_ALL_WEIGTHS = False # TRUE TO SAVE ALL WEIGHTS FROM EACH SEED
ENV = "CARTPOLE" # "ACROBOT" is also possible if you uncomment the corresponding code, but is not assessed for DQN.


### ASSIGNMENT: CHANGE epsilon_decay_strategy: "constant" TO "linear" OR "exponential" TO ANSWER QUESTIONS 3.2 TO 3.6 IN answer_sheet.py ###
CARTPOLE_CONFIG = {
    "eval_freq": 5000, # HOW OFTEN WE EVALUATE (AND RENDER IF RENDER=TRUE)
    "eval_episodes": 100,  # DECREASING THIS MIGHT REDUCE EVALUATION ACCURACY; BUT MAKES IT EASIER TO SEE HOW THE POLICY EVOLVES OVER TIME (BY ENABLING RENDER ABOVE)
    "learning_rate": 2.3e-3,
    "hidden_size": (256, 256),
    "target_update_freq": 10,
    "batch_size": 64,
    "epsilon_decay_strategy": "constant", #"constant" or "linear" or "exponential"
    "epsilon_start": .25,
    "epsilon_min": 0.04, #only used in linear and exponential decay strategies
    "epsilon_decay": None, #For exponential epsilon decay
    "exploration_fraction": None, # For linear epsilon decay, fraction of training time at which epsilon=epsilon_min
    "buffer_capacity": int(1e6),
    "plot_loss": False, # SET TRUE FOR 3.3 (Understanding the Loss)
}
CARTPOLE_CONFIG.update(CARTPOLE_CONSTANTS)

CARTPOLE_HPARAMS_LINEAR_DECAY = {
    "epsilon_start": [1.0,],
    "exploration_fraction" : [.75, .25, .01],
    }

CARTPOLE_HPARAMS_EXP_DECAY = {
    "epsilon_start": [1.0, ],
    "epsilon_decay"     : [1.0, 0.75, 0.001]
    }

if CARTPOLE_CONFIG['epsilon_decay_strategy'] == "linear":
    CARTPOLE_HPARAMS = CARTPOLE_HPARAMS_LINEAR_DECAY
elif CARTPOLE_CONFIG['epsilon_decay_strategy'] == "exponential":
    CARTPOLE_HPARAMS = CARTPOLE_HPARAMS_EXP_DECAY
else:
    CARTPOLE_HPARAMS = None

SWEEP_RESULTS_FILE_CARTPOLE = f"DQN-Cartpole-sweep-decay-{CARTPOLE_CONFIG['epsilon_decay_strategy']}-results.pkl"

ACROBOT_CONFIG = {
    "eval_freq": 10000,
    "eval_episodes": 100,
    "learning_rate": 6e-3,
    "hidden_size": (32,),
    "target_update_freq": 50,
    "batch_size": 128,
    "epsilon_decay_strategy": "constant",
    "epsilon_start": 0.5,
    "epsilon_min": 0.5,
    "epsilon_decay"       : None,
    "exploration_fraction": None,
    "buffer_capacity": int(5e4),
    "plot_loss"      : False,
}

ACROBOT_CONFIG.update(ACROBOT_CONSTANTS)

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
                np.array([action], dtype=np.float32),
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
    Execute training of DQN on given environment using the provided configuration
      
    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]): average eval returns during training, evaluation
            timesteps, compute times at evaluation and a dictionary containing other training metrics specific to DQN
    """
    timesteps_elapsed = 0

    agent = DQN(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_timesteps_all = []
    eval_times_all = []

    start_time = time.time()
    run_data = defaultdict(list)
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
                if config["env"] == "CartPole-v1" or config["env"] == "Acrobot-v1":
                    max_steps = config["episode_length"]
                else:
                    raise ValueError(f"Unknown environment {config['env']}")

                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=max_steps,
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    pbar.write(f"Epsilon = {agent.epsilon}")
                eval_returns_all.append(eval_returns)
                eval_timesteps_all.append(timesteps_elapsed)
                eval_times_all.append(time.time() - start_time)

        # you may add logging of additional metrics here
        run_data["train_timesteps"] = (config["batch_size"] * np.arange(1, len(run_data["q_loss"]) + 1)).tolist()
        run_data["train_episodes"] = np.arange(1, len(run_data["train_ep_returns"]) + 1).tolist()
        
    if config["save_filename"]:
        print("\nSaving to: ", agent.save(config["save_filename"]))

    if config["plot_loss"]:
        print("Plotting DQN loss...")
        plt.plot(run_data["train_timesteps"], run_data["q_loss"], "-", alpha=0.7)
        plt.xlabel("Timesteps", fontsize=30)
        plt.ylabel("DQN Loss", fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout(pad=0.3)

        plt.show()

    return np.array(eval_returns_all), np.array(eval_timesteps_all), np.array(eval_times_all), run_data


if __name__ == "__main__":

    if ENV == "CARTPOLE":
        CONFIG = CARTPOLE_CONFIG
        HPARAMS_SWEEP = CARTPOLE_HPARAMS
        SWEEP_RESULTS_FILE = SWEEP_RESULTS_FILE_CARTPOLE
    elif ENV == "ACROBOT":
        CONFIG = ACROBOT_CONFIG
        HPARAMS_SWEEP = None
        SWEEP_RESULTS_FILE = None
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
