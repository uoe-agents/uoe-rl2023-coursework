import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class Run:

    def __init__(self, config: Dict):
        self._config = config
        self._run_name = None

        self._final_returns = []
        self._train_times = []
        self._run_data = []
        self._agent_weights_filenames = []

        self._run_ids = []
        self._all_eval_timesteps = []
        self._all_returns = []

    def update(self, eval_returns, eval_timesteps, times=None, run_data=None):

        self._run_ids.append(len(self._run_ids))
        if self._config['save_filename'] is not None:
            self._agent_weights_filenames.append(self._config['save_filename'])
            self._config['save_filename'] = None

        self._all_eval_timesteps.append(eval_timesteps)
        self._all_returns.append(eval_returns)
        self._final_returns.append(eval_returns[-1])
        if times is not None:
            self._train_times.append(times[-1])
        if run_data is not None:
            self._run_data.append(run_data)

    def set_save_filename(self, filename):
        if self._config["save_filename"] is not None:
            print(f"Warning: Save filename already set in config. Overwriting to {filename}.")

        self._config['save_filename'] = f"{filename}.pt"

    @property
    def run_name(self):
        return self._run_name

    @run_name.setter
    def run_name(self, name):
        self._run_name = name

    @property
    def final_return_mean(self) -> float:
        final_returns = np.array(self._final_returns)
        return final_returns.mean()

    @property
    def final_return_ste(self) -> float:
        final_returns = np.array(self._final_returns)
        return np.std(final_returns, ddof=1) / np.sqrt(np.size(final_returns))

    @property
    def final_return_iqm(self) -> float:
        final_returns = np.array(self.final_returns)
        q1 = np.percentile(final_returns, 25)
        q3 = np.percentile(final_returns, 75)
        trimmed_ids = np.nonzero(np.logical_and(final_returns >= q1, final_returns <= q3))
        trimmed_returns = final_returns[trimmed_ids]
        return trimmed_returns.mean()

    @property
    def final_returns(self) -> np.ndarray:
        return np.array(self._final_returns)

    @property
    def train_times(self) -> np.ndarray:
        return np.array(self._train_times)

    @property
    def config(self):
        return self._config

    @property
    def run_ids(self) -> List[int]:
        return self._run_ids

    @property
    def agent_weights_filenames(self) -> List[str]:
        return self._agent_weights_filenames

    @property
    def run_data(self) -> List[Dict]:
        return self._run_data

    @property
    def all_eval_timesteps(self) -> np.ndarray:
        return np.array(self._all_eval_timesteps)

    @property
    def all_returns(self) -> np.ndarray:
        return np.array(self._all_returns)


# The helper functions below are provided to help you process the results of your runs.

def rank_runs(runs: List[Run]):
    """Sorts runs by mean final return, highest to lowest."""

    return sorted(runs, key=lambda x: x.final_return_mean, reverse=True)


def get_best_saved_run(runs:List[Run]) -> Tuple[Run, str]:
    """Returns the run with the highest mean final return and the filename of the saved weights of its highest scoring
    seed, if it exists."""

    ranked_runs = rank_runs(runs)
    best_run = ranked_runs[0]

    if best_run.agent_weights_filenames:
        best_run_id = np.argmax(best_run.final_returns)
        return best_run, best_run.agent_weights_filenames[best_run_id]
    else:
        raise ValueError(f"No saved runs found for highest mean final returns run {best_run.run_name}.")