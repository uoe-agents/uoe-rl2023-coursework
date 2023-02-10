# Piazza Post for Results

We understand that having solutions or at least intuition with respect to possible solutions to the coursework can be an important part to learn from the assignment.

We hope that our feedback (see announcement @120 if you missed this) is helpful for you to understand where your implementations might have failed and you could have improved your work. However, most frustration appears to have been the result of performance evaluation and hyperparameter tuning. While we are unable to provide you with our exact solutions, as already outlined in @109, we decided to publish some plots and high-level description of configurations for our solutions.

We have noticed that many students have spent a lot of time manually tuning hyperaparameters. This is a very time consuming task and should never be done manually. A script can be used to run multiple seeds for several combinations of hyperparameters and be left running on a large computing system such as DICE or MLP. This search could be left for as long as required (with no human supervision) and afterwards the best identified combination can be chosen.

Generally speaking, as already mentioned in our tutorial lecture on "Building a complete RL system", there is not as much science to hyperparameter tuning as maybe wished. Especially deep reinforcement learning is quite sensitive to hyperparameters, so it is important to do a wide gridsearch. We suggest to start with a coarse search over the hyperparameter space and get finer over time. Consider the time you (still) have and run this accordingly.


EDIT: on request, we also provide plots for question 2.

## Question 2: Tabular Reinforcement Learning

FrozenLake is an environment of high stochasticity. Therefore, it is essential to do multiple experiments on varying seeds and evaluate over a large number of evaluation episodes. We train Q-learning and Monte Carlo for up to 1M episodes and evaluate these approaches every 1,000 episodes by running 500 episodes of evaluation and recording the mean evaluation performance. This training and evaluation procedure is repeated for 5 seeds and averaged to obtain the following plot. Shading indicates a single standard deviation over training runs (of varying seeds).

[FrozenLake Evaluation Plot](exercise2/frozenlake_results.pdf)

These results were actually obtained with the parameters we provided. However, keep in mind that for Q-learning and Monte Carlo the epsilon decay is crucial just as for DQN. For MC, we observed that a large epsilon (higher than 0.7-0.8) for the majority of training can help to reach this performance. Also, due to the stochasticity of the environment, we suggest to use at least 500 episodes for evaluation to get meaningful results.


## Question 3: Deep Reinforcement Learning

### Cartpole

Cartpole is a fairly simple task. Therefore, it should not be too difficult to solve it with DQN or REINFORCE.

For DQN, a simple one-layer neural network with 32 or 64 hidden units, linear decay in epsilon and a learning rate equal or close to the originally provided (1e-3) one should be sufficient.
Keep in mind that epsilon decay, next to the configuration of hyperparameters, is essential. Epsilon for at least some training episodes at the end should be close (but probably not equal) to 0.

Similarly for REINFORCE, a small one-layer neural network and parameters equal or close to the originally provided ones should be enough to solve Cartpole.

See the following plot for mean evaluation returns of DQN and REINFORCE on Cartpole. Evaluation is done every 5000 training episodes and we train both algorithms on 5 random seeds. Shading corresponds to a single standard deviation of the evaluation returns obtained over all 5 runs.

[Cartpole Evaluation Plot](exercise3/cartpole_results.pdf)

In comparing, keep the update frequency and nature of these algorithms in mind. While the plot might suggest that DQN converges much quicker to REINFORCE, this is in fact not true for our implementation in terms of time. DQN as a off-policy algorithm, which updates every timestep (once enough samples are collected in the replay buffer), runs much slower but more sample efficient than the on-policy REINFORCE algorithm updating once after each episode.

### LunarLander

LunarLander is a considerably more difficult task and requires more precise tuning of hyperparameters. Even small changes of the learning rate and epsilon decay might lead to drastically improved performance and therefore finer hyperparemeter search is required.

We found that DQN was able to solve LunarLander consistently with a two-layer network with 64x64 hidden units. Epsilon decay and learning rate was kept similar to Cartpole, but much more running time and timesteps are needed for training.

REINFORCE is difficult to tune to solve LunarLander. We ended up solving it using a two-layer policy network with 32x32 hidden units and a learning rate close to the originally provided one.

See the following plot for mean evaluation returns of DQN and REINFORCE on Cartpole. Evaluation is and plotting is done in the same way: Evaluation is done every 5000 training episodes and we train both algorithms on 5 random seeds. Shading corresponds to a single standard deviation of the evaluation returns obtained over all 5 runs.

[DQN LunarLander Evaluation Plot](exercise3/lunarlander_results_dqn.pdf)

[REINFORCE LunarLander Evaluation Plot](exercise3/lunarlander_results_reinforce.pdf)


## Question 4: n-step Actor Critic

### Cartpole

Similar to DQN and REINFORCE, it should be fairly simple to solve Cartpole with a correct implementation of 1-step and 10-step AC.

We ended up solving Cartpole with the same hyperparameters for 1-step and 10-step AC using a single-layer network of 32 or 64 hidden units for policy and critic. A default learning rate of around 1e-3 should be fitting to solve the task, but we noticed that many configurations should be able to solve Cartpole despite training time varying.

[Cartpole Evaluation Plot](exercise4/cartpole_results.pdf)


### LunarLander

LunarLander again is much more difficult to solve. We found that solving LunarLander becomes significantly simpler by decoupling network size and learning rate of the policy and critic networks.

We use small two-layer networks for the policy of 16x32 hidden units and apply a larger critic network of 64x64 hidden units. Similarly, we found that a slightly larger learning rate for the critic network was helpful for more stable training. Policy learning rate was around 1e-4 while the critic was optimised using a learning rate close to 1e-3. Again, the same hyperparameters were used to solve LunarLander for both 1-step and 10-step AC.

See the following plot for the final evaluation returns.

[LunarLander Evaluation Plot](exercise4/lunarlander_results.pdf)
