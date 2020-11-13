#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks.callbacks import CSVLogger, BaseLogger

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger

# Define environment name:
ENV_NAME = 'LunarLander-v2'

# Create the lunar lander environemnt
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n # Number of actions

env.lander # Displays lander information

# HYPERPARAMETERS
eps = 0.1 # exploration
memory_limit = 1000000 # experience replay memory
gamma = 0.99 # dicount factor
batch_size = 32 
warmup_steps = 1000
tau = 0.01 # tau for soft update or C for hard update
alpha = 0.001 # learning rate
double_dqn = True
dueling_dqn = True # If true, add 'dueling_type'
visualize = True # Training runs much faster without visualizaiton
train_steps = 500000 # Length of training

# Descriptive trial name
trial_name = '2layers_128units_DQN_spyder'

# Load weights (NOTE: IF LOADING WEIGHTS, THE MODEL MUST MATCH THE LOADED WEIGHTS)
load_weights = True
# Save weights
save_weights = False


# Build neural network used in DQN algortihm
# Reference Keras documentation for info
model = Sequential()
model.add(Flatten(input_shape = (1,) + env.observation_space.shape)) # Add 1 for bias
model.add(Dense(128, activation ='relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(nb_actions, activation='linear'))
print(model.summary())

# Compile model using keras-rl
policy = EpsGreedyQPolicy(eps = eps)
memory = SequentialMemory(limit = memory_limit, window_length = 1)
dqn = DQNAgent(model = model, gamma = gamma, batch_size = batch_size, nb_actions = nb_actions, memory = memory, nb_steps_warmup = warmup_steps,
target_model_update = tau, policy = policy, enable_double_dqn = double_dqn, enable_dueling_network = dueling_dqn, dueling_type = 'max') #dueling_type='max'
dqn.compile(Adam(lr = alpha), metrics = ['mse'])

# Load weights (INSERT WEIGHTS PATH)
if load_weights:
	weights_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Weights/Final Weights/2layers_128units_DuelMaxDQN1.hdf5'
	dqn.load_weights(weights_path)
else:
	# Log training at filepath
	# INSERT FILEPATH
	log_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Logs/' + trial_name	+ '.log'
	csv_logger = FileLogger(log_path)

	# TRAIN AGENT
	train_history = dqn.fit(env, nb_steps = train_steps, visualize = visualize, verbose = 2)

# TEST AGENT
test_log = dqn.test(env, nb_episodes = 50, visualize=True)

# Save weights
if save_weights:
	file_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Weights/Final Weights/' + trial_name	+ '.hdf5'
	dqn.save_weights(file_path)
else:
	pass

# Plot episode rewards
train_rewards = train_history.history['episode_reward']
train_episodes = range(len(train_history.history['episode_reward']))
plt.plot(train_episodes, train_rewards, linewidth=0.5)
plt.xlabel('Training Episode')
plt.ylabel('Reward')

# Test rewards
test_rewards = test_log.history['episode_reward']
test_steps = test_log.history['nb_steps']
print('The mean test rewards is {} +/- {}'.format(round(np.mean(test_rewards), 2), round(np.std(test_rewards), 2)))
print('The average number of steps to obtain this reward is {} +/- {}'.format(round(np.mean(test_steps), 2), round(np.std(test_steps), 2)))




