# dqn-lunarlander
Implementation of DQN in OpenAI Gym LunarLander-v2 discrete environment.

Sam Weinberg (sam.weinberg@mail.utoronto.ca)

This file contains information on how to run my implementation of DQN in the LunarLander-v2 environment.

## 1. Setup

The set-up for this project is tedious. First, we require the installation of OpenAI Gym's Box2D environments. These environments have a MuJoCo dependancy that makes it slightly more difficult to install. The environemnt is designed for Linux, however I have managed to get it working on Windows. For installation on Windows, follow the instructions at this link exactly:

https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5

In addition to this, we will be using out-of-the-box RL packages including Keras, Tensorflow, and keras-rl. It is important to use
a version of Pyhton that is compatible across all of the packages. This is the configuration that I used:

Python - 3.6
Tensorflow - 1.14.0
Keras - 2.3.1
keras-rl - 0.4.2

## 2. Files

There are two files

	- training.py 
	
	This file is used to train and test new models or previously trained models.
	
	- plotting.py

	This file is used to plot the data from log files.

## 3. Training

There are two options: a) train a new model or b) load model weights. 

a) To train a model, insert hyperparamters and specify network architecture. A descriptive trial name should be used to save the training logs and model weights. Change the file paths to the corresponding location on your computer. To run DDQN or Dueling DQN make the corresponding hyperparamters True. For dueling, an additional paramter called 'dueling_type' must be specified. To save weights, spcify the save weights paramter to True. Please refer to keras-rl and Keras documentation for more details.

b) To load in model weights, specifiy the load_weights paramter to True. Specify the path to the weights in which you would like to load. The final weights used for each algortihm have been included in the folder 'Final Weights'. Note that the model architecture and the loaded weights MUST match.

## 4. Plotting

The plotting file plots the logs of the training data. They require one log for each of the four algorithm types. Change the file locations to the location on your computer. The final training logs can be found in the 'Logs' folder.
