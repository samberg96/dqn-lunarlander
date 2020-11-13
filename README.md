# dqn-lunarlander
Implementation of DQN in OpenAI Gym LunarLander-v2 discrete environment.

Sam Weinberg (sam.weinberg@mail.utoronto.ca)

This file contains information on my implementation of DQN in the LunarLander-v2 environment. Refer to the report (LunarLander_Report.pdf) for further readings.

## 1. Environment
This image shows the LunarLander-v2 environment and control inputs.
![Env](/img/LunarLanderEnv.JPG)

## 2. DQN Model
This image demonstrates the neural network used in the implementation of DQN. The 8D state is input to the network, which approximates a Q-Value for each of the four actions.
![Model](/img/neuralnetwork.JPG)

## 3. Setup

First, we require the installation of OpenAI Gym's Box2D environments. These environments have a MuJoCo dependancy that makes it slightly more difficult to install. The environemnt is designed for Linux, however I have managed to get it working on Windows. For installation on Windows, follow the instructions at this link exactly:

https://medium.com/@sayanmndl21/install-openai-gym-with-box2d-and-mujoco-in-windows-10-e25ee9b5c1d5

In addition to this, we will be using out-of-the-box RL packages including Keras, Tensorflow, and keras-rl. It is important to use
a version of Pyhton that is compatible across all of the packages. This is the configuration that I used:

Python - 3.6

Tensorflow - 1.14.0

Keras - 2.3.1

keras-rl - 0.4.2

## 4. Files

There are two files

	- training.py 
	
	This file is used to train and test new models or previously trained models.
	
	- plotting.py

	This file is used to plot the data from log files.

## 5. Training

There are two options: a) train a new model or b) load model weights. 

a) To train a model, insert hyperparamters and specify network architecture. A descriptive trial name should be used to save the training logs and model weights. Change the file paths to the corresponding location on your computer. To run DDQN or Dueling DQN make the corresponding hyperparamters True. For dueling, an additional paramter called 'dueling_type' must be specified. To save weights, spcify the save weights paramter to True. Please refer to keras-rl and Keras documentation for more details.

b) To load in model weights, specifiy the load_weights paramter to True. Specify the path to the weights in which you would like to load. The final weights used for each algortihm have been included in the folder 'Final Weights'. Note that the model architecture and the loaded weights MUST match.

## 6. Plotting

The plotting file plots the logs of the training data. They require one log for each of the four algorithm types. Change the file locations to the location on your computer. The final training logs can be found in the 'Logs' folder.
