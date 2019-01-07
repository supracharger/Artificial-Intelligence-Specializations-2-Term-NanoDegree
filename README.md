[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project includes a trained agent to collect yellow bananas and avoid blue bananas in a large, square world.  

![Trained Agent][image1]

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana.  Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.  

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

**REQUIREMENT(when the environment is considered solved):** Agent must get an average score of +13 over 100 consecutive episodes.
**AGENT ACHIEVED: A Max Average Score of 14.52** over 100 consecutive episodes.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux_NoVis.zip) to obtain the environment.

2. Place the file in the DRLND GitHub repository, in the `p1_navigation/` folder, and unzip (or decompress) the file. 

### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!  

## Dependencies
* Numpy
* unityagents import UnityEnvironment
* random
* collections
* Torch
* pickle
* matplotlib
Use "pip install --yourPackage--" to install dependencies.

## Class Objects
* **model:** PyTorch two Layer Fully Connected Linear model with ELU Activation.
* **Agent:** RL Agent to act in environment. Also contains hyper-parameters.
	* BUFFER_SIZE: Size of buffer in memory
	* BATCH_SIZE: Size of batch to train model.
	* GAMMA: Reward Decay
	* LR: Learning Rate
	* HIDDEN_SIZE: Hidden size of model parameters.
	* UPDATE_EVERY: When to periodically update the network.
	* TAU: Not used because only one model is used.
	* Action(): Take Model action from state. isRandom=True: Takes a random action.
	* Step(): Adds env information to buffer & learns periodically using _SarsaQUpdate() Rule & UPDATE_EVERY parameter
* **ReplayBuffer:** Buffer for Experiance Replay. Used within Agent class.

## Additional Information
Use an initial for loop to loop through ‘n’ episodes, using env.reset() to reset the environment each time. Then, use another
loop (while True) to loop through the state action spaces in the environment until the done condition breaks out of the loop. 
Use Agent.Action() to get a  selected action for the state. Also, to use Epsilon Greedy with the function use parameter 
“isRandom:eps>random.random()” to use Epsilon Greedy. Where ‘eps’ is epsilon and ‘random()’ is a random number between zero and 
one. The following information is as follows: env_info = env.step(action)[brain_name], next_state = env_info.vector_observations[0], 
reward = env_info.rewards[0], done = env_info.local_done[0], score += reward (To get the score for the entire episode).  Next, use 
Agent.Step() to step the agent to the solution to solve the problem. Last, use the Jupyter notebook for more in depth detail of how 
to solve the problem, and make sure to print out the agent score periodically to know how the Agent is doing.