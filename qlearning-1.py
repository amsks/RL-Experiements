#! /usr/bin/python3

'''
	State = (Position, velocity)
	Actions = {
		0 - Push teh car left 
		1 - Do nothing 
		2 - Push the car right
	}
	reward = -1 till agent reaches the goal


'''

import gym
import numpy as np
 
env = gym.make("MountainCar-v0")	#Import evironment from openAI -gym
state = env.reset()					# Reset the states

print(env.observation_space.high)
print(env.observation_space.low)

# Window size for converting the values into bucket integers
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
print(discrete_os_win_size)

#Create the Q table from each combination of actions
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
print(q_table.shape)
print(q_table)


done = False

while not done:
    action = 2											# Fixed action
    new_state, reward, done, _ = env.step(action)		# Get the reward
    print(reward, new_state)							# 