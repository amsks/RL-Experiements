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
# print(env.observation_space.high)
# print(env.observation_space.low)

#Hperparameters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 2000
SHOW_EVERY = 500

# Window size for converting the values into bucket integers
DISCRETE_OS_SIZE = [20, 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
# print(discrete_os_win_size)

#Create the Q table from each combination of actions
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table.shape)
# print(q_table)

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

discrete_state = get_discrete_state(env.reset()) 	# Initialize the discrete states

done = False

for episode in range(EPISODES):
	discrete_state = get_discrete_state(env.reset())
	done = False

	if episode % SHOW_EVERY == 0:
		render = True
		print(episode)
	else:
		render = False

	while not done:

		action = np.argmax(q_table[discrete_state])
		new_state, reward, done, _ = env.step(action)

		new_discrete_state = get_discrete_state(new_state)

		if episode % SHOW_EVERY == 0:
			env.render()
		#new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

		# If simulation did not end yet after last step - update Q table
		if not done:

			# Maximum possible Q value in next step (for new state)
			max_future_q = np.max(q_table[new_discrete_state])

			# Current Q value (for current state and performed action)
			current_q = q_table[discrete_state + (action,)]

			# And here's our equation for a new Q value for current state and action
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

			# Update Q table with new Q value
			q_table[discrete_state + (action,)] = new_q


		# Simulation ended (for any reson) - if goal position is achived - update Q value with reward directly
		elif new_state[0] >= env.goal_position:
			#q_table[discrete_state + (action,)] = reward
			print("Done on episode :" + str(episode))
			q_table[discrete_state + (action,)] = 0 

		discrete_state = new_discrete_state


env.close()