#! /usr/bin/python3

'''
	State = (Position, velocity)
	Actions = {
		0 - Push the car left 
		1 - Do nothing 
		2 - Push the car right
	}
	reward = -1 till agent reaches the goal

    TODO: -> Reward shaping
          -> Plan in state state space
          -> Euclidean distances  
'''

import gym
import math
import numpy as np
import matplotlib.pyplot as plt


env = gym.make("MountainCar-v0")

# F = np.abs(env.goal_position - env.agent_position)


LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 35000
SHOW_EVERY = 1000
STAT_EVERY = 100

# Q-table Settings
# DISCRETE_OS_SIZE = [20, 20]
DISCRETE_OS_SIZE = [40] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
# END_EPSILON_DECAYING = EPISODES
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/np.abs(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
EPSILON_MINIMUM = 0.01

victory_counter = 0
first = True
first_victory = 0

# Reward tracking mechanism
ep_rewards = []
aggr_ep_rewards = {
    'ep'  : [],     # Disctionary to track episodes
    'avg' : [],     # Trailing average of average reward
    'min' : [],     # worst model we had 
    'max' : []      # Best model 
}

#Function to get the discrete value of a Q state

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))  # we use this tuple to look up the 3 Q values for the available actions in the q-table


def height(position):
    low = - 0.85
    high = 0.5
    
    if position < low:
        position = low + np.abs(position - low)    

    return_val = ((position - low)/(high - low)) - 1
    return return_val 

def height2(x_position):
    x_low = - 0.85
    x_high = 0.5

    y_low = -0.5
    y_high = + 0.5

    if x_position < x_low :
        x_position = x_low + np.abs(x_position - x_low)

    slope = (y_high - y_low)/((x_high - x_low)**3) 

    y_position = slope * ((x_position - x_low)**3) + y_low

    return y_position - 1


def Phi(position):
    
    output = 0.5 * (height(env.goal_position) - height(position))/0.8


    return output

# Wrapper to modify the rewards
class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # Reward Shaping
        # reward = next_state[0] - env.goal_position
        reward = height2(next_state[0])
        # reward = DISCOUNT * height(next_state)
        # print(env.observation_space.low)
        return next_state, reward, done, info


env = RewardWrapper(env)


for episode in range(EPISODES):
    episode_reward = 0
    state = env.reset()
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
        average_reward = sum(ep_rewards[-STAT_EVERY:])/STAT_EVERY
        print(f'Episode: {episode}, average reward: {average_reward}, current epsilon: {epsilon}')
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)


        new_state, reward, done, _ = env.step(action)

        reward = DISCOUNT * height2(new_state[0]) - height2(state[0]) - 1

        # next_state = np.reshape(new_state, (1, 2))
        # curr_state = np.reshape(state, (1, 2))
        # #Customised reward function
        # reward = 100*((math.sin(3*next_state[0,0]) * 0.0025 + 0.5 * next_state[0,1] * next_state[0,1]) - (math.sin(3*curr_state[0,0]) * 0.0025 + 0.5 * curr_state[0,1] * curr_state[0,1])) 

        new_discrete_state = get_discrete_state(new_state)

        # reward = reward + np.abs(DISCOUNT*new_discrete_state[0] - discrete_state[0])
        
        episode_reward += reward

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
            q_table[discrete_state + (action,)] = 0
            print(f"Done on episode: {episode}" )
            victory_counter += 1 

            if first:
                first_victory = episode
                first = False

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING and epsilon > EPSILON_MINIMUM:
        epsilon -= epsilon_decay_value
    else:
        epsilon = EPSILON_MINIMUM



    ep_rewards.append(episode_reward)

    if episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    if not episode % STAT_EVERY:
        average_reward = sum(ep_rewards[-STAT_EVERY:])/STAT_EVERY
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['max'].append(max(ep_rewards[-STAT_EVERY:]))
        aggr_ep_rewards['min'].append(min(ep_rewards[-STAT_EVERY:]))

        # np.save(f"qtables/{episode}-qtable.npy", q_table)
print(f"Total Victories: {victory_counter}, First Victory : {first_victory}" )

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.grid(True)
plt.show()