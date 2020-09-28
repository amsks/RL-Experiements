import numpy as np

# Class to implement the replay buffer which will later be used by the agent
class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_ctr = 0        # To keep a track of the first available memory
        self.state_memory = np.zeros((self.mem_size, *input_shape))         
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))     # The agent is lookin for correlations between the states and the encounters, thus functiona equivalence
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)         # is the new state terminal or not, since the value of the terminal state is 0 -> 'done' flags

    # Update the data from transitions into the buffer
    def store_transition(self, state, action, reward, state_, done):
        
        index = self.mem_ctr % self.mem_size            # the agent will start updating when the counter exceeds the memory size

        self.state_memory[index] = state
        self.new_state_memory[index ] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_ctr += 1

    # Generate the samples in batches
    def sample_buffer(self, batch_size):
        
        max_mem = min(self.mem_ctr, self.mem_size)          # What is the position of the highest occupied memory ?

        batch = np.random.choice(max_mem, batch_size)           # Get a batch of random numbers to the highest occupied memory 

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones

    