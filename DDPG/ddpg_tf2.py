import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, alpha=0.001, beta=0.002, input_dims=[8], env=None,
            gamma=0.99, n_actions=2, max_size=1000000, tau=0.005, 
            fc1=512, fc2=512, batch_size=64):
        
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        
        # Define the actor and the critic
        self.actor = ActorNetwork(n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(n_actions=n_actions, name='critic')
        
        # Use slowly moving copies of the actor and the critic to train loss
        self.target_actor = ActorNetwork(n_actions=n_actions, name='target_actor')
        self.target_critic = CriticNetwork(n_actions=n_actions, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)
       

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
    
    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.target_actor.save_weights(self.target_actor.checkpoint_file)
        self.critic.save_weights(self.critic.checkpoint_file)
        self.target_critic.save_weights(self.target_critic.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.actor.checkpoint_file)
        self.target_actor.load_weights(self.target_actor.checkpoint_file)
        self.critic.load_weights(self.critic.checkpoint_file)
        self.target_critic.load_weights(self.target_critic.checkpoint_file)

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)

        # Add some noise for exploration
        # Not using OU Noise, jut gaussians
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions], 
                    mean=0.0, stddev=0.1)

        # To handle value overloads
        # If an environment has actions more than 1, we have to multiple by max_action
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]
    
    def learn(self):
        
        # If a batch has not been completed,  learning will not begin
        # can also do an alternative test on making it do random stuff in this period
        if self.memory.mem_ctr < self.batch_size:
            return

        # Get the samples from the buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

       
        # Update for the critic network
        with tf.GradientTape() as tape:
            
            # What does the target actor think of future states
            target_actions = self.target_actor(states_)
            
            # Use the target actions generated by target actor's opinions to get what 
            # the target critic think of the current states ( One-step Lookahead)
            critic_value_ = tf.squeeze(self.target_critic(
                                states_, target_actions), 1)

            # What does the critic think of the current states and actions
            critic_value = tf.squeeze(self.critic(states, actions), 1)
            
            # Define the target as chasing the opinion of the target critic
            target = reward + self.gamma*critic_value_*(1-done)
            
            # minimize the loss between the the target critic's opinion and the current critic's opinion
            critic_loss = keras.losses.MSE(target, critic_value)

        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        # Update for the actor network
        with tf.GradientTape() as tape:
            # What actions should we take according to the current policy ?
            new_policy_actions = self.actor(states)

            # define the loss by passing the actions suggested by the current actor thorugh the critic
            actor_loss = -self.critic(states, new_policy_actions)
            
            # Minimize the variance between what the actor thinks of acting on and what the critic suggest - Align their opinions
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        #Update the target networks
        self.update_network_parameters()