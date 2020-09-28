'''
    Environment - Pendulum 
    Task - keep a frictionless pendulum standing up
    Actions : 0 - Joint Effort [-2.0, +2.0]
    Observations:
        0 - cos(theta)  [-1.0, +1.0]
        1 - sin(theta)  [-1.0, +1.0]
        2 - theta dor   [-8.0, 8.0]
    
    Reward: -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2)
'''


import gym 
import numpy as np
from ddpg_tf2 import Agent
from utils import plot_learning_curve

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
            n_actions=env.action_space.shape[0])
    episodes = 250

    figure_file = 'plots/pendulum.png'

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        n_steps = 0

        # Tf2 does not allow loading weights since it is an empty array
        # so we  do some stuff to initialize them for a batch if loading from checkpoint
        while n_steps <= agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            n_steps += 1

        agent.learn()
        agent.load_models()
        evaluate = True    

    else:
        evaluate = False
    
    for i in range(episodes):
        
        # Reset env, done and score
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation, evaluate)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            
            if not load_checkpoint:
                agent.learn()
            
            observation = observation_

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
    
    print('episode ', i, 'score %.1f' % score, 'avg score %.1f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(episodes)]
        plot_learning_curve(x, score_history, figure_file)