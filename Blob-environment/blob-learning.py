'''
	Working with blob environment, and creating a learning agent. 

	State = Obs -> (Distance to food, Distance to Enemy)
	Actions -> move in one of the random directions
	Rewards = Penalties for each step, hitting the enemy and getting the food 

'''



import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use("ggplot")

SIZE = 10

#Constants
EPISODES = 25000
SHOW_EVERY = 3000  # how often to play through env visually.
LEARNING_RATE = 0.1
DISCOUNT = 0.95
PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

#Rewards
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

#Exploration Settings
epsilon = 0.9
EPS_DECAY = 0.9998  # Every episode will be epsilon*EPS_DECAY

#Initialize q table 
start_q_table = None # None or Filename


# the dictionary for colors
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}


# TODO: try experimenting with other class types for multi-agent systems
class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)

    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def action(self, choice):
        '''
        4 movement options. (0,1,2,3) - TODO Extend for diagonal movements and see what happens
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y


        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


if start_q_table is None:
    # initialize the q-table
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                    for y2 in range(-SIZE+1, SIZE):
                        q_table[((x1,y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)


# print(q_table[((-9, -2), (3, 9))]) for example

#Accumulated rewards
episode_rewards = []

for episode in range(EPISODES):
    player = Blob()
    food = Blob()
    enemy = Blob()
    
    #Render at SHOW_EVERY
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    
    for i in range(200):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### TODO - try different move initializations ###
        #enemy.move()
        #food.move()
        ##############

        # Logic for rewards
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        
        # Get the observations
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)             # start an rbg of size
            env[food.x][food.y] = d[FOOD_N]                             # set the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]                       # set the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]                          # set the enemy location to red
            img = Image.fromarray(env, 'RGB')                           # read to rgb
            img = img.resize((300, 300))                                # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))                          # Render the image
            
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:       # Handling of abrupt ending
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break

    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
