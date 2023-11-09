import numpy as np
import gymnasium as gym
import random
import time
from IPython.display import clear_output

env = gym.make("Taxi-v3")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros([state_space_size, action_space_size])

num_episodes = 100000             #total nbr of episodes the agents should play during training
max_steps_per_episode = 100000    # max steps the agent makes if the episode does not terminate
alpha = 0.05                      # learning rate
gamma = 0.998                     # discount rate
epsilon = 1                       # exploration rate, (exploration vs exploitation), 1st move can't be exploitation
max_epsilon = 1
min_epsilon = 0.001
epsilon_decay_rate = 0.0001

total_rewards = []                        #hold all the rewards won in all the episodes, see how the score evolves

for episode in range(num_episodes):
    state = env.reset()
    try:
        state = state[0]
    except:
        pass
    done = False
    rewards_current_episode = 0
    
    for step in range (max_steps_per_episode):
        # Epsilon set up
        epsilon_threshold = random.uniform(0, 1) # exploration rate threshold
        if epsilon < epsilon_threshold :
            action = np.argmax(q_table[state])
        else:
            action = env.action_space.sample()
            
        new_state, reward, done, truncated, info = env.step(action)
        # update Q_table for Q(s, a)
        q_table[state, action] = (q_table[state, action])*(1-alpha) + alpha * (reward + gamma * (np.max(q_table[new_state])))
        state = new_state
        rewards_current_episode += reward
        
        if done == True:
            break
            
    # Exploration rate decay
    epsilon = min_epsilon + (max_epsilon - min_epsilon)* np.exp(-epsilon_decay_rate*episode)
    
    total_rewards.append(rewards_current_episode)
    
    # calculate average reward per 10000 episodes
    rewards_per_10000ep = np.split(np.array(total_rewards), num_episodes/10000)
    count = 10000
    for i in rewards_per_10000ep:
        print(count, ";", str(sum(i)/10000))
        count += 10000
        
    print(q_table)
    
    