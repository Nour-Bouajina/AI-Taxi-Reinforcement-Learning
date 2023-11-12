import numpy as np
import gymnasium as gym
import random
import time
import tensorflow as tf
import tqdm
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from collections import deque
import numpy as np

env = gym.make("Taxi-v3")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n
q_table = np.zeros([state_space_size, action_space_size])

num_episodes = 100000             #total nbr of episodes the agents should play during training
max_steps_per_episode = 1000      # max steps the agent makes if the episode does not terminate
alpha = 0.05                      # learning rate
gamma = 0.998                     # discount rate
epsilon = 1                       # exploration rate, (exploration vs exploitation), 1st move can't be exploitation
max_epsilon = 1
min_epsilon = 0.001
epsilon_decay_rate = 0.0001
replay_memory_capacity = 100
min_replay_memory_capacity = 60
batch_size = 64 
UPDATE_TARGET_EVERY = 5

tf.random.set_seed(42)
q_net = tf.keras.models.Sequential([
            Dense(64, activation = "relu", input_shape = (env.observation_space.n,)),
            Dense(64, activation = "relu"),
            Dense(env.action_space.n, activation = "linear")])

target_net = tf.keras.models.Sequential([
            Dense(64, activation = "relu", input_shape = (env.observation_space.n,)),
            Dense(64, activation = "relu"),
            Dense(env.action_space.n, activation = "linear")])

def get_action(q_values, epsilon):
    epsilon_threshold = random.uniform(0, 1) # exploration rate threshold
    if epsilon < epsilon_threshold :
        action = np.argmax(q_values.numpy()[0])
    else:
        action = np.random.randint(0, env.action_space.n)
    
def get_experiences(replay_memory):
    exp = random.sample(replay_memomry, k=64)
    experiences = random.sample(memory_buffer, k=64)
    states = tf.convert_to_tensor(np.array([e.state for e in exp if e is not None]),dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array([e.action for e in exp if e is not None]), dtype=tf.float32)
    rewards = tf.convert_to_tensor(np.array([e.reward for e in exp if e is not None]), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array([e.next_state for e in exp if e is not None]),dtype=tf.float32)
    done_vals = tf.convert_to_tensor(np.array([e.done for e in exp if e is not None]).astype(np.uint8),dtype=tf.float32)
    return (states, actions, rewards, next_states, done)

def update_target_net(q_net, target_net):
    for i in range(len(target_q_network.weights)):
        target_q_network.weights[i] = 0.001 * q_network.weights[i] + (1.0 - 0.001) * target_q_network.weights[i]
    
def compute_loss(experiences, gamma, q_net, target_net):
    states, actions, rewards, next_states, done = experiences
    max_qsa = tf.reduce_max(target_net(next_states), axis=-1)
    y_targets = rewards + (gamma * max_qsa * (1-done))
    q_values = q_net(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]), tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)
    return loss

def agent_learn(experiences, gamma, q_net, target_net, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_net, target_net)
        
    gradients = tape.gradient(loss, q_net.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_net.trainable_variables))
    update_target_net(q_net, target_net)
    
    
def train():
    replay_memory = deque(maxlen = replay_memory_capacity)
    target_net.set_weights(q_net.get_weights()) 
    rewards_history = []
    
    for episode in range(1, num_episodes+1):
        rewards_current_episode = 0
        step = 1
        #initialize starting state
        current_state = env.reset()
        done = False
        #for each time step
        while not done:
            q_values = q_net(current_state)
            action = (q_values, epsilon)
            #execute the action
            new_state, reward, done, _ , __ = env.step(action)
            # add experience to mem
            replay_memory.append((current_state, action, reward, new_state, done))
            
            if (len(replay_memory) > 60):
                experiences = get_experiences(replay_memory)
                agent_learn(experiences, gamma, q_net, target_net, optimizer)
                
            state = copy.deepcopy(next_state)
            rewards_current_episode += reward
            
        rewards_history.append(rewards_current_episode)
        epsilon = max(min_epsilon, epsilon_decay_rate * epsilon)
        
        avg_rewards = np.mean(rewards_history[-50:])
        if(avg_points >= 8):
            q_net.save('./TaxiSol.h5')
            
train()
            

