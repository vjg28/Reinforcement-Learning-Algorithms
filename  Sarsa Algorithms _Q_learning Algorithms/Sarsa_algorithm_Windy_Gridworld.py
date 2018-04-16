import gym
import numpy as np
import pandas as pd    
from collections import defaultdict
import plotting

def epsilon_greedy_policy(observation , Q, epsilon, nA):
    A= np.ones(nA, dtype=float) * epsilon / nA
    best_action= np.argmax(Q[observation])
    A[best_action] += (1 - epsilon)
    return A

def sarsa_control_epsilon_greedy(env, n_episodes, epsilon, discount_factor, alpha ):
    
    Q=defaultdict(lambda: np.zeros(env.action_space.n))
    final_policy=defaultdict(lambda: np.zeros(env.action_space.n))
    stats = plotting.EpisodeStats(episode_lengths=np.zeros(n_episodes),episode_rewards=np.zeros(n_episodes))
    
    for i in range(n_episodes):
        state=env.reset()
        done=False
        prob= epsilon_greedy_policy(state, Q, epsilon, env.action_space.n)
        action = np.random.choice(np.arange(len(prob)), p=prob)
        while not done:            
            
            next_state, reward, done, _ = env.step(action)
            next_prob = epsilon_greedy_policy(next_state, Q , epsilon, env.action_space.n) 
            next_action = np.random.choice(np.arange(len(next_prob)),p=next_prob )
            
            Q[state][action] += alpha*(reward + discount_factor * Q[next_state][next_action] - Q[state][action])
            
            stats.episode_rewards[i] += reward
            stats.episode_lengths[i] += 1
            
            state=next_state
            action=next_action
        
    for _state in Q:
        final_policy[_state]= epsilon_greedy_policy( _state, Q, 0.0 , env.action_space.n)
        
    return Q, final_policy , stats

