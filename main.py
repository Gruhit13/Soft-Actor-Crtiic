from agent import Agent
from config import Config

import gym
import numpy as np
import torch as T
import random

SEED = 13
np.random.seed(SEED)
T.manual_seed(SEED)
random.seed(SEED)

import warnings
warnings.filterwarnings("ignore")

def train(env: gym.Env, agent: Agent, config: Config):

    rewards = []
    best_reward = -np.inf

    for e in range(config.episodes):
        state = env.reset()
        done = False
        episodic_reward = 0
        steps = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            steps += 1
            episodic_reward += reward

            agent.append(state, action, reward, done, next_state)

            state = next_state

            agent.update(config.batch_size)

        rewards.append(episodic_reward)
        if episodic_reward > best_reward:
            best_reward = episodic_reward

            # Save the best models
            # agent.save()
        
        if (e+1) % config.print_at == 0: 
            avg_reward = np.mean(rewards[-config.avg_over:])
            log = (f"{e+1} | Steps: {steps} | Episodic Reward: {episodic_reward:.1f} |",
                f"Avg Reward: {avg_reward:.1f} | Best Reward: {best_reward:.1f}")
            
            print(*log)

if __name__ == "__main__":

    config = Config

    env = gym.make(config.env_name)

    agent = Agent(
        env = env,
        gamma = config.gamma,
        tau = config.tau,
        alpha = config.init_temperature,
        q_lr = config.critic_lr,
        policy_lr = config.actor_lr,
        a_lr = config.alpha_lr,
        buffer_maxlen = config.max_experience
    )
    train(env, agent, config)