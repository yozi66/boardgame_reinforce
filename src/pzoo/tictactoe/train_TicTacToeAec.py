# train TicTacToe AEC agent
import numpy as np
from TicTacToeAecEnv import TicTacToeAecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
# --------------------------------
# Train PPO agent for TicTacToe AEC environment
# --------------------------------
env = TicTacToeAecEnv()
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=128, batch_size=64, gamma=0.99)
model.learn(total_timesteps=100_000)
