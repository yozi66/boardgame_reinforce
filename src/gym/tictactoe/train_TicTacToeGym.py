from matplotlib import pyplot as plt
import numpy as np
from TicTacToeXoEnv import TicTacToeXoEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# --------------------------------
# Train PPO agent for TicTacToe X player
# --------------------------------

env = TicTacToeXoEnv()
env = Monitor(env)
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, n_steps=128, batch_size=64, gamma=0.99)
model.learn(total_timesteps=100_000)

# --------------------------------
# Plot learning curve
# --------------------------------

results = env.get_episode_rewards()
n_blocks = 500
block_size = len(results) // n_blocks
results = [np.mean(results[i*block_size:(i+1)*block_size]) for i in range(n_blocks)]
plt.plot(results)
plt.xlabel(f"Blocks of {block_size} Episodes")
plt.ylabel("Reward")
plt.title("PPO learning curve")
plt.show()

# --------------------------------
# Use the learned policy
# --------------------------------
obs, info = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    env.render()
    done = terminated or truncated

# --------------------------------
# Evaluate the trained agent
# --------------------------------

env = TicTacToeXoEnv()
n_episodes = 1000
wins = draws = losses = 0
for episode in range(n_episodes):
    done = False
    reward = 0.0 # Initialize reward to make pylance happy
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    if reward == 1.0:
        wins += 1
    elif reward == 0.0:
        draws += 1
    else:
        losses += 1
    obs, info = env.reset()
print(f"Out of {n_episodes} episodes: Wins={wins}, Draws={draws}, Losses={losses}")

