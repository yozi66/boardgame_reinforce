from TicTacToeEnv import TicTacToeEnv
import numpy as np

env = TicTacToeEnv(render_mode="human")
obs, info = env.reset()

reward = None # make pylance happy
done = False
while not done:
    # random valid move:
    valid = np.flatnonzero(obs["board"] == 0)
    a = int(np.random.choice(valid))
    obs, reward, terminated, truncated, info = env.step(a)
    done = terminated or truncated
print("reward:", reward, "info:", info)
