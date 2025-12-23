from TicTacToeAecEnv import TicTacToeAecEnv
import numpy as np

env = TicTacToeAecEnv()
env.reset()

done = False
while not done:
    obs = env.observe(env.agent_selection)
    mask = obs["action_mask"]
    valid = np.flatnonzero(mask)
    # random valid move:
    a = int(np.random.choice(valid))
    env.step(a)
    env.render()
    done = all(env.terminations.values()) or all(env.truncations.values())
    print("rewards:", env.rewards, "infos:", env.infos)
