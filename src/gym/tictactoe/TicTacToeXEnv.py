# This is a TicTacToe environment for training the X player using Gym and Stable Baselines3

from TicTacToeEnv import TicTacToeEnv

class TicTacToeXEnv(TicTacToeEnv):
    """
    Tic-Tac-Toe environment where the agent plays as X (1) and O (-1) is played by a random agent.
    """

    def step(self, action):
        # Agent (X) makes a move
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Random agent (O) makes a move
        empty_cells = [i for i in range(9) if self.board[i] == 0]
        if empty_cells:
            o_action = self.np_random.choice(empty_cells)
            obs, reward_o, terminated, truncated, info = super().step(o_action)
            reward = -reward_o  # From X's perspective

        return obs, reward, terminated, truncated, info
