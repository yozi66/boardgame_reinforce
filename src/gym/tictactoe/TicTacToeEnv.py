import numpy as np
import gymnasium as gym
from gymnasium import spaces

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
]

class TicTacToeEnv(gym.Env):
    """
    Two-player turn-based Tic-Tac-Toe.

    Observation:
      Dict({
        "board": Box(low=-1, high=1, shape=(9,), dtype=int8),
        "player": Discrete(2)  # 0 -> X to move, 1 -> O to move
      })

    Action:
      Discrete(9) cell index 0..8

    Rewards:
      +1 if the acting player wins
       0 if draw
      -1 if illegal move (and episode terminates)
       0 otherwise (non-terminal move)
    """
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8),
            "player": spaces.Discrete(2),
        })
        self.action_space = spaces.Discrete(9)

        self.board = np.zeros(9, dtype=np.int8)
        self.player = 1  # 1 for X, -1 for O (we'll map to 0/1 in obs)

    def _obs(self):
        return {
            "board": self.board.copy(),
            "player": 0 if self.player == 1 else 1
        }

    def _winner(self):
        # returns 1 if X wins, -1 if O wins, 0 otherwise
        b = self.board
        for i, j, k in WIN_LINES:
            s = b[i] + b[j] + b[k]
            if s == 3:
                return 1
            if s == -3:
                return -1
        return 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board[:] = 0
        self.player = 1  # X starts
        info = {}
        if self.render_mode == "human":
            self.render()
        return self._obs(), info

    def step(self, action):
        terminated = False
        truncated = False
        info = {}

        # Illegal move: occupied cell
        if self.board[action] != 0:
            if (self.render_mode == "human"):
                print(f"Illegal move by player {'X' if self.player == 1 else 'O'} at cell {action}")
            terminated = True
            reward = -1.0
            info["illegal_move"] = True
            if self.render_mode == "human":
                self.render()
            return self._obs(), reward, terminated, truncated, info

        # Apply move
        self.board[action] = self.player

        w = self._winner()
        if w != 0:
            terminated = True
            reward = 1.0  # acting player just won
            info["winner"] = "X" if w == 1 else "O"
            if self.render_mode == "human":
                print(f"Player {'X' if w == 1 else 'O'} wins!")
        elif np.all(self.board != 0):
            terminated = True
            reward = 0.0
            info["winner"] = None  # draw
            if self.render_mode == "human":
                print("Game ended in a draw.")
        else:
            reward = 0.0

        # Switch player if game continues
        if not terminated:
            self.player *= -1

        if self.render_mode == "human":
            self.render()

        return self._obs(), reward, terminated, truncated, info

    def render(self):
        symbols = {1: "X", -1: "O", 0: " "}
        b = [symbols[int(v)] for v in self.board]
        s = (
            f"{b[0]}|{b[1]}|{b[2]}\n"
            f"-+-+-\n"
            f"{b[3]}|{b[4]}|{b[5]}\n"
            f"-+-+-\n"
            f"{b[6]}|{b[7]}|{b[8]}\n"
        )
        if self.render_mode == "ansi":
            return s
        print(s)

    def close(self):
        pass
