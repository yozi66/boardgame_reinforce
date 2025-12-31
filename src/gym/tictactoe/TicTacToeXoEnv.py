# This is a TicTacToe environment for training a player using Gym and Stable Baselines3
# The player can be either X (1) or O (-1)

from TicTacToeEnv import TicTacToeEnv

class TicTacToeXoEnv(TicTacToeEnv):
    """
    Tic-Tac-Toe environment where the agent plays as X (1) and O (-1) is played by a heuristic agent.
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        info = {}
        # 50% chance to play as X or O
        if self.np_random.random() < 0.5:
            self._heuristicStep()  # internal agent plays as X
            info["player"] = "O" # info is from external agent perspective
        else:
            info["player"] = "X"
        return self._obs(), info

    def step(self, action):
        # External agent makes a move
        obs, reward, terminated, truncated, info = super().step(action)

        if terminated or truncated:
            return obs, reward, terminated, truncated, info

        # Internal agent makes a move
        obs_i, reward_i, terminated_i, truncated_i, info_i = self._heuristicStep()
        if obs_i is not None:
            obs = obs_i
            reward = -reward_i  # From external agent perspective
            terminated = terminated_i
            truncated = truncated_i
            info.update(info_i)
        return obs, reward, terminated, truncated, info

    def _heuristicStep(self):
        # evaluate all possible moves
        empty_cells = [i for i in range(9) if self.board[i] == 0]
        win_moves = []
        block_moves = []
        for cell in empty_cells:
            # check if winning move
            self.board[cell] = self.player
            if self._winner() == self.player:
                win_moves.append(cell)
            else:
                # check if blocking move
                self.board[cell] = -self.player
                if self._winner() == -self.player:
                    block_moves.append(cell)
            self.board[cell] = 0 # undo move
        # Heuristic agent makes a move
        if win_moves:
            o_action = self.np_random.choice(win_moves)
        elif block_moves:
            o_action = self.np_random.choice(block_moves)
        elif empty_cells:
            o_action = self.np_random.choice(empty_cells)
            return super().step(o_action)
        # No valid moves -> terminal game end (not truncated)
        return self._obs(), 0.0, True, False, {"error": "No valid moves"}
    