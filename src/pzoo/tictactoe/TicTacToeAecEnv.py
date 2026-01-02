from pettingzoo import AECEnv
from gymnasium import spaces
from typing import Optional
import numpy.typing as npt
import numpy as np

class TicTacToeAecEnv(AECEnv):
    metadata = {"render_modes": ["human"], "name": "tictactoe_v0"}

    def __init__(self) -> None:
        super().__init__()
        self.possible_agents = ["player_0", "player_1"]
        self.agents = []
        self.board: Optional[npt.NDArray[np.int8]] = None
        self.action_spaces = {a: spaces.Discrete(9) for a in self.possible_agents}
        # Example obs: 3x3 board with {-1,0,1} plus action mask
        self.observation_spaces = {
            a: spaces.Dict({
                "observation": spaces.Box(low=-1, high=1, shape=(3,3), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
            }) for a in self.possible_agents
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        self.agents = self.possible_agents[:]
        self.board = np.zeros((3,3), dtype=np.int8)
        self.rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self._agent_index = 0

    @property
    def agent_selection(self) -> str:
        return self.agents[self._agent_index]

    def observe(self, agent: str) -> dict:
        mask = (self.board.reshape(-1) == 0).astype(np.int8)
        return {"observation": self.board.copy(), "action_mask": mask}

    # Perform action for the current agent
    # action: int in [0..8] representing cell to mark
    def step(self, action: int) -> None:
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._agent_index = (self._agent_index + 1) % len(self.agents)
            return

        r, c = divmod(int(action), 3)
        if self.board[r, c] != 0:
            # illegal move: you can penalize or force loss
            self.rewards[agent] = -1.0
            self.terminations = {a: True for a in self.agents}
            return

        mark = 1 if agent == "player_0" else -1
        self.board[r, c] = mark

        winner = self._check_winner()
        if winner is not None:
            # +1 for winner, -1 for loser
            for a in self.agents:
                self.rewards[a] = 1.0 if a == winner else -1.0
                self.terminations[a] = True
        elif (self.board != 0).all():
            # draw
            for a in self.agents:
                self.rewards[a] = 0.0
                self.terminations[a] = True
        else:
            # continue
            for a in self.agents:
                self.rewards[a] = 0.0

        self._agent_index = (self._agent_index + 1) % len(self.agents)

    def _check_winner(self) -> Optional[str]:
        assert self.board is not None, "Call reset() before using board"
        # return "player_0" or "player_1" or None
        lines = []
        lines += [self.board[i, :] for i in range(3)]
        lines += [self.board[:, j] for j in range(3)]
        lines += [np.diag(self.board), np.diag(np.fliplr(self.board))]
        for line in lines:
            s = line.sum()
            if s == 3: return "player_0"
            if s == -3: return "player_1"
        return None

    def render(self) -> None:
        symbol_map = {1: "X", -1: "O", 0: " "}
        for r in range(3):
            print("|".join(symbol_map[self.board[r, c]] for c in range(3)))
            if r < 2:
                print("-----")
        print()
