import numpy as np

from generals.core.action import Action, compute_valid_move_mask
from generals.core.config import DIRECTIONS
from generals.core.observation import Observation
from generals.agents.agent import Agent


class MyExpanderAgent(Agent):
    def __init__(self, id: str = "MyExpander"):
        super().__init__(id)

    def act(self, observation: Observation) -> Action:
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells.
        Distributes armies to weaker cells when no captures are possible.
        """

        mask = compute_valid_move_mask(observation)
        valid_moves = np.argwhere(mask == 1)

        # Skip the turn if there are no valid moves.
        if len(valid_moves) == 0:
            return Action(to_pass=True)

        army_mask = observation.armies
        opponent_mask = observation.opponent_cells
        neutral_mask = observation.neutral_cells

        # Find moves that capture opponent or neutral cells
        capture_opponent_moves = np.zeros(len(valid_moves))
        capture_neutral_moves = np.zeros(len(valid_moves))
        distribute_army_moves = np.zeros(len(valid_moves))

        for move_idx, move in enumerate(valid_moves):
            orig_row, orig_col, direction = move
            row_offset, col_offset = DIRECTIONS[direction].value
            dest_row, dest_col = (orig_row + row_offset, orig_col + col_offset)
            enough_armies_to_capture = army_mask[orig_row, orig_col] > army_mask[dest_row, dest_col] + 1

            if opponent_mask[dest_row, dest_col] and enough_armies_to_capture:
                capture_opponent_moves[move_idx] = 1
            elif neutral_mask[dest_row, dest_col] and enough_armies_to_capture:
                capture_neutral_moves[move_idx] = 1
            elif army_mask[dest_row, dest_col] < army_mask[orig_row, orig_col] // 2:
                distribute_army_moves[move_idx] = army_mask[orig_row, orig_col] // 2 - army_mask[dest_row, dest_col]

        to_split = False
        if np.any(capture_opponent_moves):  # Capture random opponent cell if possible
            move_index = np.random.choice(np.nonzero(capture_opponent_moves)[0])
            move = valid_moves[move_index]
        elif np.any(capture_neutral_moves):  # Capture random neutral cell if possible
            move_index = np.random.choice(np.nonzero(capture_neutral_moves)[0])
            move = valid_moves[move_index]
        elif np.any(distribute_army_moves):  # Distribute armies to weaker cells
            move_index = np.argmax(distribute_army_moves)
            move = valid_moves[move_index]
            to_split = True
        else:  # Otherwise, select a random valid action
            move_index = np.random.choice(len(valid_moves))
            move = valid_moves[move_index]

        action = Action(to_pass=False, row=move[0], col=move[1], direction=move[2], to_split=to_split)
        return action

    def reset(self):
        pass
