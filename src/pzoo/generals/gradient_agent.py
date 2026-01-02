import numpy as np

from collections import deque
from generals.core.action import Action, compute_valid_move_mask
from generals.core.config import DIRECTIONS
from generals.core.observation import Observation
from generals.agents.agent import Agent

class GradientAgent(Agent):
    def __init__(self, id: str = "Gradient"):
        super().__init__(id)

    def act(self, observation: Observation) -> Action:
        """
        Heuristically selects a valid (expanding) action.
        Prioritizes capturing opponent and then neutral cells.
        Moves the biggest army towards free or enemy cells if no captures are possible.
        """

        valid_move_mask = compute_valid_move_mask(observation)
        valid_moves = np.argwhere(valid_move_mask == 1)

        # Skip the turn if there are no valid moves.
        if len(valid_moves) == 0:
            return Action(to_pass=True)

        army_mask = observation.armies
        my_mask = observation.owned_cells
        opponent_mask = observation.opponent_cells
        neutral_mask = observation.neutral_cells
        mountains_mask = observation.mountains

        # Find moves that capture opponent or neutral cells
        capture_opponent_moves = np.zeros(len(valid_moves))
        capture_neutral_moves = np.zeros(len(valid_moves))

        for move_idx, move in enumerate(valid_moves):
            orig_row, orig_col, direction = move
            row_offset, col_offset = DIRECTIONS[direction].value
            dest_row, dest_col = (orig_row + row_offset, orig_col + col_offset)
            enough_armies_to_capture = army_mask[orig_row, orig_col] > army_mask[dest_row, dest_col] + 1

            if opponent_mask[dest_row, dest_col] and enough_armies_to_capture:
                capture_opponent_moves[move_idx] = 1
            elif neutral_mask[dest_row, dest_col] and enough_armies_to_capture:
                capture_neutral_moves[move_idx] = 1

        move = None
        if np.any(capture_opponent_moves):  # Capture random opponent cell if possible
            move_index = np.random.choice(np.nonzero(capture_opponent_moves)[0])
            move = valid_moves[move_index]
        elif np.any(capture_neutral_moves):  # Capture random neutral cell if possible
            move_index = np.random.choice(np.nonzero(capture_neutral_moves)[0])
            move = valid_moves[move_index]
        else:
            # find the biggest army
            my_army_mask = army_mask * my_mask
            max_army_flat_index = np.argmax(my_army_mask)
            max_army_pos = np.unravel_index(max_army_flat_index, army_mask.shape)
            height = army_mask.shape[0]
            width = army_mask.shape[1]

            # breadth-first search to find the nearest opponent or neutral cell
            queue = deque()
            visited = set()
            queue.append((max_army_pos, None))
            visited.add(max_army_pos)
            while queue:
                current_pos, old_move = queue.popleft()
                if opponent_mask[current_pos] or neutral_mask[current_pos]:
                    move = old_move
                    break
                # valid moves from current position
                row, col = current_pos
                for direction in range(4):
                    row_offset, col_offset = DIRECTIONS[direction].value
                    next_row, next_col = row + row_offset, col + col_offset
                    if (next_row < 0 or next_row >= height or next_col < 0 or next_col >= width):
                        continue
                    neighbor = (row + row_offset, col + col_offset)
                    if mountains_mask[neighbor]:
                        continue
                    if neighbor not in visited:
                        new_move = (row, col, direction) if old_move is None else old_move
                        visited.add(neighbor)
                        queue.append((neighbor, new_move))
            # If no move was found, select a random valid action
            if move is None:
                move_index = np.random.choice(len(valid_moves))
                move = valid_moves[move_index]

        action = Action(to_pass=False, row=move[0], col=move[1], direction=move[2], to_split=False)
        return action

    def reset(self):
        pass
