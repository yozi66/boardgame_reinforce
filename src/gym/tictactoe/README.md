gym/tictactoe
=============

* [TicTacToeEnv.py](TicTacToeEnv.py): A symple gymnasium Env for the Tic Tac Toe game. Not useful for training since the players must to take turns and it is not supported by gymnasium learners. 
* [TicTacToeXEnv.py](TicTacToeXEnv.py): Improved version of TicTacToeEnv for agent X that alway takes the first turn. The other player always takes a random valid move. 
* [TicTacToeXoEnv.py](TicTacToeXoEnv.py): Further improvements, the AI player is selected by a 50% chance to be X (first) or O (second). The built-in opponent is a heuristics that tries to take a winning move or (if no winning move) tries to block the winning move of the opponent. Otherwise it takes a random move. 
* [quick_test.py](quic_test.py): Plays a random game in TicTacToeEnv and prints the gameplay.
* [train_TicTacToeGym.py](train_TicTacToeGym.py): trains the policy in TicTacToeXoEnv with PPO (Proximal Policy Optimization), shows the learning curve, shows a sample gameplay in deterministic mode (the trained agent does not make and random "exploratory" moves) and computes gameplay statistics in deterministic mode. 
