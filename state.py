import numpy as np
import random
import copy

# random.seed(1)

# returns new board
def init():
    return encode(np.zeros((3, 3), dtype=int))

def get_random_player():
    return random.randrange(2) + 1

def encode(state):
    encoded = ""
    for row in state:
        for element in row:
            encoded += str(element)
    return encoded

def decode(state):
    encoded = []
    for c in state:
        encoded.append(int(c))
    return np.array(encoded).reshape(3, 3)

def move(state, action, player):
    state = decode(state)
    x = action % state.shape[0]
    y = action // state.shape[0]
    if state[y][x] != 0: print("wrong move")
    state[y][x] = player
    won = check_won(state)
    return encode(state), won

def next_player(player):
    return 1 + 1 * (player == 1)

# returns legal actions in given state
def get_action_space(state):
    state = decode(state)
    flat_list = [item for row in state for item in row]
    return [i for i, item in enumerate(flat_list) if item == 0]

# returns random legal action
def get_action_space_sample(state):
    return random.choice(get_action_space(state))

# returns True if the game has a winner, False if the game is a draw, otherwise None
def check_won(state):
    diagonal1 = list(np.diag(np.fliplr(state)))
    diagonal2 = list(np.diag(state))
    if diagonal1.count(diagonal1[0]) == len(diagonal1) and diagonal1[0] != 0: return True
    if diagonal2.count(diagonal2[0]) == len(diagonal2) and diagonal2[0] != 0: return True
    for row1, row2 in zip(state, state.T):
        if list(row1).count(row1[0]) == len(row1) and row1[0] != 0: return True
        if list(row2).count(row2[0]) == len(row2) and row2[0] != 0: return True
    if 0 not in state: return False

# prepares a game state for neural network input (no historic states)
def state_to_batch(state, player):
    state = decode(state)
    b1 = [1 if el == player else 0 for row in state for el in row]
    b2 = [1 if el == next_player(player) else 0 for row in state for el in row]
    b3 = [player - 1 for row in state for el in row]
    b1 = np.array(b1).reshape((3, 3))
    b2 = np.array(b2).reshape((3, 3))
    b3 = np.array(b3).reshape((3, 3))
    return [b1, b2, b3]