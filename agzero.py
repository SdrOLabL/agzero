import math
import state as s
import copy
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model import Net
import model
import random
import os

BATCH_SIZE = 256    
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 1000
EVALUATE_EVERY_STEP = 300

class MCTS:
    def __init__(self, c_puct = 1.0):
        self.c_puct = c_puct
        self.visit_counts = {}
        self.values = {} 
        self.values_avg = {}
        self.probs = {}

    def is_leaf(self, state):
        return state not in self.probs

    def find_leaf(self, state, player):
        # store traversed states and actions to be able to backpropagate / update
        states = []
        actions = []
        cur_state = state
        cur_player = player
        value = None

        while not self.is_leaf(cur_state):
            states.append(cur_state)
            
            counts = self.visit_counts[cur_state]
            total_sqrt = math.sqrt(sum(counts))
            probs = self.probs[cur_state]
            values_avg = self.values_avg[cur_state]

            # dirichlet noise for root node
            if cur_state == state:
                noises = np.random.dirichlet([0.03] * 9)
                probs = [0.75 * prob + 0.25 * noise for prob, noise in zip(probs, noises)]

            # calculates the upper confidence bound for each action
            score = [value + self.c_puct * prob * total_sqrt / (1 + count)
                     for value, prob, count in zip(values_avg, probs, counts)]

            # set invalid actions to -np.inf so they wont get chosen
            invalid_actions = set(range(9)) - set(s.get_action_space(cur_state))
            for invalid_action in invalid_actions:
                score[invalid_action] = -np.inf

            action = np.argmax(score)
            actions.append(action)
            cur_state, won = s.move(cur_state, action, cur_player)

            # the value if 0 if the game is a draw
            # and -1 if a player has won
            if won is not None:
                value = 0 if not won else -1

            cur_player = s.next_player(cur_player)

        return cur_state, cur_player, states, actions, value

    def expand(self, state, player, net):
        # expand a leaf node
        self.visit_counts[state] = [0] * 9
        self.values[state] = [0] * 9
        self.values_avg[state] = [0] * 9

        values, probs = self.evaluate(state, player, net)

        self.probs[state] = probs

        return values

    def evaluate(self, state, player, net):
        # get probabilities and value prediction from neural net
        X = s.state_to_batch(state, player)
        X = torch.FloatTensor([X])

        logits_v, values_v = net(X)
        probs_v = F.softmax(logits_v, dim=1)
        values = values_v.data.cpu().numpy()[:, 0].squeeze()
        probs = probs_v.data.cpu().numpy().squeeze()

        return values, probs

    def backpropagate(self, states, actions, value):
        # backpropagate the value
        cur_value = -value
        for state, action in zip(states[::-1], actions[::-1]):
            self.visit_counts[state][action] += 1
            self.values[state][action] += cur_value
            self.values_avg[state][action] = self.values[state][action] / self.visit_counts[state][action]
            cur_value = -cur_value

    def get_policy(self, state):
        # returns the pi probabilities from mcts (no temperature parameter)
        visit_counts = self.visit_counts[state]
        sum_visits = sum(visit_counts)
        probs = [visit_count / sum_visits for visit_count in visit_counts]
        return probs

    def monte_carlo_tree_search(self, state, player, net, iterations):
        # main mcts loop
        for _ in range(iterations):
            leaf_state, leaf_player, states, actions, value = self.find_leaf(state, player)
            if value is None:
                value = self.expand(leaf_state, leaf_player, net)
            self.backpropagate(states, actions, value)
        return self.get_policy(state)

def evaluate(nets, mcts_stores, rounds):
    # games between the trained net and the current best net
    results = {None: 0, 1: 0, 2: 0}
    mcts_stores = [MCTS(), MCTS()]

    for _ in range(rounds):
        player_won = play_game(nets, mcts_stores)
        print(player_won)
        results[player_won] += 1

    return results[1] / (results[1] + results[2])

def play_game(nets, mcts_stores, replay_buffer = None):
    # game between two nets
    won = None
    cur_player = s.get_random_player()
    state = s.init()
    player_won = None
    # store history to train the network
    game_history = []

    while won is None:
        probs = mcts_stores[cur_player-1].monte_carlo_tree_search(state, cur_player, nets[cur_player-1], 100)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(9, p=probs)
        state, won = s.move(state, action, cur_player)
        if won:
            player_won = cur_player
        cur_player = s.next_player(cur_player)

    result = won * 1

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append((state, cur_player, probs, result))
            result = -result

    return player_won

best_net = Net(model.OBS_SHAPE, 9)
apprentice = copy.deepcopy(best_net)
optimizer = optim.SGD(apprentice.parameters(), lr=0.1, momentum=0.9)
mcts = MCTS()
replay_buffer = collections.deque(maxlen=5000)

model_changes = 0
i = 0

while True:

    # self play
    player_won = play_game([best_net, best_net], [mcts, mcts], replay_buffer)
    i += 1

    print(i, len(replay_buffer))

    if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
        continue

    # training
    for _ in range(TRAIN_ROUNDS):
        batch = random.sample(replay_buffer, BATCH_SIZE)
        batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
        # convert states to neural input
        batch_states_lists = [s.state_to_batch(state, player) for state, player in zip(batch_states, batch_who_moves)]
        states_v = torch.FloatTensor(batch_states_lists)

        optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs)
        values_v = torch.FloatTensor(batch_values)
        out_logits_v, out_values_v = apprentice(states_v)

        # calculate loss and train the net
        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + loss_value_v
        loss_v.backward()
        optimizer.step()

    # play games between the current best net and the trained model
    # and sync wheights if the trained model wins 80% of the games (no draws)
    if i % EVALUATE_EVERY_STEP == 0:
        win_rate = evaluate([apprentice, best_net], [MCTS(), MCTS()], 100)
        print(win_rate)
        
        if win_rate > 0.8:
            print("syncing weights")
            best_net = copy.deepcopy(apprentice)
            mcts = MCTS()
            model_changes += 1
            path = os.getcwd()
            torch.save(best_net, os.path.join(path, f"model{model_changes}"))