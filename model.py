import collections
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F

import state as s

import mcts

# torch.manual_seed(1)

OBS_SHAPE = (3, 3, 3)
NUM_FILTERS = 64

class Net(nn.Module):
    def __init__(self, input_shape, actions_n):
        super(Net, self).__init__()

        self.conv_in = nn.Sequential(
            nn.Conv2d(input_shape[0], NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        # layers with residual
        self.conv_1 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, NUM_FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(NUM_FILTERS),
            nn.LeakyReLU()
        )

        body_out_shape = (NUM_FILTERS, ) + input_shape[1:]

        # value head
        self.conv_val = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )
        conv_val_size = self._get_conv_val_size(body_out_shape)
        self.value = nn.Sequential(
            nn.Linear(conv_val_size, 20),
            nn.LeakyReLU(),
            nn.Linear(20, 1),
            nn.Tanh()
        )

        # policy head
        self.conv_policy = nn.Sequential(
            nn.Conv2d(NUM_FILTERS, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.LeakyReLU()
        )
        conv_policy_size = self._get_conv_policy_size(body_out_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_policy_size, actions_n)
        )

    def _get_conv_val_size(self, shape):
        o = self.conv_val(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def _get_conv_policy_size(self, shape):
        o = self.conv_policy(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        v = self.conv_in(x)
        v = v + self.conv_1(v)
        v = v + self.conv_2(v)
        v = v + self.conv_3(v)
        v = v + self.conv_4(v)
        v = v + self.conv_5(v)
        val = self.conv_val(v)
        val = self.value(val.view(batch_size, -1))
        pol = self.conv_policy(v)
        pol = self.policy(pol.view(batch_size, -1))
        return pol, val

def play_game(mcts_stores, replay_buffer, net1, net2, steps_before_tau_0, mcts_searches, mcts_batch_size,
              net1_plays_first=None, device="cpu"):
    """
    Play one single game, memorizing transitions into the replay buffer
    :param mcts_stores: could be None or single MCTS or two MCTSes for individual net
    :param replay_buffer: queue with (state, probs, values), if None, nothing is stored
    :param net1: player1
    :param net2: player2
    :return: value for the game in respect to player1 (+1 if p1 won, -1 if lost, 0 if draw)
    """

    if mcts_stores is None:
        mcts_stores = [mcts.MCTS(), mcts.MCTS()]
    elif isinstance(mcts_stores, mcts.MCTS):
        mcts_stores = [mcts_stores, mcts_stores]

    state = s.init()
    nets = [net1, net2]
    if net1_plays_first is None:
        cur_player = s.get_random_player()
    else:
        cur_player = 1 if net1_plays_first else 2
    step = 0
    tau = 1 if steps_before_tau_0 > 0 else 0
    game_history = []

    result = None
    net1_result = None

    while result is None:
        mcts_stores[cur_player - 1].search_batch(mcts_searches, mcts_batch_size, state,
                                             cur_player, nets[cur_player - 1], device=device)
        probs, _ = mcts_stores[cur_player - 1].get_policy_value(state, tau=tau)
        game_history.append((state, cur_player, probs))
        action = np.random.choice(9, p=probs)
        if action not in s.get_action_space(state):
            print("Impossible action selected")
        state, won = s.move(state, action, cur_player)
        if won:
            result = 1
            net1_result = 1 if cur_player == 1 else -1
            break
        cur_player = s.next_player(cur_player)
        # check the draw case
        if len(s.get_action_space(state)) == 0:
            result = 0
            net1_result = 0
            break
        step += 1
        if step >= steps_before_tau_0:
            tau = 0

    if replay_buffer is not None:
        for state, cur_player, probs in reversed(game_history):
            replay_buffer.append((state, cur_player, probs, result))
            result = -result

    return net1_result, step