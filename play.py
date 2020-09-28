#!/usr/bin/env python3
import os
import time
import random
import argparse
import collections
import copy
import numpy as np

import state as s
import model, mcts

import torch
import torch.optim as optim
import torch.nn.functional as F


PLAY_EPISODES = 1  #25
MCTS_SEARCHES = 10
MCTS_BATCH_SIZE = 8
REPLAY_BUFFER = 5000 # 30000
LEARNING_RATE = 0.1
BATCH_SIZE = 256
TRAIN_ROUNDS = 10
MIN_REPLAY_TO_TRAIN = 2000 #10000

BEST_NET_WIN_RATIO = 0.60

EVALUATE_EVERY_STEP = 100
EVALUATION_ROUNDS = 20
STEPS_BEFORE_TAU_0 = 10

device = torch.device("cpu")

path = os.getcwd()
model_path = os.path.join(path, f"model4")
net = torch.load(model_path)
mcts = mcts.MCTS()

while True:

    won = None
    cur_player = s.get_random_player()
    cur_state = s.init()

    print(s.decode(cur_state))

    while won is None:

        print(f"Player: {cur_player}")

        if cur_player == 1:
            mcts.search_batch(20, 16, cur_state, cur_player, net, device=device)
            probs, _ = mcts.get_policy_value(cur_state, tau=0)
            action = np.random.choice(9, p=probs)
        else:
            action = int(input("Field: "))

        cur_state, won = s.move(cur_state, action, cur_player)
        cur_player = s.next_player(cur_player)

        print(s.decode(cur_state))