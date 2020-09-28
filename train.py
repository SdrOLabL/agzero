#!/usr/bin/env python3
import os
import time
import random
import argparse
import collections
import copy

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


def evaluate(net1, net2, rounds, device="cpu"):
    n1_win, n2_win = 0, 0
    mcts_stores = [mcts.MCTS(), mcts.MCTS()]

    for r_idx in range(rounds):
        r, _ = model.play_game(mcts_stores=mcts_stores, replay_buffer=None, net1=net1, net2=net2,
                               steps_before_tau_0=0, mcts_searches=20, mcts_batch_size=16,
                               device=device)
        if r < -0.5:
            n2_win += 1
        elif r > 0.5:
            n1_win += 1

    if (n1_win + n2_win) == 0: return 0
    return n1_win / (n1_win + n2_win)

device = torch.device("cpu")

net = model.Net(model.OBS_SHAPE, 9).to(device)
best_net = copy.deepcopy(net)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9)

replay_buffer = collections.deque(maxlen=REPLAY_BUFFER)
mcts_store = mcts.MCTS()
step_idx = 0
best_idx = 0

while True:
    prev_nodes = len(mcts_store)
    game_steps = 0
    for _ in range(PLAY_EPISODES):
        _, steps = model.play_game(mcts_store, replay_buffer, best_net, best_net,
                                    steps_before_tau_0=STEPS_BEFORE_TAU_0, mcts_searches=MCTS_SEARCHES,
                                    mcts_batch_size=MCTS_BATCH_SIZE, device=device)
        game_steps += steps
    game_nodes = len(mcts_store) - prev_nodes
    step_idx += 1

    print(step_idx, len(replay_buffer))

    if len(replay_buffer) < MIN_REPLAY_TO_TRAIN:
        continue

    # train
    for _ in range(TRAIN_ROUNDS):
        batch = random.sample(replay_buffer, BATCH_SIZE)
        batch_states, batch_who_moves, batch_probs, batch_values = zip(*batch)
        batch_states_lists = [s.state_to_batch(state, player) for state, player in zip(batch_states, batch_who_moves)]
        states_v = torch.FloatTensor(batch_states_lists)

        optimizer.zero_grad()
        probs_v = torch.FloatTensor(batch_probs)
        values_v = torch.FloatTensor(batch_values)
        out_logits_v, out_values_v = net(states_v)

        loss_value_v = F.mse_loss(out_values_v.squeeze(-1), values_v)
        loss_policy_v = -F.log_softmax(out_logits_v, dim=1) * probs_v
        loss_policy_v = loss_policy_v.sum(dim=1).mean()

        loss_v = loss_policy_v + loss_value_v
        loss_v.backward()
        optimizer.step()

    # evaluate net
    if step_idx % EVALUATE_EVERY_STEP == 0:
        win_ratio = evaluate(net, best_net, rounds=EVALUATION_ROUNDS, device=device)
        print(f"Net evaluated, win ratio = {win_ratio}")
        if win_ratio > BEST_NET_WIN_RATIO:
            print("Net is better than cur best, sync")
            best_idx += 1
            best_net = copy.deepcopy(net)
            path = os.getcwd()
            torch.save(best_net, os.path.join(path, f"model"))
            mcts_store.clear()