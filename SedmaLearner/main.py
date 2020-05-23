import random

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.distributions import Categorical

from SedmaLearner.game import Game, FirstHandPlayer, Player, player2ix, rank2ix, suit2ix, ranks, suits, \
    RandomHandPlayer, card2ix
from utils import one_hot, rewards_to_go, ModelSaver, running_average


class MyPlayer(Player):
    def __init__(self, name, cards):
        super().__init__(name, cards)
        self.ix = player2ix[name]
        self.probabilities = []
        self.actions = []
        self.played_cards = []

    def play_card(self, trick, leader):
        state = (self.ix, player2ix[leader], trick, self.hand, self.played_cards)
        p, card_index = get_action(state)
        card_index = min(card_index, len(self.hand) - 1)
        self.probabilities.append(p)
        self.actions.append(card_index)
        self.played_cards += trick + [self.hand[0]]
        return self.hand[0]


card_size = len(ranks) + len(suits)
hand_size = 4 * card_size
player_size = 4

obs_size = 2 * player_size + hand_size + hand_size + 32

num_actions = 4
hidden_1 = 256
hidden_2 = 64

# Actor maps state to actions' probabilities
actor = nn.Sequential(nn.Linear(obs_size, hidden_1),
                      nn.ReLU(),
                      nn.Linear(hidden_1, hidden_2),
                      nn.ReLU(),
                      nn.Linear(hidden_2, num_actions),
                      nn.Softmax(dim=1))

optimizer = optim.Adam(actor.parameters(), lr=0.01)
discounting = 0.99999

saver = ModelSaver({'actor': actor, 'optim_actor': optimizer}, './models/Sedma/VPG-3')


# saver.load()


# saver.load(ignore_errors=True)

def card2tensor(c):
    suit, rank = c
    suit = suit2ix[suit]
    rank = rank2ix[rank]
    return torch.cat((one_hot(suit, len(suits)), one_hot(rank, len(ranks))))


def state2tensor(state):
    player, leader, trick, hand, played_cards = state

    player = one_hot(player, 4)
    leader = one_hot(leader, 4)
    trick = hand2tensor(trick)
    hand = hand2tensor(hand)
    pc = torch.zeros(32)
    for c in played_cards:
        pc[card2ix[c]] = 1.

    state = torch.cat((player, leader, trick, hand, pc)).unsqueeze(0)
    return state


def hand2tensor(hand):
    hand_tensor = torch.zeros(4, card_size)
    for i, c in enumerate(hand):
        hand_tensor[i] = card2tensor(c)
    return hand_tensor.flatten()


def get_action(state):
    state = state2tensor(state)
    probabilities = actor(state)

    card_index = Categorical(probabilities).sample().item()
    return probabilities, card_index


def learn(probabilities, actions, reward):
    probabilities = torch.stack(probabilities).squeeze(1)
    rewards = torch.tensor([reward * (discounting ** (len(actions) - i)) for i in range(len(actions))])
    actions = torch.tensor(actions).long()

    return (-torch.log(probabilities[range(probabilities.shape[0]), actions]) * rewards).mean()


epochs = 1000
rollouts = 100
print_epochs = 10

rollout_points = []

for e in range(epochs):
    loss = 0
    points_ac = []

    # s = 0
    for rollout in range(rollouts):
        game = Game(RandomHandPlayer, MyPlayer)
        winner, points, _ = game.play(silent=True)
        # print(winner)

        if winner == 'AC':
            points_ac.append(points)
        else:
            points_ac.append(-points)
            points = -points

        for p in game.players:
            if isinstance(p, MyPlayer):
                loss += learn(p.probabilities, p.actions, points)

        game = Game(MyPlayer, RandomHandPlayer)
        winner, points, _ = game.play(silent=True)
        # print(winner)

        if winner == 'BD':
            points_ac.append(points)
        else:
            points_ac.append(-points)
            points = -points

        for p in game.players:
            if isinstance(p, MyPlayer):
                loss += learn(p.probabilities, p.actions, points)

    rollout_points.append(sum(points_ac))

    loss /= rollouts
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if e % print_epochs == 0:
        print(
            f'Epoch {e}: Mean AC total {rollouts} rollouts points in last {print_epochs} episodes: {np.mean(rollout_points[-print_epochs:])}')
        # plt.plot(points_ac)
        plt.plot(rollout_points)
        plt.plot(running_average(rollout_points))
        # plt.plot(points_bd)
        # plt.plot(running_average(points_bd))
        plt.show()
        saver.save()
