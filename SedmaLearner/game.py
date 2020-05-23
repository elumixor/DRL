import random

jack = 11
queen = 12
king = 13
ace = 14

ranks = ['7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['Heart', 'Diamond', 'Spade', 'Club']
players = ['A', 'B', 'C', 'D']

ix2string = {c: ranks[i] for i, c in enumerate(range(len(ranks)))}
player2ix = {p: i for i, p in enumerate(players)}
suit2ix = {p: i for i, p in enumerate(suits)}
rank2ix = {p: i for i, p in enumerate(ranks)}


def card_value(card):
    _, rank = card
    if rank == '10' or rank == 'A':
        return 10
    else:
        return 0


suits_cards = [[(s, r) for r in ranks] for s in suits]
sorted_deck = [item for sublist in suits_cards for item in sublist]
card2ix = {p: i for i, p in enumerate(sorted_deck)}


def get_deck():
    r = sorted_deck[:]
    random.shuffle(r)
    return r


# We will need this to cycle through players as they play
def cycle(arr, count, start=0):
    def generate():
        i = 0
        while True:
            for connection in arr:
                if i >= start:
                    yield connection
                if i >= count + start - 1:
                    return
                i += 1

    return list(generate())


class Player:
    def __init__(self, name, cards):
        self.hand = cards
        self.name = name

    def play_card(self, trick, leader):
        pass

    def take_cards(self, new_cards):
        self.hand += new_cards


class FirstHandPlayer(Player):
    def __init__(self, name, cards):
        super().__init__(name, cards)

    def play_card(self, trick, leader):
        return self.hand[0]


class RandomHandPlayer(Player):
    def __init__(self, name, cards):
        super().__init__(name, cards)

    def play_card(self, trick, leader):
        return random.choice(self.hand)


class Game:
    def __init__(self, team_ac_player, team_bd_player):
        self.deck = get_deck()

        cards_a = self.get_cards(4)
        cards_b = self.get_cards(4)
        cards_c = self.get_cards(4)
        cards_d = self.get_cards(4)

        self.A = team_ac_player('A', cards_a)
        self.B = team_bd_player('B', cards_b)
        self.C = team_ac_player('C', cards_c)
        self.D = team_bd_player('D', cards_d)

        self.AC_cards = []
        self.BD_cards = []

        self.players = [self.A, self.B, self.C, self.D]
        self.team = {self.A: self.AC_cards, self.C: self.AC_cards, self.B: self.BD_cards, self.D: self.BD_cards}

    def get_cards(self, number):
        picked = random.sample(self.deck, k=number)
        self.deck = [c for c in self.deck if c not in picked]
        return picked

    def play(self, silent=False):
        for r in range(8):  # for each round
            trick = []
            for p in self.players:
                card = p.play_card(trick, self.players[0].name)
                p.hand = [c for c in p.hand if c != card]
                trick.append(card)

            winner_index = self.get_winner(trick)
            winner = self.players[winner_index]

            winner_tricks = self.team[winner]
            winner_tricks += trick

            if not silent:
                print(f'Player {winner.name} has won the trick. ({trick})')

            self.players = cycle(self.players, 4, winner_index)

            if r < 4:
                for p in self.players:
                    p.take_cards(self.get_cards(1))

        AC_points = sum([card_value(c) for c in self.AC_cards])
        BD_points = sum([card_value(c) for c in self.BD_cards])

        if winner_tricks == self.AC_cards:
            AC_points += 10
        else:
            BD_points += 10

        if AC_points > BD_points:
            winning_team = 'AC'

            if len(self.BD_cards) == 0:
                points = 3
            elif BD_points == 0:
                points = 2
            else:
                points = 1

        else:
            winning_team = 'BD'

            if len(self.AC_cards) == 0:
                points = 3
            elif AC_points == 0:
                points = 2
            else:
                points = 1

        return winning_team, points, (self.AC_cards, self.BD_cards)

    @staticmethod
    def get_winner(trick):
        for i in reversed(range(len(trick))):
            if trick[i][1] == trick[0][1] or trick[i][1] == '7':
                return i


game = Game(FirstHandPlayer, FirstHandPlayer)
