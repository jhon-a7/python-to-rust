import numpy as np
from random import shuffle

chance_nodes = {'bc','xx', 'xbc', 'brc', 'xbrc'}

def rank(cards):
    ranks = {
        'KK': 1,
        'QQ': 2,
        'JJ': 3,
        'KQ': 4, 'QK': 4,
        'KJ': 5, 'JK': 5,
        'QJ': 6, 'JQ': 6
    }
    return ranks[cards]

def is_terminal(history):
    return history[-1:] == 'f' or ('d' in history and history.split('d')[1] in chance_nodes)

def is_chance_node(history):
    return history in chance_nodes

def terminal_util(history, card_player, card_opponent, card_flop):
    '''Return player's utility when we arrive at a terminal node'''
    ante = 1
    # ante is 1$, bet is 2$, raise is 4$
    payoffs = {'xx':0, 'bf':0, 'xbf':0, 'brf':2, 'xbrf':2, 'bc':2, 'xbc':2, 'brc':4, 'xbrc':4}
    if 'd' not in history:  # if there was a fold pre-flop
        return ante + payoffs[history]
    else:  # if there was a fold post-flop or if we went to showdown 
        payoffs = {'xx':0, 'bf':0, 'xbf':0, 'brf':2, 'xbrf':2, 'bc':2, 'xbc':2, 'brc':4, 'xbrc':4}
        preflop, flop = history.split('d')
        pot = ante + payoffs[preflop] + payoffs[flop]
        if history[-1:] == 'f':
            return pot
        else:  # showdown
            # hand_player = card_str(card_player) + card_str(card_flop)
            hand_player = card_player + card_flop

            # hand_opponent = card_str(card_opponent) + card_str(card_flop)
            hand_opponent = card_opponent + card_flop

            if rank(hand_player) < rank(hand_opponent):
                return pot
            elif rank(hand_player) > rank(hand_opponent):
                return -pot
            else:
                return 0

def valid_actions(history):
    '''card dealt: d, check: x, fold: f, call: c, bet: b, raise: r'''
    if history[-1:] == '' or history[-1] == 'd' or history[-1] == 'x':
        return ['x', 'b']
    elif history[-1] == 'b':
        return ['f', 'c', 'r']
    elif history[-1] == 'r':
        return ['f', 'c']

def get_active_player(history):
    if 'd' not in history:
        return len(history) % 2
    else:  # after flop is dealt player with index 0 is the first to play
        return len(history.split('d')[1]) % 2

class Leduc:
    '''
    Implementation of a poker bot that learns a strategy for heads up Leduc Poker 
    using Counterfactual Regret Minimization (CFR), specifically the variant known 
    as Chance Sampling CFR. The theoretical grounds of the algorithm can be found
    in An Introduction to Counterfactual Regret Minimization, by Todd W. Neller
    and Marc Lanctot.
    '''
    def __init__(self):
        self.deck = np.array(['K', 'K', 'Q', 'Q', 'J', 'J'])

    def cfr(self, i_map, history="", pr_1=1, pr_2=1):
  
        curr_player = get_active_player(history)
        card_player = self.deck[curr_player] 
        card_opponent = self.deck[1-curr_player] 

        if is_terminal(history):
            return terminal_util(history, card_player, card_opponent, self.deck[2])

        if is_chance_node(history):  # if first round of betting is finished
            next_history = history + 'd' # add dealt card to history
            if history in {'xbc', 'brc'}:
                return -1 * self.cfr(i_map, next_history, pr_1, pr_2)
            else:
                return self.cfr(i_map, next_history, pr_1, pr_2)  # here we don't multiply by -1 because next player = current player = 0

        info_set = self.get_info_set(i_map, history, card_player, self.deck[2])

        strategy = info_set.strategy

        if curr_player == 0:
            info_set.reach_pr += pr_1
        else:
            info_set.reach_pr += pr_2

        val_act = valid_actions(history)
        action_utils = np.zeros(info_set.n_actions)
        for i, action in enumerate(val_act):
            next_history = history + action

            if curr_player == 0: 
                action_utils[i] = -1 * self.cfr(i_map, next_history,
                                            pr_1 * strategy[i], pr_2)  # utility of current player equals - utility of player corresponding to next history
            else:
                action_utils[i] = -1 * self.cfr(i_map, next_history,
                                            pr_1, pr_2 * strategy[i])

        util = sum(action_utils * strategy) # expected value: sum of outcome for each action times the probability that said action is chosen
        regrets = action_utils - util  # entry a stores the difference between the value of always choosing action a and the expected value for given strategy

        if curr_player == 0:
            info_set.regret_sum += pr_2 * regrets  # eq.(4) in "Introduction to Counterfactual Regret Minimisation"
        else:
            info_set.regret_sum += pr_1 * regrets

        return util
    
    def get_info_set(self, i_map, history, card, flop):
        """
        Retrieve information set from dictionary
        """
        if 'd' in history:
            key = card + flop + " " + history

        else:
            key = card + " " +  history

        if key in i_map:
            return i_map[key]
            
        n_actions = 3 if history[-1:] == 'b' else 2

        info_set = InformationSet(key, n_actions)
        i_map[key] = info_set
        return i_map[key]

class InformationSet():
    '''
    Each instance of this class represents an information set: a state in 
    the CFR tree that is keyed by the public hands known to the player (either 
    their pocket hand or their pocket hand and the flop) and the betting history 
    so far and that stores their current regrets and strategy
    '''
    def __init__(self, key, n_actions):
        self.key = key
        self.n_actions = n_actions
        self.regret_sum = np.zeros(self.n_actions)  # each entry stores the accumulated regret of not having takem an action for each possible action
        self.strategy_sum = np.zeros(self.n_actions)  # sum_{t=1}^T \pi_i^{\sigma^t}(I)*\sigma^t(I)(a); used to compute average strategy
        self.strategy = np.repeat(1/self.n_actions, self.n_actions)  # strategy for given information set 
        self.reach_pr = 0  # sum of probabilities of reaching information set over all histories in the information set for a single iteration 
        self.reach_pr_sum = 0  # sum of reach probabilities over all iterations; used to compute average strategy

    def update_strategy(self):
        self.strategy_sum += self.reach_pr * self.strategy
        self.reach_pr_sum += self.reach_pr
        self.strategy = self.get_strategy()
        self.reach_pr = 0

    def get_strategy(self):
        strategy = self.to_nonnegative(self.regret_sum)
        total = sum(strategy)
        if total > 0:
            strategy /= total
            return strategy
        return np.repeat(1/self.n_actions, self.n_actions)

    def get_average_strategy(self):
        strategy = self.strategy_sum
        total = sum(strategy)
        if total > 0:
            strategy /= total
            return strategy
        return np.repeat(1/self.n_actions, self.n_actions)

    def __str__(self):
        strategies = ['{:03.2f}'.format(x)
                      for x in self.get_average_strategy()]
        return '{} {}'.format(self.key.ljust(6), strategies)

    def to_nonnegative(self, val):
        return np.where(val > 0, val, 0)

def display_results(ev, i_map):
    print('player 1 expected value: {}'.format(ev))
    print('player 2 expected value: {}'.format(-1 * ev))

    print()
    print('player 1 strategies:')
    sorted_items = sorted(i_map.items(), key=lambda x: x[0])
    for _, v in filter(lambda x: get_active_player(x[0]) == 0, sorted_items):
        print(v)
    print()
    print('player 2 strategies:')
    for _, v in filter(lambda x: get_active_player(x[0]) == 1, sorted_items):
        print(v)

def train(n_iterations = 10000):
    leduc = Leduc()
    i_map = {}  # map of with histories and cards as keys and their corresponding information sets as values
    expected_game_value = 0
    for _ in range(n_iterations):
        shuffle(leduc.deck)
        expected_game_value += leduc.cfr(i_map)

        for key in i_map:
            i_map[key].update_strategy()  # after each iteration, update strategy for each information set according to updated regrets

    expected_game_value /= n_iterations
    print(display_results(expected_game_value, i_map))
    return expected_game_value, i_map

if __name__ == "__main__":
    train()
