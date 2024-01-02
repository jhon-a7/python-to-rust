use rand::seq::SliceRandom;
use std::collections::HashMap;

const CHANCE_NODES: [&str; 5] = ["bc", "xx", "xbc", "brc", "xbrc"];

fn rank(cards: &str) -> i32 {
    let ranks = [
        ("KK", 1),
        ("QQ", 2),
        ("JJ", 3),
        ("KQ", 4),
        ("QK", 4),
        ("KJ", 5),
        ("JK", 5),
        ("QJ", 6),
        ("JQ", 6),
    ];

    for &(pattern, value) in ranks.iter() {
        if pattern == cards {
            return value;
        }
    }

    0
}

fn is_terminal(history: &str) -> bool {
    history.ends_with('f') || (history.contains('d') && CHANCE_NODES.contains(&&history.split('d').collect::<Vec<&str>>()[1]))
}

fn is_chance_node(history: &str) -> bool {
    CHANCE_NODES.contains(&history)
}

fn terminal_util(history: &str, card_player: &str, card_opponent: &str, card_flop: &str) -> i32 {
    let ante = 1;
    let payoffs = [
        ("xx", 0),
        ("bf", 0),
        ("xbf", 0),
        ("brf", 2),
        ("xbrf", 2),
        ("bc", 2),
        ("xbc", 2),
        ("brc", 4),
        ("xbrc", 4),
    ];

    if !history.contains('d') {
        return ante + payoffs.iter().find(|&&(pattern, _)| pattern == history).unwrap().1;
    } else {
        let (preflop, flop) = history.split_once('d').unwrap();
        let pot = ante + payoffs.iter().find(|&&(pattern, _)| pattern == preflop).unwrap().1
            + payoffs.iter().find(|&&(pattern, _)| pattern == flop).unwrap().1;

        if history.ends_with('f') {
            return pot;
        } else {
            let hand_player = format!("{}{}", card_player, card_flop);
            let hand_opponent = format!("{}{}", card_opponent, card_flop);

            if rank(&hand_player) < rank(&hand_opponent) {
                return pot;
            } else if rank(&hand_player) > rank(&hand_opponent) {
                return -pot;
            } else {
                return 0;
            }
        }
    }
}

fn valid_actions(history: &str) -> Vec<&str> {
    if history.is_empty() || history.ends_with('d') || history.ends_with('x') {
        vec!["x", "b"]
    } else if history.ends_with('b') {
        vec!["f", "c", "r"]
    } else if history.ends_with('r') {
        vec!["f", "c"]
    } else {
        vec![]
    }
}

fn get_active_player(history: &str) -> usize {
    if !history.contains('d') {
        history.len() % 2
    } else {
        history.split_once('d').unwrap().1.len() % 2
    }
}

struct Leduc {
    deck: Vec<&'static str>,
}

impl Leduc {
    fn new() -> Self {
        Leduc {
            deck: vec!["K", "K", "Q", "Q", "J", "J"],
        }
    }

    fn cfr(&mut self, i_map: &mut HashMap<String, InformationSet>, history: &str, pr_1: i32, pr_2: i32) -> i32 {
        let curr_player = get_active_player(history);
        let card_player = self.deck[curr_player];
        let card_opponent = self.deck[1 - curr_player];

        if is_terminal(history) {
            return terminal_util(history, card_player, card_opponent, self.deck[2]);
        }

        if is_chance_node(history) {
            let next_history = format!("{}d", history);
            if CHANCE_NODES.contains(&&history[..3]) {
                return -1 * self.cfr(i_map, &next_history, pr_1, pr_2);
            } else {
                return self.cfr(i_map, &next_history, pr_1, pr_2);
            }
        }

        let info_set = self.get_info_set(i_map, history, card_player, self.deck[2]);

        let strategy = info_set.strategy;

        if curr_player == 0 {
            info_set.reach_pr += pr_1;
        } else {
            info_set.reach_pr += pr_2;
        }

        let val_act = valid_actions(history);
        let mut action_utils = vec![0; info_set.n_actions];
        for (i, &action) in val_act.iter().enumerate() {
            let next_history = format!("{}{}", history, action);

            if curr_player == 0 {
                action_utils[i] =
                    -1 * self.cfr(i_map, &next_history, pr_1 * strategy[i], pr_2); // utility of current player equals - utility of player corresponding to next history
            } else {
                action_utils[i] =
                    -1 * self.cfr(i_map, &next_history, pr_1, pr_2 * strategy[i]);
            }
        }

        let util = action_utils.iter().zip(&strategy).fold(0, |acc, (&action_util, &prob)| {
            acc + action_util * prob
        });
        let regrets: Vec<i32> = action_utils.iter().zip(&strategy).map(|(&action_util, &prob)| {
            action_util - util
        }).collect();

        if curr_player == 0 {
            info_set.regret_sum
                .iter_mut()
                .zip(&regrets)
                .for_each(|(regret_sum, &regret)| {
                    *regret_sum += pr_2 * regret;
                });
        } else {
            info_set.regret_sum
                .iter_mut()
                .zip(&regrets)
                .for_each(|(regret_sum, &regret)| {
                    *regret_sum += pr_1 * regret;
                });
        }

        util
    }

    fn get_info_set(&self, i_map: &mut HashMap<String, InformationSet>, history: &str, card: &str, flop: &str) -> InformationSet {
        if history.contains('d') {
            let key = format!("{}{} {}", card, flop, history);
            if let Some(info_set) = i_map.get(&key) {
                return info_set.clone();
            }
        } else {
            let key = format!("{} {}", card, history);
            if let Some(info_set) = i_map.get(&key) {
                return info_set.clone();
            }
        }

        let n_actions = if history.ends_with('b') { 3 } else { 2 };
        let info_set = InformationSet::new(key.clone(), n_actions);
        i_map.insert(key, info_set.clone());
        info_set
    }
}

struct InformationSet {
    key: String,
    n_actions: usize,
    regret_sum: Vec<i32>,
    strategy_sum: Vec<i32>,
    strategy: Vec<f32>,
    reach_pr: i32,
}

impl InformationSet {
    fn new(key: String, n_actions: usize) -> Self {
        InformationSet {
            key,
            n_actions,
            regret_sum: vec![0; n_actions],
            strategy_sum: vec![0; n_actions],
            strategy: vec![1.0 / n_actions as f32; n_actions],
            reach_pr: 0,
        }
    }

    fn update_strategy(&mut self) {
        self.strategy_sum.iter_mut().zip(&self.strategy).for_each(|(strategy_sum, &strategy)| {
            *strategy_sum += self.reach_pr * strategy;
        });

        self.reach_pr_sum += self.reach_pr;
        self.strategy = self.get_strategy();
        self.reach_pr = 0;
    }

    fn get_strategy(&self) -> Vec<f32> {
        let strategy = self.to_nonnegative(&self.regret_sum);
        let total = strategy.iter().sum::<i32>() as f32;
        if total > 0.0 {
            return strategy.iter().map(|&regret| regret as f32 / total).collect();
        }

        vec![1.0 / self.n_actions as f32; self.n_actions]
    }

    fn get_average_strategy(&self) -> Vec<f32> {
        let strategy = &self.strategy_sum;
        let total = strategy.iter().sum::<i32>() as f32;
        if total > 0.0 {
            return strategy.iter().map(|&strategy_sum| strategy_sum as f32 / total).collect();
        }

        vec![1.0 / self.n_actions as f32; self.n_actions]
    }

    fn to_nonnegative(&self, val: &[i32]) -> Vec<f32> {
        val.iter().map(|&v| if v > 0 { v as f32 } else { 0.0 }).collect()
    }
}

fn display_results(ev: i32, i_map: &HashMap<String, InformationSet>) {
    println!("player 1 expected value: {}", ev);
    println!("player 2 expected value: {}", -1 * ev);

    println!("\nplayer 1 strategies:");
    let mut sorted_items: Vec<(&String, &InformationSet)> = i_map.iter().filter(|&(k, _)| get_active_player(k) == 0).collect();
    sorted_items.sort_by(|(a, _), (b, _)| a.cmp(b));
    for (_, v) in sorted_items.iter() {
        println!("{}", v);
    }

    println!("\nplayer 2 strategies:");
    let sorted_items: Vec<(&String, &InformationSet)> = i_map.iter().filter(|&(k, _)| get_active_player(k) == 1).collect();
    for (_, v) in sorted_items.iter() {
        println!("{}", v);
    }
}

fn train(n_iterations: usize) -> (i32, HashMap<String, InformationSet>) {
    let mut leduc = Leduc::new();
    let mut i_map = HashMap::new();
    let mut expected_game_value = 0;
    
    for _ in 0..n_iterations {
        leduc.deck.shuffle(&mut rand::thread_rng());
        expected_game_value += leduc.cfr(&mut i_map, "", 1, 1);

        for info_set in i_map.values_mut() {
            info_set.update_strategy();
        }
    }

    expected_game_value /= n_iterations as i32;
    display_results(expected_game_value, &i_map);

    (expected_game_value, i_map)
}

fn main() {
    train(10000);
}
