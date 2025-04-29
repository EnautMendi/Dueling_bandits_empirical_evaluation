import pandas as pd
from tqdm import tqdm
from strategies import *
from pathlib import Path

strategy_classes = {
    'thompson': ThompsonSamplingBandit,
    'double_thompson': DoubleThompsonBandit,
    'epsilon_greedy': EpsilonGreedyBandit,
    'ucb': UCBBandit,
    'random': BaseDuelingBandit,
    'max_win_rate': MaxWinRateBandit,
    'RMED': RMEDBandit,
    'halving_battle': HalvingBattleBandit
}

def compare_strategies(num_arms, num_runs, confidence_threshold, max_iterations, logs, condorcet, savepath):
    strategies = strategy_classes.keys()
    results_df = pd.DataFrame(columns=['Run', 'Strategy', 'Correct', 'Position', 'Exec_Time', 'Num_Iterations', 'Regret_Loss', 'True_rank', 'Obtained_rank'])

    for strategy in tqdm(strategies, desc='Strategies running...'):

        if logs:
            Path(f'{savepath}/logs').mkdir(parents=True, exist_ok=True)
            log_filename = f"{savepath}/logs/{strategy}_{num_arms}.txt"
            log_file = open(log_filename, "a")
            log_file.write("Run,Iteration,Position\n")

        else:
            log_file=None

        for run in range(num_runs):
            if condorcet:
                random_strengths = 0.5 * np.random.rand(num_arms-1)
                condorcet_winner = 1
                random_index = np.random.randint(0, len(random_strengths) + 1)
                true_strengths = np.insert(random_strengths, random_index, condorcet_winner)
            else:
                true_strengths = np.random.rand(num_arms)

            true_ranking = np.argsort(-true_strengths)
            bandit = strategy_classes[strategy](num_arms, confidence_threshold=confidence_threshold)
            best_arm, is_correct, exec_time, iterations, position, regret, ranking, win_rates = bandit.run(true_strengths, max_iterations, strategy, run, log_file)

            true_dict = {f"arm_{pos}": true_strengths[pos] for pos in true_ranking}
            obtained_dict = {f"arm_{pos}": win_rates[pos] for pos in ranking}

            new_row = {'Run': run, 'Strategy':strategy, 'Correct': is_correct, 'Position': position, 'Exec_Time': exec_time, 'Num_Iterations': iterations,'Regret_Loss': regret, 'True_rank': [true_dict], 'Obtained_rank': [obtained_dict]}
            new_df = pd.DataFrame.from_dict(new_row, orient='index').T
            results_df = pd.concat([results_df, new_df], ignore_index=True)

    Path(f'{savepath}/results').mkdir(parents=True, exist_ok=True)
    results_df.to_csv(f'{savepath}/results/results_{num_arms}_{confidence_threshold}_{max_iterations}.csv')


if __name__ == '__main__':
    arms = [5, 10, 20, 50, 100, 200]
    for n_arms in arms:
        print('\n----------------------------------------------------------------------------')
        print(f'Experiment with {n_arms} arms & {n_arms*100} iterations running!')
        print('----------------------------------------------------------------------------\n')

        compare_strategies(num_arms=n_arms, num_runs=30, confidence_threshold=0.99, max_iterations=n_arms*100, logs=True, condorcet=False, savepath='full_experiment')
        compare_strategies(num_arms=n_arms, num_runs=30, confidence_threshold=0.99, max_iterations=n_arms*100, logs=True, condorcet=True, savepath='full_experiment_condorcet')

