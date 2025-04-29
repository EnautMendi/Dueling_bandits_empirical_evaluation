import numpy as np
import random
from itertools import combinations
import time

class BaseDuelingBandit:
    def __init__(self, num_arms, confidence_threshold=0.95):
        self.num_arms = num_arms
        self.wins = np.ones(num_arms)
        self.losses = np.ones(num_arms)
        self.active_arms = list(range(num_arms))
        self.confidence_threshold = confidence_threshold

    def select_arms(self):
        return random.sample(self.active_arms, 2)

    def duel(self, arm1, arm2, true_strengths):
        prob_arm1_wins = true_strengths[arm1] / (true_strengths[arm1] + true_strengths[arm2])
        return arm1 if random.random() < prob_arm1_wins else arm2

    def update(self, winner, loser, t):
        self.wins[winner] += 1
        self.losses[loser] += 1

    def has_confident_winner(self):
        total_duels = self.wins + self.losses
        win_rates = self.wins / (total_duels + 1e-10) #Add a small epsilon to avoid possible division by 0
        best_arm = np.argmax(win_rates)
        best_confidence = win_rates[best_arm]
        return best_confidence >= self.confidence_threshold, best_arm

    def calculateLoss(self, true_strengths, iterations):
        max_score = max(true_strengths)*iterations
        actual_score = 0
        for arm in range(self.num_arms):
            actual_score += (self.wins[arm]-1)*true_strengths[arm]
        regret = max_score - actual_score
        return regret
    def run(self, true_strengths, max_iterations, strategy_name, run, log_file):
        start_time = time.time()
        correct_winner = np.argmax(true_strengths)
        for iterations in range(1, max_iterations + 1):
            if len(self.active_arms) < 2:
                confident, best_arm = self.has_confident_winner()
                break
            arm1, arm2 = self.select_arms()
            winner = self.duel(arm1, arm2, true_strengths)
            loser = arm2 if winner == arm1 else arm1
            self.update(winner, loser, iterations)

            win_rates = self.wins / (self.wins + self.losses)
            ranking = np.argsort(-win_rates)
            position = np.where(ranking == correct_winner)[0][0] + 1

            if log_file is not None:
                # log_file.write(f"{run},{iterations},{position},{win_rates},{ranking},{true_strengths}\n")
                log_file.write(f"{run},{iterations},{position}\n")


            confident, best_arm = self.has_confident_winner()
            if confident:
                break

        execution_time = time.time() - start_time
        win_rates = self.wins / (self.wins + self.losses)
        ranking = np.argsort(-win_rates)
        position = np.where(ranking == correct_winner)[0][0] + 1
        regret = self.calculateLoss(true_strengths, iterations)
        return best_arm, best_arm == correct_winner, execution_time, iterations, position, regret, ranking, win_rates

class ThompsonSamplingBandit(BaseDuelingBandit):
    def select_arms(self):
        samples = [np.random.beta(self.wins[i], self.losses[i]) for i in self.active_arms]
        return np.argsort(samples)[-2:]

class DoubleThompsonBandit(BaseDuelingBandit):
    def select_arms(self):
        samples1 = [np.random.beta(self.wins[i], self.losses[i]) for i in self.active_arms]
        best_arm1 = np.argmax(samples1)
        samples2 = [np.random.beta(self.wins[i], self.losses[i]) for i in self.active_arms]
        best_arm2 = np.argmax(samples2)
        return (best_arm1, best_arm2)# if best_arm1 != best_arm2 else (best_arm1, np.argsort(samples1)[-2])

class EpsilonGreedyBandit(BaseDuelingBandit):
    def select_arms(self):
        epsilon = 0.1
        if random.random() < epsilon:
            return random.sample(self.active_arms, 2)
        else:
            win_rates = self.wins / (self.wins + self.losses)
            return np.argsort(win_rates)[-2:]

class UCBBandit(BaseDuelingBandit):
    def select_arms(self):
        total_duels = self.wins + self.losses
        ucb_values = self.wins / total_duels + np.sqrt(2 * np.log(np.sum(total_duels)) / total_duels)
        return np.argsort(ucb_values)[-2:]

class MaxWinRateBandit(BaseDuelingBandit):
    def select_arms(self):
        win_rates = self.wins / (self.wins + self.losses)
        return np.argsort(win_rates)[-2:]

class RMEDBandit(BaseDuelingBandit):
    def __init__(self, num_arms, confidence_threshold=0.95, f_scale=0.3):
        super().__init__(num_arms, confidence_threshold)
        self.pairwise_counts = np.zeros((num_arms, num_arms))
        self.pairwise_wins = np.zeros((num_arms, num_arms))
        self.time = 1
        self.initialized_pairs = set()
        self.f_K = f_scale * (num_arms ** 1.01)  # Matches the paper: f(K) = c * K^1.01

    def kl_divergence(self, p, q):
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1 - epsilon)
        q = np.clip(q, epsilon, 1 - epsilon)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def empirical_winrate(self, i, j):
        total = self.pairwise_counts[i, j] + self.pairwise_counts[j, i]
        if total == 0:
            return 0.5
        return self.pairwise_wins[i, j] / total

    def empirical_divergence(self, i):
        score = 0
        for j in self.active_arms:
            if i == j:
                continue
            p_ij = self.empirical_winrate(i, j)
            if p_ij < 0.5:
                n_ij = self.pairwise_counts[i, j] + self.pairwise_counts[j, i]
                score += n_ij * self.kl_divergence(p_ij, 0.5)
        return score

    def select_arms(self):
        # Step 1: Initial exploration — make sure each pair is tried at least once
        for i, j in combinations(self.active_arms, 2):
            if (i, j) not in self.initialized_pairs and (j, i) not in self.initialized_pairs:
                self.initialized_pairs.add((i, j))
                return i, j

        # Step 2: Compute empirical divergence I_i(t) for each arm
        divergence_scores = {i: self.empirical_divergence(i) for i in self.active_arms}
        i_star = min(divergence_scores, key=divergence_scores.get)
        I_star = divergence_scores[i_star]

        # Step 3: Filter arms using Ji(t): I_i(t) - I^*(t) ≤ log(t) + f(K)
        candidates = [
            i for i in self.active_arms
            if divergence_scores[i] - I_star <= np.log(self.time) + self.f_K
        ]

        # Step 4: Pick l(t) in fixed order (we randomize here for generality)
        l_t = random.choice(candidates)

        # Step 5: Opponent selection using RMED1 subroutine (Algorithm 2)
        O_l = [j for j in self.active_arms if j != l_t and self.empirical_winrate(l_t, j) <= 0.5]
        if i_star in O_l or not O_l:
            m_t = i_star
        else:
            m_t = min(O_l, key=lambda j: self.empirical_winrate(l_t, j))

        return l_t, m_t

    def update(self, winner, loser, t):
        super().update(winner, loser, t)
        self.pairwise_counts[winner, loser] += 1
        self.pairwise_counts[loser, winner] += 1
        self.pairwise_wins[winner, loser] += 1
        self.time += 1

class HalvingBattleBandit(BaseDuelingBandit):
    def __init__(self, num_arms, confidence_threshold=0.95):
        super().__init__(num_arms, confidence_threshold)
        self.max_duels_per_round = num_arms*2  # Max number of duels per round for median elimination

    def median_elimination(self):
        # Calculate win rates
        total_duels = self.wins + self.losses
        win_rates = self.wins / total_duels

        # Get the median win rate
        median_win_rate = np.median(win_rates)

        # Eliminate arms whose win rate is lower than the median
        arms_to_keep = [i for i in self.active_arms if win_rates[i] >= median_win_rate]
        self.active_arms = arms_to_keep

    def run(self, true_strengths, max_iterations, strategy_name, run, log_file):
        start_time = time.time()
        correct_winner = np.argmax(true_strengths)

        for iterations in range(1, max_iterations + 1):
            # Select two arms to duel
            arm1, arm2 = self.select_arms()

            # Perform duel and update win/loss statistics
            winner = self.duel(arm1, arm2, true_strengths)
            loser = arm2 if winner == arm1 else arm1
            self.update(winner, loser, iterations)

            win_rates = self.wins / (self.wins + self.losses)
            ranking = np.argsort(-win_rates)
            position = np.where(ranking == correct_winner)[0][0] + 1

            if log_file is not None:
                # log_file.write(f"{run},{iterations},{position},{win_rates},{ranking},{true_strengths}\n")
                log_file.write(f"{run},{iterations},{position}\n")


            # After each round of duels, eliminate poorly performing arms
            if iterations % self.max_duels_per_round == 0:
                self.median_elimination()


            if len(self.active_arms) < 2:
                confident, best_arm = self.has_confident_winner()
                break
            # Check if we have a confident winner
            confident, best_arm = self.has_confident_winner()
            if confident:
                break

        execution_time = time.time() - start_time
        win_rates = self.wins / (self.wins + self.losses)
        ranking = np.argsort(-win_rates)
        position = np.where(ranking == correct_winner)[0][0] + 1
        regret = self.calculateLoss(true_strengths, iterations)

        return best_arm, best_arm == correct_winner, execution_time, iterations, position, regret, ranking, win_rates
