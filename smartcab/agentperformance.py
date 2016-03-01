import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from simulator import initialize_simulator_environment
from strategies import *

class PerformanceTracker(object):
    def __init__(self, performances, agent_info):
        self.performances = performances
        self.agent_info = agent_info

    @property
    def prop_reached_destination(self):
        return np.mean([p.reached_dest for p in self.performances])

# Performance evaluation
def evaluate_performance():
    """
    Run training trials for each of many learning parameter/strategy
    combinations, then run evaluation trials to see how well we learned.
    """
    alpha_values = [.2, .4, .6, .8]
    gamma_values = [.2, .4, .6, .8]
    strategies = [weighted_q_average, decay_logarithmic, decay_linear, decay_exponential]

    num_training_trials = 100
    num_evaluation_trials = 100
    all_results = dict()

    for param_values in product(alpha_values, gamma_values, strategies):
        evaluate_performance_helper(*param_values)

    return all_results


def evaluate_performance_helper(alpha, gamma, strategy, q_boost=0, \
        num_training_trials=100, num_evaluation_trials=100, num_repetitions=10):
    agent_params = {'alpha': alpha, 'gamma': gamma, 'q_boost': q_boost, 'strategy': strategy}
    for _ in xrange(num_repetitions):
        sim, e = initialize_simulator_environment(agent_params)
        training_performances = sim.run(num_training_trials)
        e.primary_agent.learning = False
        evaluation_performances = sim.run(num_evaluation_trials)
    return training_performances, evaluation_performances


def separate_performances(paps):
    """
    Separate a sequence of PrimaryAgentPerformances, separate into two groups:
    those that reached their destination and those that didn't.

    Also include the trial number of each, which is assumed to be captured by
    the order of the PrimaryAgentPerformances sequence.
    """
    reached = []
    stalled = []
    for trial, pap in enumerate(paps):
        if pap.reached_dest:
            reached.append((trial, pap))
        else:
            stalled.append((trial, pap))
    assert len(reached) + len(stalled) == len(paps)
    return reached, stalled

def plot_performances(training_paps, evaluation_paps):
    all_paps = training_paps + evaluation_paps
    total_num_trials = len(all_paps)

    reached, stalled = separate_performances(all_paps)

    reached_x = [trial for trial, pap in reached]
    reached_y_positive = [pap.positive_reward for _, pap in reached]
    reached_y_negative = [pap.negative_reward for _, pap in reached]

    stalled_x = [trial for trial, pap in stalled]
    stalled_y_positive = [pap.positive_reward for _, pap in stalled]
    stalled_y_negative = [pap.negative_reward for _, pap in stalled]



    fig, ax = plt.subplots(figsize=(10, 6))
    xmin = -3
    xmax = total_num_trials + 3
    ymin = min(reached_y_negative + stalled_y_negative) - 3
    ymax = max(reached_y_negative + stalled_y_positive) + 5
    xhalfway = len(training_paps) + 1.5

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("")

    # Create the scatter plots
    ax.scatter(reached_x, reached_y_positive, c='green', marker='o')
    ax.scatter(stalled_x, stalled_y_positive, c='green', marker='x')
    ax.scatter(reached_x, reached_y_negative, c='red', marker='o')
    ax.scatter(stalled_x, stalled_y_negative, c='red', marker='x')

    # Add lines
    ax.plot((0, total_num_trials), (0, 0), "b--")     # line at reward == 0
    ax.plot((xhalfway, xhalfway), (ymin, ymax), "b--")     # line at reward == 0

    # Add textual information
    ax.set_xlabel('Trial #')
    ax.set_ylabel('Reward')

if __name__ == '__main__':
    t, e = evaluate_performance_helper(0.5, 0.8, exploiter, 2)
    plot_performances(t, e)
