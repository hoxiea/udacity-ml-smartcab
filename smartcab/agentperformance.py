import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from itertools import product
from simulator import initialize_simulator_environment
from strategies import *

class TrainEvalPerformance(object):
    """
    A sequence of PrimaryAgentPerformances, along with helper methods for
    analyzing those performances.
    """
    def __init__(self, training_perfs, eval_perfs, agent_info):
        self.training_perfs = training_perfs
        self.eval_perfs = eval_perfs
        self.agent_info = agent_info

    def plot(self):
        all_paps = self.training_perfs + self.eval_perfs
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
        ymax = max(reached_y_positive + stalled_y_positive) + 5
        xhalfway = len(self.training_perfs) + 1.5

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        title = "Alpha: {}, Gamma: {}, Strategy: {}"
        title = title.format(self.agent_info['alpha'], self.agent_info['gamma'],
            self.agent_info['strategy'])
        ax.set_title(title)

        # Create the scatter plots
        ax.scatter(reached_x, reached_y_positive, c='green', marker='o')
        ax.scatter(stalled_x, stalled_y_positive, c='green', marker='x')
        ax.scatter(reached_x, reached_y_negative, c='red', marker='o')
        ax.scatter(stalled_x, stalled_y_negative, c='red', marker='x')

        # Add lines
        ax.plot((0, total_num_trials), (0, 0), "k--")  # reward == 0
        ax.plot((xhalfway,)*2, (ymin, ymax), "k--")  # separate train/eval

        # Add textual information
        ax.set_xlabel('Trial #')
        ax.set_ylabel('Reward')
        plt.show()


# Performance evaluation
def parameter_gridsearch(num_train=100, num_eval=100):
    """
    Run training trials for each of many learning parameter/strategy
    combinations, then run evaluation trials to see how well we learned.

    Yields a sequence of TrainEvalPerformances.
    """
    alpha_values = (.2, .4, .6, .8)
    gamma_values = (.2, .4, .6, .8)
    strategies = (explorer, weighted_q_average, decay_logarithmic, decay_linear, decay_exponential)

    for a, g, s in product(alpha_values, gamma_values, strategies):
        agent_params = {'alpha': a, 'gamma': g, 'strategy': s}
        yield train_evaluate(agent_params, num_train, num_eval)


def train_evaluate(agent_params, num_train=100, num_eval=100):
    """
    Run num_train training trials and num_eval evaluation trials for a
    LearningAgent initialized with a single set of agent_params.

    Returns a TrainEvalPerformance that captures performances and agent info.
    """
    sim, e = initialize_simulator_environment(agent_params)

    training_perfs = sim.run(num_train)
    e.primary_agent.stop_learning()
    evaluation_perfs = sim.run(num_eval)

    agent_info = e.primary_agent.agent_info
    return TrainEvalPerformance(training_perfs, evaluation_perfs, agent_info)


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


# def plot_performances(training_paps, evaluation_paps):
#     all_paps = training_paps + evaluation_paps
#     total_num_trials = len(all_paps)
#
#     reached, stalled = separate_performances(all_paps)
#
#     reached_x = [trial for trial, pap in reached]
#     reached_y_positive = [pap.positive_reward for _, pap in reached]
#     reached_y_negative = [pap.negative_reward for _, pap in reached]
#
#     stalled_x = [trial for trial, pap in stalled]
#     stalled_y_positive = [pap.positive_reward for _, pap in stalled]
#     stalled_y_negative = [pap.negative_reward for _, pap in stalled]
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     xmin = -3
#     xmax = total_num_trials + 3
#     ymin = min(reached_y_negative + stalled_y_negative) - 3
#     ymax = max(reached_y_positive + stalled_y_positive) + 5
#     xhalfway = len(training_paps) + 1.5
#
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.set_title("")
#
#     # Create the scatter plots
#     ax.scatter(reached_x, reached_y_positive, c='green', marker='o')
#     ax.scatter(stalled_x, stalled_y_positive, c='green', marker='x')
#     ax.scatter(reached_x, reached_y_negative, c='red', marker='o')
#     ax.scatter(stalled_x, stalled_y_negative, c='red', marker='x')
#
#     # Add lines
#     ax.plot((0, total_num_trials), (0, 0), "k--")     # line at reward == 0
#     ax.plot((xhalfway, xhalfway), (ymin, ymax), "k--")     # line at reward == 0
#
#     # Add textual information
#     ax.set_xlabel('Trial #')
#     ax.set_ylabel('Reward')

if __name__ == '__main__':
    gs = parameter_gridsearch()
    result = next(gs)
    result.plot()

