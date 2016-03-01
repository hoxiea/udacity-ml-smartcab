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


def evaluate_performance_helper(alpha, gamma, strategy, \
        num_training_trials=100, num_evaluation_trials=100, num_repetitions=10):
    agent_params = {'alpha': alpha, 'gamma': gamma, 'strategy': strategy}
    for _ in xrange(num_repetitions):
        sim, e = initialize_simulator_environment(agent_params)
        training_performances = sim.run(num_training_trials)
        e.primary_agent.learning = False
        evaluation_performances = sim.run(num_evaluation_trials)
    return training_performances, evaluation_performances

if __name__ == '__main__':
    t, e = evaluate_performance_helper(0.5, 0.8, explorer)
