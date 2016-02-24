from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator, SimulatorNoGraphics

import random
from collections import namedtuple
from itertools import product
from numpy.random import choice
from math import exp, log

State = namedtuple('State', 'light next_waypoint')

# Exploration-exploitation strategies
def explorer(state_q_values, trial):
    """
    An explorer always returns a random action choice, ignoring Q values.
    The trial is also ignored.
    """
    return choice(state_q_values.keys())

def exploiter(state_q_values, trial):
    """
    An exploiter always returns the action with the largest Q value.
    The trial is also ignored.
    """
    return max(state_q_values, key=state_q_values.get)

def weighted_q_average(state_q_values, trial):
    """
    Take a weighted random sample of the available actions, where the weights
    are the Q values.

    https://docs.python.org/2/library/stdtypes.html#dict.items says that if a
    dictionary isn't modified, then the order of items returned by the methods
    used below will remain consistent.
    """
    total = sum(w for w in state_q_values.itervalues())
    probs = [q / total for q in state_q_values.itervalues()]   # normalize
    return choice(state_q_values.keys(), p=probs)

def decay_exponential(state_q_values, trial, num_trials=100):
    """
    Explore with probability p and exploit with probability (1-p),
    where p(trial) = exp(-trial / 0.3 * num_trials).

    The 0.3 was picked to decay from 1 to 0 over ~100 trials at a reasonable rate:
    - On trial 0,  we explore w.p. 1
    - On trial 1,  we explore w.p. 0.967
    - On trial 10, we explore w.p. 0.717
    - On trial 50, we explore w.p. 0.189
    - On trial 99, we explore w.p. 0.037
    """
    p = exp(-1.0 * trial / (num_trials * 0.3))
    if p >= random.random():
        return explorer(state_q_values, trial)
    else:
        return exploiter(state_q_values, trial)

def decay_linear(state_q_values, trial, num_trials=100):
    """
    Explore with probability p and exploit with probability (1-p),
    where p(trial) is a decreasing linear function of trial.

    - On trial 0,  we explore w.p. 1
    - On trial 10, we explore w.p. 0.9
    - On trial 50, we explore w.p. 0.5
    - On trial 99, we explore w.p. 0.01
    """
    p = 1 - (1.0 * trial) / num_trials
    if p >= random.random():   # if p < 0, then this never happens
        return explorer(state_q_values, trial)
    else:
        return exploiter(state_q_values, trial)

def decay_logarithmic(state_q_values, trial, num_trials=100):
    """
    Explore with probability p and exploit with probability (1-p),
    where p(trial) is a decreasing logarithmic function of trial.

    With 100 trials, the decay looks like:
    - On trial 0,  we explore w.p. 1
    - On trial 10, we explore w.p. 0.977
    - On trial 50, we explore w.p. 0.85
    - On trial 90, we explore w.p. 0.5
    - On trial 95, we explore w.p. 0.35
    - On trial 98, we explore w.p. 0.15
    """
    try:
        p = log(num_trials - trial, num_trials)
    except ValueError:
        p = 0.0

    if p >= random.random():
        return explorer(state_q_values, trial)
    else:
        return exploiter(state_q_values, trial)



class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world via Q-Learning.

    Q-Learning maps state-action pairs to values, based on the feedback
    received from the environment in response to taking actions from various
    states.

    The main data structure for this is q_map, a dictionary that maps
    State -> {action -> Q-value}. When the agent is in learning mode
    (controlled by instance variable learning), these Q values are updated as a
    weighted average of the current value and the discounted, predicted future
    value of its new state.
    """

    def __init__(self, env, **kwargs):
        # Simulation infrastructure
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)

        # Q-Learning data structures
        self.current_state = None
        self.q_map = dict()

        # Q-Learning default parameters
        self.strategy = explorer
        self.q_boost = 0   # boost added to initial Q-value when action matches next_waypoint
        self.alpha = 0.5   # learning rate, i.e. how much of q-value depends on future versus past learning
        self.gamma = 0.8   # discount rate, i.e. how much do you value future rewards

        # Update Q-Learning parameters based on supplied keyword-arguments
        for key, value in kwargs.iteritems():
            if "boost" in key.lower():
                self.q_boost = value
            elif key == "alpha":
                self.alpha = value
            elif key == "gamma":
                self.gamma = value
            elif "strat" in key.lower():
                self.strategy = value

        # Initialize Q-values for all possible states to be random(0,1),
        # UNLESS you're using a strategy that works better with a different
        # initialization
        valid_light = ['green', 'red']   # as returned by Environment.sense
        valid_waypoints = Environment.valid_actions
        for state_pair in product(valid_light, valid_waypoints):
            state = State(*state_pair)
            self.q_map[state] = dict()
            for action in Environment.valid_actions:
                if self.strategy == weighted_q_average:
                    self.q_map[state][action] = 1.0
                else:
                    self.q_map[state][action] = random.random()

        print self.format_q_map()
        print self.q_map.keys()

        # Add in the Q-boost
        for state, q_map in self.q_map.iteritems():
            for action in Environment.valid_actions:
                if state.next_waypoint == action:
                    q_map[action] += self.q_boost

        # Should the agent update its q_map in response to environment feedback?
        self.learning = True

        # Performance tracking
        self.net_reward = 0   # reward after each action

        # How many times has this agent been reset?
        # This is equivalent to how many trials the agent has participated in
        self.num_resets = 0

    def reset(self, destination=None):
        """
        Get the agent ready for the next trial.
        """
        self.planner.route_to(destination)
        self.current_state = None
        self.net_reward = 0
        self.num_resets += 1

    def update(self, t, debug=True):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update current_state based on surroundings
        light = inputs['light']
        next_waypoint = self.next_waypoint
        self.current_state = State(light, next_waypoint)
        if debug:
            print self.current_state

        # Select the action for the current state with the largest Q value
        q_values_current_state = self.q_map[self.current_state]
        action = self.strategy(q_values_current_state, self.num_resets)
        if debug:
            print "Action choices:"
            print q_values_current_state
            print "Selected action: {}".format(action)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward += reward

        # Calling act() causes location to change if a valid move was made
        if self.learning:
            new_status = self.env.sense(self)
            new_light = new_status['light']
            new_waypoint = self.planner.next_waypoint()
            new_state = State(new_light, new_waypoint)

            # Learn policy based on state, action, reward via Q-Learning update
            previous_value = self.q_map[self.current_state][action]
            future_value = reward + self.gamma * max(self.q_map[new_state].itervalues())
            updated_q_value = ((1 - self.alpha) * previous_value) + (self.alpha * future_value)
            self.q_map[self.current_state][action] = updated_q_value

        if debug:
            if self.learning:
                print "Old value for ({}, {}, {}): {}".format(light, next_waypoint, action, previous_value)
                print "Reward received: {}".format(reward)
                print "Updated value for ({}, {}, {}): {}".format(light, next_waypoint, action, updated_q_value)
                print "LA.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)
                print
            else:
                print "Not learning; no update made"

    def __str__(self):
        s = "Agent with learning={}, alpha={}, gamma={}, boost={}, num_trials={}"
        return s.format(self.learning, self.alpha, self.gamma, self.q_boost, self.num_trials)

    @property
    def agent_info(self):
        info = {
            'q_boost': self.q_boost,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'strategy': self.strategy.__name__,
            'q_map': self.q_map,
            'num_resets': self.num_resets
        }
        return info

    def format_q_map(self):
        output = []
        for state, actions in self.q_map.iteritems():
            output.append(str(state))
            for action, q_value in actions.iteritems():
                output.append("\t{}: {:.2f}".format(action, q_value))
        return "\n".join(output)


def run_with_params(agent_params, use_deadline, show_graphics=True):
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, **agent_params)  # create agent
    e.set_primary_agent(a, enforce_deadline=use_deadline)  # set agent to track

    # Now simulate it
    sim = Simulator(e) if show_graphics else SimulatorNoGraphics(e)
    agent_performance = sim.run(100)
    print a.format_q_map()
    print agent_performance
    agent_info = a.agent_info
    return agent_performance, agent_info


def q1_random_action():
    agent_params = {'strategy': explorer}
    return run_with_params(agent_params, False)


def q2_max_q_value():
    agent_params = {'strategy': exploiter, 'q_boost': 1}
    return run_with_params(agent_params, True)


def q3_weighted_q_ave():
    agent_params = {'strategy': weighted_q_average}
    return run_with_params(agent_params, True, False)

def q3_decay_logarithmic():
    agent_params = {'strategy': decay_logarithmic}
    return run_with_params(agent_params, True, False)

def plot_agent_performances(performances):
    pass


def evaluate_performance():
    """
    Run training trials for each of many learning parameter/strategy
    combinations, then run evaluation trials to see how well we learned.
    """
    alpha_values = [.2, .4, .6, .8]
    gamma_values = [.2, .4, .6, .8]
    strategies = [weighted_q_average, decay1]

    num_training_trials = 100
    num_evaluation_trials = 100
    all_results = dict()

    for param_values in product(alpha_values, gamma_values, strategies):
        alpha, gamma, strategies = param_values

        e = Environment()
        sim = SimulatorNoGraphics(e)
        a = e.create_agent(LearningAgent, alpha=alpha, gamma=gamma, q_boost=boost)
        e.set_primary_agent(a, enforce_deadline=True)

        sim.run(num_training_trials)

        a.learning = False
        perfs = sim.run(num_evaluation_trials)
        all_results[param_values] = perfs

    return all_results


if __name__ == '__main__':
    # q1_random_action()
    # performance, info = q2_max_q_value()
    performance, info = q3_decay_logarithmic()
    # results = evaluate_performance()

    import matplotlib.pyplot as plt
    # plt.plot(rewards)
