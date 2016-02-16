import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from itertools import product

State = namedtuple('State', 'light next_waypoint')

class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world.

    TODO: discuss the main learning data structure, q_map
    """

    def __init__(self, env, **kwargs):
        # Simulation stuff
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)

        # Q-Learning parameters
        self.current_state = None
        self.q_map = dict()
        self.q_boost = 1   # priority given to initial Q-value when action matches next_waypoint
        self.alpha = 0.5   # learning rate
        self.gamma = 0.8   # discount rate

        for key, value in kwargs.iteritems():
            if key == "q_boost":
                self.q_boost = value
            elif key == "alpha":
                self.alpha = value
            elif key == "gamma":
                self.gamma = value

        # Initialize Q-values for all possible states
        valid_light = ['green', 'red']   # as returned by Environment.sense
        valid_waypoints = Environment.valid_actions
        for state_pair in product(valid_light, valid_waypoints):
            state = State(*state_pair)
            self.q_map[state] = dict()
            state_light, state_waypoint = state
            for action in Environment.valid_actions:
                if state_waypoint == action:
                    self.q_map[state][action] = self.q_boost + random.random()
                else:
                    self.q_map[state][action] = random.random()

        self.learning = True

        # Performance tracking
        self.net_reward = 0

    def reset(self, destination=None):
        """
        Get the agent ready for the next trial.
        """
        self.planner.route_to(destination)
        self.current_state = None
        self.net_reward = 0

    def update(self, t, debug=True):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        light = inputs['light']
        next_waypoint = self.next_waypoint
        self.current_state = State(light, next_waypoint)
        if debug:
            print self.current_state

        # Select the action for the current state with the largest Q value
        action_values = self.q_map[self.current_state]
        action = max(action_values, key=action_values.get)
        if debug:
            print "Action choices:"
            print action_values
            print "Selected action: {}".format(action)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.net_reward += reward

        # Calling act causes location to change if a valid move was made
        if self.learning:
            new_status = self.env.sense(self)
            new_light = new_status['light']
            new_waypoint = self.planner.next_waypoint()
            new_state = State(new_light, new_waypoint)

            # Learn policy based on state, action, reward via Q-Learning update
            previous_value = self.q_map[self.current_state][action]
            future_value = reward + self.gamma * max(self.q_map[new_state].itervalues())
            updated_q_value = (1 - self.alpha) * previous_value + (self.alpha * future_value)
            self.q_map[self.current_state][action] = updated_q_value

        if debug:
            if self.learning:
                print "Old value for ({}, {}, {}): {}".format(light, next_waypoint, action, previous_value)
                print "Reward received: {}".format(reward)
                print "Updated value for ({}, {}, {}): {}".format(light, next_waypoint, action, updated_q_value)
                print "LA.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)
            else:
                print "Not learning; no update made"

    def __str__(self):
        return "Agent with learning={}, alpha={}, gamma={}, boost={}".format(self.learning, self.alpha, self.gamma, self.q_boost)

    def format_q_map(self):
        output = []
        for state, actions in self.q_map.iteritems():
            output.append(str(state))
            for action, q_value in actions.iteritems():
                output.append("\t{}: {:.2f}".format(action, q_value))
        return "\n".join(output)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=.3)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


def evaluate_performance():
    e = Environment()

    alpha_values = [.2, .4, .6, .8]
    gamma_values = [.2, .4, .6, .8]
    boost_values = [0, 0.5, 1]

    num_training_trials = 25
    num_evaluation_trials = 25
    all_results = dict()

    for param_values in product(alpha_values, gamma_values, boost_values):
        alpha, gamma, boost = param_values
        a = e.create_agent(LearningAgent, alpha=alpha, gamma=gamma, q_boost=boost)
        e.set_primary_agent(a, enforce_deadline=True)

        print a
        raw_input("Press any key to start this agent's learning trials")
        sim = Simulator(e, update_delay=.001)
        sim.run(n_trials=num_training_trials)

        a.learning = False
        print a.format_q_map()
        raw_input("Press any key to start this agent's evaluation trials")
        results = sim.run(n_trials=num_evaluation_trials)
        all_results[param_values] = results

    return all_results


if __name__ == '__main__':
    # run()
    evaluate_performance()
