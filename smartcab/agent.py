import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from itertools import product

State = namedtuple('State', 'light oncoming left right next_waypoint')

# TODO: can we get away with a simpler state? I'm not sure that 'right' provides any useful information
# If the light is red:
# - And there's nobody to your left going forward, then you can take a right on red
# - Don't need to worry about oncoming, since light is red for them too, and no left thru red light
# - Don't need to worry about right, since nothing they do can interfere with you
# If the light is green:
# - You can go forward or right with the right of way and no problems
# - You can go left if there's nobody oncoming going forward
# In both cases, right is irrelevant

class LearningAgent(Agent):
    """
    An agent that learns to drive in the smartcab world.

    TODO: discuss the main learning data structure, q_map
    """

    def __init__(self, env):
        # Simulation stuff
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)

        # Learning stuff
        self.current_state = None
        self.alpha = 0.5   # learning rate
        self.gamma = 0.8   # discount rate
        self.q_map = dict()

        valid_light = Environment.valid_inputs['light']
        valid_oncoming = Environment.valid_inputs['oncoming']
        valid_left = Environment.valid_inputs['left']
        valid_right = Environment.valid_inputs['right']
        valid_waypoints = Environment.valid_actions
        for state in product(valid_light, valid_oncoming, valid_left, valid_right, valid_waypoints):
            self.q_map[state] = dict()
            for action in Environment.valid_actions:
                self.q_map[state][action] = random.random()

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.current_state = None
        # I don't think we want to reset q_map here... all the agents get reset
        # after each trial, so if we're going to remember what we learn, we
        # shouldn't clear out q_map
        # self.q_map = dict()
        # TODO: reset any other variables here

    def update(self, t, debug=True):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        light = inputs['light']
        oncoming = inputs['oncoming']
        left = inputs['left']
        right = inputs['right']
        next_waypoint = self.next_waypoint
        self.current_state = State(light, oncoming, left, right, next_waypoint)
        if debug:
            print self.current_state

        # Select the action for the current state with the largest Q value
        action_values = self.q_map[self.current_state]
        action = max(action_values, key=action_values.get)
        assert action in Environment.valid_actions
        if debug:
            print "Action choices:"
            print action_values
            print "Selected action: {}".format(action)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Calling act causes location to change if a valid move was made
        # But will other agents have moved, giving us updated intersection information?
        # TODO: figure this out and you're done
        next_state = 0

        # Learn policy based on state, action, reward via Q-Learning update
        previous_value = self.q_map[state][action]
        future_value = reward + self.gamma * max(self.q_map[next_state].itervalues())
        self.q_map[state][action] = (1 - self.alpha) * previous_value + (self.alpha * future_value)

        if debug:
            print "LA.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
