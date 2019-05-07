
        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.
        
        # TESTING
        #print("Distance:")
        #print(state['tree']['dist'])


# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg

from SwingyMonkey import SwingyMonkey

# Some hyperparameters
binsize = 100
max_height = 1000 # Not sure what the full dimensions are so I guessed until it stopped giving me errors!
max_width = 1000
# Need to play with these settings a lot
gamma = 0.9
epsilon = 0.03
# Set alpha - can do this more dynamically if desired
alpha = 0.5
best_score = []


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        self.max_reward = 0
        self.cumulative_reward = 0
        # Q is an array with 4 dimensions:
        # 1) tree bottom height (since gap is alway same, would be redundant to include top also)
        # 2) Monkey bottom height (monkey height stays same, so same reasoning)
        # 3) tree distance
        # 4) action
        self.Q = np.zeros([int(max_height/binsize), int(max_height/binsize), int(max_width/binsize), 2])
        # N - counts number of actions taken in each state (same dimensions as Q)
        self.N = np.zeros([int(max_height/binsize), int(max_height/binsize), int(max_width/binsize), 2])
        
    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.cumulative_reward = 0
        self.epoch += 1

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''
        
        # Pull in the current state in terms of the bins
        tree = int(np.floor(np.divide(state['tree']['bot'],binsize)))
        monkey = int(np.floor(np.divide(state['monkey']['bot'],binsize)))
        # Special condition for distance (since it can be negative)
        # Give it max value if negative
        if state['tree']['dist'] > 0:
            dist = int(np.floor(np.divide(state['tree']['dist'],binsize)))
        else:
            dist = int(max_width/binsize-1)
        
        if not self.last_state == None:
            # Pull in the last state in terms of the bins
            last_tree = int(np.floor(np.divide(self.last_state['tree']['bot'],binsize)))
            last_monkey = int(np.floor(np.divide(self.last_state['monkey']['bot'],binsize)))
            # Special condition for distance (since it can be negative)
            # Give it max value if negative
            if state['tree']['dist'] > 0:
                last_dist = int(np.floor(np.divide(self.last_state['tree']['dist'],binsize)))
            else:
                last_dist = int(max_width/binsize-1)
        
            # Define a new action based on the Q values of different actions
            if self.Q[tree,monkey,dist,1] > self.Q[tree,monkey,dist,0]:
                new_action = 1
            else:
                new_action = 0
        
            # E-greedy replacement routine
            if self.N[tree,monkey,dist,new_action] == 0:
                if npr.rand() < epsilon:
                    new_action = npr.rand() > 0.5
            else:
                if npr.rand() < np.divide(epsilon,self.N[tree,monkey,dist,new_action]):
                    new_action = npr.rand() > 0.5
            
            # Get max Q result
            max_Q = np.max(self.Q[tree,monkey,dist,:])
            print("max Q", max_Q)
            # Update Q
            self.Q[last_tree,last_monkey,last_dist,self.last_action] += alpha*(self.last_reward + gamma*max_Q - self.Q[last_tree,last_monkey,last_dist,self.last_action])
            
            # Update max_reward
            self.cumulative_reward += int(self.last_reward)
            if self.cumulative_reward > self.max_reward:
                self.max_reward = self.cumulative_reward
                
        
        # Do a random action to start off (ignoring the above Q decision)
        if self.last_state == None:
            new_action = npr.rand() > 0.99

        # Update N
        self.N[tree,monkey,dist,new_action] += 1
        
        # Keeping track of reward   
        print("Max reward:", self.max_reward)
        
        # Reset actions for next iteration and submit new action
        self.last_action = new_action
        self.last_state  = state
        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''
        self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass
        
        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()
    pg.quit()
    return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 500, 10)

	# Save history. 
	np.save('hist',np.array(hist))
