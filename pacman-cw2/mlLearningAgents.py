# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py
# learningAgents.py

from pacman import Directions
from game import Agent
import random
import game
import util
import numpy as np


# QLearnAgent
#
class QLearnAgent(Agent):

    # Constructor, called when we start running the
    def __init__(self, alpha=0.2, epsilon=0.01, gamma=0.8, numTraining=10):
        # alpha       - learning rate
        # epsilon     - exploration rate
        # gamma       - discount factor
        # numTraining - number of training episodes
        #
        # These values are either passed from the command line or are
        # set to the default values above. We need to create and set
        # variables for them
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # initialise teh q learning algorithm
        self.qLearning = QLearning(alpha=alpha, epsilon=epsilon, gamma=gamma)

    # Accessor functions for the variable episodesSoFars controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value):
        self.epsilon = value

    def getAlpha(self):
        return self.alpha

    def setAlpha(self, value):
        self.alpha = value

    def getGamma(self):
        return self.gamma

    def getMaxAttempts(self):
        return self.maxAttempts

    # getAction
    #
    # The main method required by the game. Called every time that
    # Pacman is expected to move
    def getAction(self, state):
        self.qLearning.preprocessState(state)
        self.qLearning.updateQValue(state)
        action = self.qLearning.getAction(state)
        return action

    # Handle the end of episodes
    #
    # This is called by the game after a win or a loss.
    def final(self, state):
        self.qLearning.preprocessState(state)
        self.qLearning.updateQValue(state)
        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print '%s\n%s' % (msg, '-' * len(msg))
            self.setAlpha(0)
            self.setEpsilon(0)


class QLearning:
    # Constructor, called when the agent initialise
    # @para alpha       - learning rate
    # @para epsilon     - exploration rate
    # @para gamma       - discount factor
    def __init__(self, alpha, epsilon, gamma):
        print(np.__version__)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)

        # qTable            - a dictionary that store the state and its corresponding action values(q)
        # lastScore         - the score of last state
        # lastState         - the last state
        # lastActionIndex   - the last chosen action
        self.qTable = dict()
        self.lastScore = None
        self.lastState = None
        self.lastActionIndex = None

    # Before updating q table, every state has to be checked if it is already in the qTable or not
    # if not, then add the new state, action values pair to the dictionary
    # @para state           - current state
    def preprocessState(self, state):
        actions = self.getLegalAction(state)
        if not state in self.qTable:
            # for all the state that has not yet been counted in the q table, we initialise its q value
            self.qTable[state] = np.array([None for i in range(5)])
            if len(actions) != 0:
                # we initial the q value of all the legal action for the given state
                self.qTable[state][actions] = 0
            else:
                # This mean, it is an final state, so we set the 5th index as the place to store its q value
                self.qTable[state][4] = 0

    # update the q value with q-learning algorithm
    # Q(s,a) = Q(s,a) + alpha*(R(s) + gamma * max_a' Q(s',a') - Q(s,a) )
    def updateQValue(self, state):
        # if it is the initial state, then leave it
        if self.lastState == None:
            return
        reward = self.getReward(state)
        self.qTable[self.lastState][self.lastActionIndex] += self.alpha * (
                    reward + self.gamma * self.qTable[state].max() - self.qTable[self.lastState][self.lastActionIndex])

    # by using epsilon greedy, select action based on the current q value table
    # @para state       - current state
    # @retrun action    - the chosen direction
    def getAction(self, state):
        actions = self.getLegalAction(state)
        probability = random.random()
        if probability >= self.epsilon:
            # choose the action with the largest q values in the q table in the given state
            actionIndex = self.qTable[state].argmax()
        else:
            # randomly choose an legal action
            actionIndex = random.choice(actions)

        self.iterate(state, actionIndex)
        action = self.fromNumToDir(actionIndex)
        return action

    # get reward of the given state
    # reward is calculated by getting the different between two continuously state
    # @para state       - current state
    # @retrun reward    - the reward for the current state
    def getReward(self, state):
        reward = state.getScore() - self.lastScore
        return reward

    # get all the possible action index that Pacman can take in the given state
    # @para state       - current state
    # @return actions   - a list of action index
    def getLegalAction(self, state):
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        actions = self.fromDirToNum(legal)
        return actions

    # transfer the direction into action index
    # @para             - a list of direction
    # @return           - a list of converted action index
    def fromDirToNum(self, dir):
        dic = np.array([Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST])
        return np.array([np.where(dic == i) for i in dir])

    # transfer the action index to direction
    # @para             - an action index
    # @return           - an converted direction
    def fromNumToDir(self, num):
        dic = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]
        return dic[num]

    # iterate all the "last" variable
    #
    # called after an action is taken
    # @para state       - current state
    # @para actionIndex - the action been taken
    def iterate(self, state, actionIndex):
        self.lastScore = state.getScore()
        self.lastState = state
        self.lastActionIndex = actionIndex

    # reset all the "last" variable
    #
    # called when an episode is finished
    def reset(self):
        self.lastScore = None
        self.lastState = None
        self.lastActionIndex = None
