# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
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

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
# from sklearn import tree

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print "Initialising"

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray
                
    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])


        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.
            
        # *********************************************
        self.training()
        # *********************************************
        
    # Tidy up when Pacman dies
    def final(self, state):

        print "I'm done!"
        
        # *********************************************
        #
        # Any code you want to run at the end goes here.
        #
        # *********************************************

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):

        # How we access the features.
        features = api.getFeatureVector(state)
        
        # *****************************************************
        actionNumber = self.naiveBayes(features)
        action = self.convertNumberToMove(actionNumber)
        # *******************************************************

        # Get the actions we can try.
        legal = api.legalActions(state)

        # getAction has to return a move. Here we pass "STOP" to the
        # API to ask Pacman to stay where they are. We need to pass
        # the set of legal moves to teh API so it can do some safety
        # checking.
        return api.makeMove(action, legal)

    # the naiveBayes method use the the probability
    # that have calculated in the training method to select the the action
    # by taking the largest conditional probability
    # from the action probability times
    # the product of the posterior of all feature under the action taken
    def naiveBayes(self,features):
        conditionalProbability = np.ones([self.numberOfAction])
        conditionalProbability *= self.actionProbability
        for feature in range(self.numberOfFeature):
            if features[feature] == 1:
                featureProbablity = self.probability[:,feature]
            else:
                featureProbablity = 1 - self.probability[:, feature]
            conditionalProbability *= featureProbablity
        action = conditionalProbability.argmax()
        return action

    # the training method is used as median
    # in which it collect the counting data from counting method
    # and giving this data to the mEstimate method
    # to get the probability that we need
    def training(self):
        self.numberOfAction = 4
        self.numberOfFeature = len(self.data[0])
        sizeOfSample = len(self.data)

        actionCount,featureCount = self.counting(sizeOfSample)
        self.actionProbability, self.probability = self.mEstimate(sizeOfSample,actionCount,featureCount)


    # The counting method count all the occurance of action taken
    # and the feature being true under the action taken condition
    # from the training data which has been collected in the good_move.txt document
    def counting(self,sizeOfSample):
        actionCount = np.zeros([self.numberOfAction])
        featureCount = np.zeros([self.numberOfAction,self.numberOfFeature])

        for obs in range(sizeOfSample):
            action = self.target[obs]
            actionCount[action] +=1
            for feature in range(self.numberOfFeature):
                featureCount[action,feature] += self.data[obs][feature]
        return actionCount,featureCount


    # The mEstimate method using m-estimate as the improvement to Naive Bayes Classifier
    # to prevent overfitting due to the small size of sample
    # the probability is calculating as (n_c + m*p) / (n + m)
    # in which n is total number the specific action is taken
    # and n_c is the number of those cases where the feature is true
    # m is a constant, equivalent sample size
    # p is a prior estimate
    # typical value of p is 1/k if the feature has k values
    def mEstimate(self,sizeOfSample, actionCount, featureCount):
        m = sizeOfSample
        # because all the feature only have 2 values, which make p = 1/2
        p = 0.5
        probability = np.empty([self.numberOfAction, self.numberOfFeature])
        for action in range(self.numberOfAction):
            probability[action] = (featureCount[action] + m*p) / (actionCount[action] + m)

        # the part below calculate the probability of action taken also using m-estimate
        # because there are 4 action value, which make p = 1/4
        p = 1 / self.numberOfAction
        # in which have the same sample size, so we are going to use the same m value
        actionProbablity = (actionCount[action] + m*p) / (sizeOfSample + m)

        return actionProbablity,probability






