from neuralNetwork import DeepQNetwork, DuelingDeepQNetwork, DeepQNetworkWithDropout, DuelingDeepQNetworkWithDropout
from replayMemory import ReplayMemory
from epsilonGreedy import EpsilonGreedy
from utils import convertImage, filePreflixProcession
from dataLogging import DataLogging

from itertools import count
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import random


LEARNING_START = 50000
BATCH_SIZE = 32
NUMBER_OF_ACTION = 7 + 4
DISCOUNT_FACTOR = 0.99
LEARNING_RATE = 0.00025
GRADIENT_MOMENTUM = 0.95
MIN_SQUARED_GRADIENT = 0.01
REPLAY_MEMORY_SIZE = 100000
Q_UPDATE_FREQUENCY = 4
TARGET_Q_UPDATE_FREQUENCY = 10000
DATA_TYPE = torch.FloatTensor
ACTION_REPEAT = 4
LENGTH_OF_HISTORY = 4
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.1
FINAL_EXPLORATION_FRAME = 100000


class DeepQLearning:
    def __init__(self):
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

        self.learningStart = LEARNING_START
        self.dataType = DATA_TYPE
        self.numberOfAction = NUMBER_OF_ACTION
        self.gamma = DISCOUNT_FACTOR
        self.replayMemory = ReplayMemory(maxSize=REPLAY_MEMORY_SIZE)
        self.epsilonGreedy = EpsilonGreedy(initialExploration=INITIAL_EXPLORATION,
                                           finalExploration=FINAL_EXPLORATION,
                                           finalExplorationFrame=FINAL_EXPLORATION_FRAME,
                                           replayStart=self.learningStart,
                                           actionSpace=self.numberOfAction)
        self.updateCount = 0
        self.episodeRewards = []
        self.isTraining = True

    def initialQFunction(self, isDueling=False, isDropout=False , dropoutRate=0, isTraining=True):
        if not isDueling:
            if not isDropout:
                self.qFunction = DeepQNetwork(LENGTH_OF_HISTORY, self.numberOfAction).type(self.dataType)
                self.targetQFunction = DeepQNetwork(LENGTH_OF_HISTORY, self.numberOfAction).type(self.dataType)
            else:
                self.qFunction = DeepQNetworkWithDropout(LENGTH_OF_HISTORY, self.numberOfAction, dropoutRate).type(self.dataType)
                self.targetQFunction = DeepQNetworkWithDropout(LENGTH_OF_HISTORY, self.numberOfAction, dropoutRate).type(self.dataType)
        else:
            if not isDropout:
                self.qFunction = DuelingDeepQNetwork(LENGTH_OF_HISTORY, self.numberOfAction).type(self.dataType)
                self.targetQFunction = DuelingDeepQNetwork(LENGTH_OF_HISTORY, self.numberOfAction).type(self.dataType)
            else:
                self.qFunction = DuelingDeepQNetworkWithDropout(LENGTH_OF_HISTORY, self.numberOfAction, dropoutRate).type(self.dataType)
                self.targetQFunction = DuelingDeepQNetworkWithDropout(LENGTH_OF_HISTORY, self.numberOfAction, dropoutRate).type(self.dataType)

        self.optimizer = optim.RMSprop(self.qFunction.parameters(), lr=LEARNING_RATE, alpha=GRADIENT_MOMENTUM, eps=MIN_SQUARED_GRADIENT)

        self.isTraining = isTraining
        self.filePreflix = filePreflixProcession(isDueling=isDueling, isDropout=isDropout, dropoutRate=dropoutRate)
        self.dataLogging = DataLogging(self.filePreflix, self.learningStart)

    def sample(self):
        size = self.replayMemory.getSize()
        batch = np.random.choice(range(size), BATCH_SIZE)

        stateBatch, rewardBatch, doneBatch, actionBatch, nextStateBatch = self.replayMemory.sample(batch)
        stateBatch = Variable(torch.from_numpy(stateBatch).type(self.dataType) / 255.0)
        actionBatch = Variable(torch.from_numpy(actionBatch).long())
        rewardBatch = Variable(torch.from_numpy(rewardBatch).type(self.dataType))
        nextStateBatch = Variable(torch.from_numpy(nextStateBatch).type(self.dataType) / 255.0)
        doneBatch = Variable(torch.from_numpy(doneBatch)).type(self.dataType)
        return stateBatch, rewardBatch, doneBatch, actionBatch, nextStateBatch

    def selectAction(self, frameCount, state):
        if frameCount < self.learningStart or self.epsilonGreedy.decide(frameCount):
            actionIndex = random.randint(0, self.numberOfAction - 1)
        else:
            state = torch.from_numpy(state).type(self.dataType).unsqueeze(0) / 255.0
            qValues = self.qFunction(state)
            actionIndex = torch.argmax(qValues).item()
        return actionIndex

    def train(self):
        state1 = self.reset(self.env)
        episodeReward1 = 0

        for frameCount in count():

            state1, episodeReward1 = self.observate(self.env, state1, frameCount, episodeReward1)

            if frameCount >= self.learningStart and frameCount % Q_UPDATE_FREQUENCY == 0:
                self.updateQFunction()

            self.dataLogging.frequentLogging(frameCount)
            if frameCount >= self.learningStart and frameCount % 10000 == 0:
                self.dataLogging.saveData(self.qFunction,frameCount)

    def observate(self, env, state, frameCount, episodeReward):
        lastState = state
        action = self.selectAction(frameCount, lastState)
        state, reward, done = self.step(action, env)
        episodeReward += reward
        self.replayMemory.store(lastState, reward, done, action)
        if done:
            self.episodeRewards.append(episodeReward)
            self.dataLogging.updateData(episodeReward)
            episodeReward = 0
            state = self.reset(env)

        return state, episodeReward

    def updateQFunction(self):
        stateBatch, rewardBatch, doneBatch, actionBatch, nextStateBatch = self.sample()
        qValues = self.qFunction(stateBatch)
        currentQValues = qValues.gather(1, actionBatch.unsqueeze(1)).squeeze()
        nextMaxQ = self.targetQFunction(nextStateBatch).detach().max(1)[0]
        nextMaxQ = nextMaxQ * (1 - doneBatch)
        targetQValues = rewardBatch + (self.gamma * nextMaxQ)

        self.processLossFunction(targetQValues, currentQValues)
        self.periodicTargetFunctionUpdate()

    def processLossFunction(self, targetQValues, currentQValues):
        loss = self.qFunction.loss(targetQValues, currentQValues)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def periodicTargetFunctionUpdate(self):
        self.updateCount += 1
        if self.updateCount % TARGET_Q_UPDATE_FREQUENCY == 0:
            self.targetQFunction.load_state_dict(self.qFunction.state_dict())

    def step(self, action, env):
        states = []
        rewards = 0
        done = False
        processedState = None

        action, multiple = self.checkExpandAction(action)

        for i in range(ACTION_REPEAT * multiple):
            if done:
                if i % multiple == 0:
                    states.append(processedState)
            else:
                if i == 0 and multiple > 1:
                    state, reward, done, info = env.step(0)
                else:
                    state, reward, done, info = env.step(action)
                if self.isTraining and reward <= -10:
                    done = True
                processedState = convertImage(state)
                reward = reward / 15
                if i % multiple == 0:
                    states.append(processedState)
                rewards += reward

        states = np.stack(states)
        rewards = rewards / multiple
        return states, rewards, done

    def checkExpandAction(self, action):
        if 7 <= action <= 8:
            multiple = 2
            action = 2 if action == 7 else 5
        elif 9 <= action <= 10:
            multiple = 5
            action = 2 if action == 9 else 5
        else:
            multiple = 1

        return action, multiple

    def reset(self, env):
        state = env.reset()
        processedState = convertImage(state)
        states = np.array([processedState, processedState, processedState, processedState])
        return states
