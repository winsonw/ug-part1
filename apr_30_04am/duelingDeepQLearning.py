from deepQLearning import DeepQLearning
import torch
import random


class DuelingDeepQLearning(DeepQLearning):
    def __init__(self):
        super().__init__()

    def selectAction(self, frameCount, state):
        if frameCount < self.learningStart or self.epsilonGreedy.decide(frameCount):
            actionIndex = random.randint(0, self.numberOfAction - 1)
        else:
            state = torch.from_numpy(state).type(self.dataType).unsqueeze(0) / 255.0
            values, advantages = self.qFunction(state)
            actionIndex = torch.argmax(advantages).item()
        return actionIndex

    def updateQFunction(self):
        stateBatch, rewardBatch, doneBatch, actionBatch, nextStateBatch, doneBatch = self.sample()

        currentQValues = self.getQValueDuelingDQN(self.qFunction,stateBatch).gather(1, actionBatch.unsqueeze(1)).squeeze()
        maxAction = torch.argmax(self.getQValueDuelingDQN(self.qFunction,nextStateBatch), dim=1)
        nextArgmaxQ = self.getQValueDuelingDQN(self.targetQFunction,nextStateBatch).gather(1, maxAction.unsqueeze(1)).squeeze(1)
        nextArgmaxQ = nextArgmaxQ * (1 - doneBatch)
        targetQValues = rewardBatch + (self.gamma * nextArgmaxQ)

        self.processLossFunction(targetQValues, currentQValues)
        self.periodicTargetFunctionUpdate()

    def getQValueDuelingDQN(self,function,stateBatch):
        values, advantages = function(stateBatch)
        qValues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qValues