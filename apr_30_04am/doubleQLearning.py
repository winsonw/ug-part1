from deepQLearning import DeepQLearning
import torch

class DoubleQLearning(DeepQLearning):
    def __init__(self):
        super().__init__()

    def updateQFunction(self):
        stateBatch, rewardBatch, doneBatch, actionBatch, nextStateBatch, doneBatch = self.sample()

        currentQValues = self.qFunction(stateBatch).gather(1, actionBatch.unsqueeze(1)).squeeze()
        maxAction = torch.argmax(self.qFunction(nextStateBatch), dim=1)
        nextArgmaxQ = self.targetQFunction(nextStateBatch).gather(1, maxAction.unsqueeze(1)).squeeze()
        nextArgmaxQ = nextArgmaxQ * (1 - doneBatch)
        targetQValues = rewardBatch + (self.gamma * nextArgmaxQ)

        self.processLossFunction(targetQValues, currentQValues)
        self.periodicTargetFunctionUpdate()
