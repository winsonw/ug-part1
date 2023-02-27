import random


class EpsilonGreedy:
    def __init__(self, initialExploration=1.0, finalExploration=0.1, finalExplorationFrame=1000000, replayStart=50000, actionSpace=12):
        self.initialExploration = initialExploration
        self.finalExploration = finalExploration
        self.finalExplorationFrame = finalExplorationFrame
        self.replayStart = replayStart
        self.actionSpace = actionSpace

    def decide(self, frame):
        r = random.random()
        if frame >= self.finalExplorationFrame:
            epsilon = self.finalExploration
        else:
            epsilon = self.initialExploration - (frame - self.replayStart)/self.finalExplorationFrame * (self.initialExploration - self.finalExploration)
        return r <= epsilon
