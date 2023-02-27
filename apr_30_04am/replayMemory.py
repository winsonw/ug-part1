import numpy as np

class ReplayMemory:
    def __init__(self, maxSize):
        self.maxSize = maxSize
        self.size = 0
        self.index = None
        self.full = False

        self.states = np.empty([maxSize,4,84,84], dtype=np.uint8)
        self.rewards = np.empty([maxSize], dtype=np.float)
        self.dones = np.empty([maxSize], dtype=np.bool)
        self.actions = np.empty([maxSize], dtype=np.int)

    def getSize(self):
        if self.full:
            return self.maxSize
        else:
            if self.size >= self.maxSize:
                self.full = True
            return self.size

    def store(self, state, reward, done, action):
        self.size += 1
        self.index = (self.size-1) % self.maxSize
        self.states[self.index] = state
        self.rewards[self.index] = reward
        self.dones[self.index] = done
        self.actions[self.index] = action

    def sample(self, index):
        elementIndex = (self.index + index) % self.maxSize
        nextElementIndex = (self.index + index + 1) % self.maxSize
        return self.states[elementIndex], self.rewards[elementIndex], self.dones[elementIndex], self.actions[elementIndex], self.states[nextElementIndex]

