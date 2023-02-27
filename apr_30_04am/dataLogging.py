import numpy as np
import torch
import pickle

Statistic = {
    "episode_reward"
}


class DataLogging:
    def __init__(self, filePreflix, learningStart=50000):
        self.dataFileName = filePreflix + 'data.pkl'
        self.paraFileName = "parameters/"+filePreflix + 'parameters'
        self.learningStart = learningStart
        self.DATA = {
            "episode_reward": [],
        }

    def frequentLogging(self, frameCount):
        if frameCount % 5000 == 0:
            print("frame:", frameCount)
            if frameCount >= self.learningStart and frameCount % 10000 == 0:
                episodeCount = len(self.DATA["episode_reward"])
                meanEpisodeReward = np.mean(self.DATA["episode_reward"][-100:])
                currentBestReward = np.max(self.DATA["episode_reward"][-100:])
                bestReward = np.max(self.DATA["episode_reward"])
                print("Best reward so far: ", bestReward)
                print("Best reward in 100 episode: ", currentBestReward)
                print("Mean reward in 100 episode: ", meanEpisodeReward)
                print("Episode count: ", episodeCount)

    def updateData(self, episodeReward):
        self.DATA["episode_reward"].append(episodeReward)

    def saveData(self, qFunction, frameCount):
        with open(self.paraFileName + str(frameCount) + ".pkl", 'wb') as paraFile:
            torch.save(qFunction.state_dict(),paraFile)
        with open(self.dataFileName, 'wb') as dataFile:
            pickle.dump(self.DATA, dataFile)

    def load(self):
        with open(self.dataFileName, 'rb') as dataFile:
            self.DATA = pickle.load(dataFile)
        return self.DATA["episode_reward"]