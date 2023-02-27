from utils import filePreflixProcession
from dataLogging import DataLogging
from dataGathering import ParameterBatchRunning

import pickle
import matplotlib.pyplot as plt
import numpy as np


class GraphPlotting:
    def __init__(self):
        self.data = []
        self.label = []

    def addData(self, data, label):
        self.data.append(data)
        self.label.append(label)

    def plotEpisodeReward(self):
        for i in range(len(self.data)):
            y = self.data[i].episodeReward
            x = np.array([j for j in range(len(self.data[i].episodeReward))])
            plt.plot(x,y,label = self.label[i])

        plt.show()

    def plotParameterReward(self, frameCount=3000000):
        for i in range(len(self.data)):
            l = (frameCount // 10000 - 5) + 1
            y = [max(self.data[i].parameterReward[:l][:(j+1)]) for j in range(l)]
            x = np.array([(50000 + j*10000) for j in range(l)])
            print(self.label[i])
            plt.plot(x,y,label = self.label[i])

        plt.xlabel("frame count")
        plt.ylabel("max episode reward")
        plt.legend()
        plt.show()


class Data:
    def __init__(self, isDueling=False, isDropout=False, dropoutRate=0.0):
        self.filePreflix = filePreflixProcession(isDueling, isDropout, dropoutRate)
        self.dataLogging = DataLogging(self.filePreflix)
        self.dataGathering = ParameterBatchRunning(isDueling, isDropout, dropoutRate)
        self.parameterReward = None
        self.episodeReward = None

    def loadEpisodeReward(self):
        self.episodeReward = np.array(self.dataLogging.load())

    def loadParameterReward(self):
        self.parameterReward = np.array(self.dataGathering.load())



def main():
    p = GraphPlotting()

    isDueling = False
    isDropout = False
    dropoutRate = 0.0
    dqn = Data(isDueling, isDropout, dropoutRate)
    dqn.loadParameterReward()
    # p.addData(dqn)

    isDueling = True
    isDropout = False
    dropoutRate = 0.0
    d = Data(isDueling, isDropout, dropoutRate)
    d.loadParameterReward()
    p.addData(d,"Without dropout")

    isDueling = True
    isDropout = True
    dropoutRate = 0.1
    dd1 = Data(isDueling, isDropout, dropoutRate)
    dd1.loadParameterReward()
    # p.addData(dd1,"0.1")

    isDueling = True
    isDropout = True
    dropoutRate = 0.2
    dd2 = Data(isDueling, isDropout, dropoutRate)
    dd2.loadParameterReward()
    p.addData(dd2,"With dropout")

    isDueling = True
    isDropout = True
    dropoutRate = 0.5
    dd5 = Data(isDueling, isDropout, dropoutRate)
    dd5.loadParameterReward()
    # p.addData(dd5,"0.5")

    p.plotParameterReward(1200000)


if __name__ == '__main__':
    main()