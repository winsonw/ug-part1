from virtualisation import VirtualisationDQN, VirtualisationDuelingDQN
import pickle
from utils import filePreflixProcession

FRAME_COUNT = 3000000
Statistic = {
    "reward"
}


class ParameterBatchRunning:
    def __init__(self, isDueling=False, isDropout=False, dropoutRate=0.0):
        self.numberOfParameter = (FRAME_COUNT - 50000) // 10000 + 1
        self.isDueling = isDueling
        self.data = {"reward": []}
        self.rewards = []

        self.filePreflix = filePreflixProcession(isDueling=isDueling, isDropout=isDropout, dropoutRate=dropoutRate)
        if isDueling:
            self.v = VirtualisationDuelingDQN()
        else:
            self.v = VirtualisationDQN()
        self.dataFileName = self.filePreflix + "collect_data"

    def runAll(self):
        for i in range(self.numberOfParameter):
            frameCount = 50000 + i*10000
            self.v.initialQFunction(isDueling=self.isDueling, filePreflix=self.filePreflix, frameCount=frameCount)
            reward = self.v.virtualise()
            self.rewards.append(reward)

    def save(self):
        for i in self.rewards:
            self.data["reward"].append(i)
        with open(self.dataFileName, 'wb') as dataFile:
            pickle.dump(self.data, dataFile)

    def load(self):
        with open(self.dataFileName, "rb") as file:
            data = pickle.load(file)
        self.rewards = data["reward"]
        return self.rewards



def main():
    isDueling = True
    isDropout = False
    dropoutRate = 0.0

    p = ParameterBatchRunning(isDueling=isDueling, isDropout=isDropout, dropoutRate=dropoutRate)
    p.runAll()
    p.save()


if __name__ == '__main__':
    main()