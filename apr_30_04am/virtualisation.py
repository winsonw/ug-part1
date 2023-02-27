from deepQLearning import DeepQLearning
from utils import filePreflixProcession
import torch


FRAME_COUNT = 50000


class VirtualisationDQN(DeepQLearning):
    def __init__(self, isDisplayed=False):
        super().__init__()
        self.isDisplayed = isDisplayed

    def initialQFunction(self, isDueling=False, filePreflix="", frameCount=50000, isTraining=False):
        super().initialQFunction(isDueling=isDueling, isTraining=isTraining)
        self.qFunction.load_state_dict(torch.load("parameters/" + filePreflix + "parameters" + str(frameCount) + ".pkl"))
        print("This simulation is for frame:", frameCount)

    def selectAction(self,state):
        state = torch.from_numpy(state).type(self.dataType).unsqueeze(0) / 255.0
        qValues = self.qFunction(state)
        actionIndex = torch.argmax(qValues).item()
        return actionIndex

    def virtualise(self):
        lastState = self.reset(self.env)
        cumulativeReward = 0
        done = False
        maxReward = 0
        while not done:
            if self.isDisplayed:
                self.env.render()
            action = self.selectAction(lastState)
            lastState, reward, done = self.step(action, self.env)
            cumulativeReward += reward
            if cumulativeReward > maxReward:
                maxReward = cumulativeReward

        print("Max reward:", maxReward)
        return maxReward



class VirtualisationDuelingDQN(VirtualisationDQN):
    def __init__(self, isDisplayed=False):
        super().__init__(isDisplayed=isDisplayed)

    def selectAction(self,state):
        state = torch.from_numpy(state).type(self.dataType).unsqueeze(0) / 255.0
        values, advantages = self.qFunction(state)
        actionIndex = torch.argmax(advantages).item()
        return actionIndex




def main():
    isDueling = False
    isDropout = False
    dropoutRate = 0.0
    isDisplayed = False

    if isDueling:
        v = VirtualisationDuelingDQN(isDisplayed=isDisplayed)
    else:
        v = VirtualisationDQN(isDisplayed=isDisplayed)
    filePreflix = filePreflixProcession(isDueling=isDueling, isDropout=isDropout, dropoutRate=dropoutRate)
    v.initialQFunction(isDueling=isDueling, filePreflix=filePreflix, frameCount=FRAME_COUNT)
    v.virtualise()


if __name__ == '__main__':
    main()