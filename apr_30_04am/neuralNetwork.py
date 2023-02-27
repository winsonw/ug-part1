import torch.nn as nn
import torch.nn.functional as F


class DeepQNetwork(nn.Module):
    def __init__(self, inChannels=4, actionNum=11):
        super(DeepQNetwork, self).__init__()
        self.loss = nn.MSELoss()
        self.convolutionLayer1 = nn.Conv2d(inChannels, 32, kernel_size=8, stride=4)
        self.convolutionLayer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.convolutionLayer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fullyConnectedLayer1 = nn.Linear(7*7*64, 512)
        self.fullyConnectedLayer2 = nn.Linear(512, actionNum)

    def forwardCNN(self, data):
        data = F.relu(self.convolutionLayer1(data))
        data = F.relu(self.convolutionLayer2(data))
        data = F.relu(self.convolutionLayer3(data))
        return data

    def forward(self, data):
        data = self.forwardCNN(data)
        data = F.relu(self.fullyConnectedLayer1(data.view(data.size(0), -1)))
        return self.fullyConnectedLayer2(data)


class DeepQNetworkWithDropout(DeepQNetwork):
    def __init__(self, inChannels=4, actionNum=11, dropoutRate=0.8):
        super(DeepQNetworkWithDropout, self).__init__(inChannels=inChannels, actionNum=actionNum)
        self.dropoutLayer = nn.Dropout(p=dropoutRate)

    def forward(self, data):
        data = self.forwardCNN(data)
        data = self.dropoutLayer(data)
        data = F.relu(self.fullyConnectedLayer1(data.view(data.size(0), -1)))
        data = self.dropoutLayer(data)
        data = self.fullyConnectedLayer2(data)
        return data


class DuelingDeepQNetwork(DeepQNetwork):
    def __init__(self, inChannels=4, actionNum=11):
        super(DuelingDeepQNetwork, self).__init__(inChannels=inChannels, actionNum=actionNum)
        self.fullyConnectedLayer1 = nn.Linear(7*7*64, 1024)
        self.fullyConnectedLayer2 = nn.Linear(1024, 512)

        self.value = nn.Linear(512, 1)
        self.action = nn.Linear(512, actionNum)

    def forward(self, data):
        data = self.forwardCNN(data)
        data = F.relu(self.fullyConnectedLayer1(data.view(data.size(0),-1)))
        data = F.relu(self.fullyConnectedLayer2(data))

        value = self.value(data)
        action = self.action(data)
        return value, action


class DuelingDeepQNetworkWithDropout(DuelingDeepQNetwork):
    def __init__(self, inChannels=4, actionNum=11, dropoutRate=0.8):
        super(DuelingDeepQNetworkWithDropout, self).__init__(inChannels=inChannels, actionNum=actionNum)
        self.dropoutLayer = nn.Dropout(p=dropoutRate)

    def forward(self, data):
        data = self.forwardCNN(data)
        data = self.dropoutLayer(data)
        data = F.relu(self.fullyConnectedLayer1(data.view(data.size(0), -1)))
        data = self.dropoutLayer(data)
        data = self.fullyConnectedLayer2(data)

        value = self.value(data)
        action = self.action(data)
        return value, action
