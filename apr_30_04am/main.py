from duelingDeepQLearning import DuelingDeepQLearning
from deepQLearning import DeepQLearning

def main():
    # DQN = DeepQLearning()
    # DQN.initialQFunction()
    # DQN.train()

    dueling = DuelingDeepQLearning()
    # dropoutRate = 0
    # dueling.initialQFunction(isDueling=True, isDropout=True, dropoutRate=dropoutRate)
    dueling.initialQFunction(isDueling=True)
    dueling.train()

if __name__ == '__main__':
    main()
