from pylab import *
import pickle

def main(is_mean = True):
    file1 = open("statistics_low.pkl","rb")
    file2 = open("statistics_regular.pkl", "rb")
    file3 = open("statistics_large.pkl", "rb")
    file4 = open("statistics_rl_large.pkl", "rb")

    data1 = pickle.load(file1)
    data2 = pickle.load(file2)
    data3 = pickle.load(file3)
    data4 = pickle.load(file4)

    plt.figure()
    plt.subplot(211)


    if is_mean :
        plt.plot(data1['mean_episode_rewards'][:10000000],label = "0.00005")
        plt.plot(data2['mean_episode_rewards'][:10000000],label = "0.00025")
        plt.plot(data3['mean_episode_rewards'][:10000000],label = "0.00125")
        plt.plot(data4['mean_episode_rewards'][:10000000],label = "0.00625")

        plt.ylabel("mean rewards in 100 episode")
        plt.xlabel("frame")
        plt.legend()
        plt.show()
    else:
        plt.plot(data1['best_reward'][:10000000], label="0.00005")
        plt.plot(data2['best_reward'][:10000000], label="0.00025")
        plt.plot(data3['best_reward'][:10000000], label="0.00125")
        plt.plot(data4['best_reward'][:10000000], label="0.00625")

        plt.ylabel("best rewards in 100 episode")
        plt.xlabel("frame")
        plt.legend()
        plt.show()