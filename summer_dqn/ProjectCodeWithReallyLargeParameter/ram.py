import gym
import torch.optim as optim
from dqn_model import DQN_RAM
from dqn_learn import OptimizerSpec, dqn_learing
from utils.gym import get_ram_env
from utils.schedule import LinearSchedule

BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE=1000000
LEARNING_STARTS=50000
LEARNING_FREQ=4
FRAME_HISTORY_LEN=1
TARGER_UPDATE_FREQ=10000
LEARNING_RATE = 0.00625
ALPHA = 0.95
EPS = 0.1

def main():
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )

    env = gym.make('Breakout-ram-v0')
    seed = 100
    env = get_ram_env(env, seed)
    exploration_schedule = LinearSchedule(1000000, EPS)

    dqn_learing(
        env=env,
        q_func=DQN_RAM,
        optimizer_spec=optimizer_spec,
        exploration=exploration_schedule,
    )

if __name__ == '__main__':
    main()
