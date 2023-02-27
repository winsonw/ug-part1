import sys
import pickle
from collections import namedtuple
from itertools import count
import random
import torch
import torch.autograd as autograd
import numpy as np

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

dtype = torch.FloatTensor
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": [],
    "best_reward": []
}

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        super(Variable, self).__init__(data, *args, **kwargs)


def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0)/255
            return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([random.randrange(num_actions)])

    input_arg = env.observation_space.shape[0]
    num_actions = env.action_space.n

    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    frame_count = 0
    mean_episode_reward = 0
    best_mean_episode_reward = 0
    current_best_reward = 0
    last_obs = env.reset()

    for t in count():
        last_idx = replay_buffer.store_frame(last_obs)
        recent_observations = replay_buffer.encode_recent_observation()

        if t > learning_starts:
            actions = select_epilson_greedy_action(Q, recent_observations, t)
            action = actions[0]
        else:
            action = random.randrange(num_actions)


        obs, reward, done, _ = env.step(action)
        reward = max(-1.0, min(reward, 1.0))
        replay_buffer.store_effect(last_idx, action, reward, done)
        if done:
            obs = env.reset()
        last_obs = obs


        if (t > learning_starts and t % learning_freq == 0 and replay_buffer.can_sample(batch_size)):

            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)
            obs_batch = Variable(torch.from_numpy(obs_batch).type(dtype) / 255.0)
            act_batch = Variable(torch.from_numpy(act_batch).long())
            rew_batch = Variable(torch.from_numpy(rew_batch))
            next_obs_batch = Variable(torch.from_numpy(next_obs_batch).type(dtype) / 255.0)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)



            current_Q_values = Q(obs_batch).gather(1, act_batch.unsqueeze(1)).squeeze()
            next_max_q = target_Q(next_obs_batch).gather(1, torch.max(Q(next_obs_batch), 1)[1].unsqueeze(1)).squeeze(1)
            next_Q_values = not_done_mask * next_max_q
            target_Q_values = rew_batch + (gamma * next_Q_values)

            bellman_error = target_Q_values - current_Q_values
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            d_error = clipped_bellman_error * -1.0


            optimizer.zero_grad()
            current_Q_values.backward(d_error.data)
            optimizer.step()

            frame_count += 1

            if frame_count % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
            current_best_reward = np.max(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)
        Statistic["best_reward"].append(current_best_reward)

        if t % target_update_freq == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("best reward in 100 episodes %f" % current_best_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % 'statistics.pkl')
