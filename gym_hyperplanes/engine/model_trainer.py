import time

import gym
import numpy as np

from gym_hyperplanes.engine.dqn import DQN

np.random.seed(round(time.time()) % 1000000)


def execute_hyperplane_search(state_manipulator, config):
    env = gym.make("gym_hyperplanes:hyperplanes-v0")
    env.set_state_manipulator(state_manipulator)

    done = False
    start = round(time.time())
    last_period = start

    dqn_agent = DQN(env=env)
    cur_state = env.reset().reshape(1, env.get_state().shape[0])

    best_reward = None
    worst_reward = None

    step = 1

    best_reward_step_mark = step
    last_reward_step = step

    while (step < config.get_max_steps()) or (config.get_max_steps_no_better_reward() > step - last_reward_step):
        if config.get_max_steps_no_better_reward() < step - last_reward_step:
            print('Was no reward improvement for {} steps. Ending'.format(step - last_reward_step))
            break

        action = dqn_agent.act(cur_state)

        new_state, reward, done, _ = env.step(action)

        if best_reward is None or best_reward < reward:
            best_reward = reward
            best_reward_step_mark = step
            last_reward_step = step
        if worst_reward is None or worst_reward > reward:
            worst_reward = reward

        new_state = new_state.reshape(1, env.get_state().shape[0])
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        dqn_agent.target_train()  # iterates target model

        cur_state = new_state
        if done:
            break

        if step % 1000 == 0:
            now = round(time.time())
            print('{} steps in {} secs, overall {} secs, {} best reward step, {} best reward'.
                  format(step, now - last_period, start - last_period, best_reward_step_mark, best_reward))
            last_period = now
        step += 1

    stop = round(time.time())
    print("rewards: best {}, worst {} in {}".format(best_reward, worst_reward, stop - start))
    if not done:
        print("last state in step {}".format(step))
        # if step % 10 == 0:
        #     dqn_agent.save_model("trial-{}.model".format(stop))
    else:
        print("success state of step {}".format(step))
        # dqn_agent.save_model("success.model")
    env.print_state('Finished in [{}] steps in [{}] secs'.format(step, stop - start))
    return done
