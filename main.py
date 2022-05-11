import os
import sys
import gym
from pylab import *
import numpy as np
import tensorflow as tf
from reinforce_with_baseline import ReinforceWithBaseline
from airsim_env_ani import Env

def visualize_episode(env, model):
    """
    HELPER - do not edit.
    Takes in an enviornment and a model and visualizes the model's actions for one episode.
    We recomend calling this function every 20 training episodes. Please remove all calls of 
    this function before handing in.

    :param env: The cart pole enviornment object
    :param model: The model that will decide the actions to take
    """

    done = False
    state = env.reset()
    env.render()

    while not done:
        newState = np.reshape(state, [1, state.shape[0]])
        prob = model.call(newState)
        newProb = np.reshape(prob, prob.shape[1])
        action = np.random.choice(np.arange(newProb.shape[0]), p = newProb)

        state, _, done, _ = env.step(action)
        env.render()


def visualize_data(total_rewards):
    """
    HELPER - do not edit.
    Takes in array of rewards from each episode, visualizes reward over episodes

    :param total_rewards: List of rewards from all episodes
    """

    x_values = arange(0, len(total_rewards), 1)
    y_values = total_rewards
    plot(x_values, y_values)
    xlabel('episodes')
    ylabel('cumulative rewards')
    title('Reward by Episode')
    grid(True)
    show()


def discount(rewards, discount_factor=.99):
    """
    Takes in a list of rewards for each timestep in an episode, and
    returns a list of the discounted rewards for each timestep, which
    are calculated by summing the rewards for each future timestep, discounted
    by how far in the future it is.
    For example, in the simple case where the episode rewards are [1, 3, 5] 
    and discount_factor = .99 we would calculate:
    dr_1 = 1 + 0.99 * 3 + 0.99^2 * 5 = 8.8705
    dr_2 = 3 + 0.99 * 5 = 7.95
    dr_3 = 5
    and thus return [8.8705, 7.95 , 5].
    Refer to the slides for more details about how/why this is done.

    :param rewards: List of rewards from an episode [r_{t1},r_{t2},...]
    :param discount_factor: Gamma discounting factor to use, defaults to .99
    :returns: discounted_rewards: list containing the discounted rewards for each timestep in the original rewards list
    """
    #   Compute discounted rewards
    discounted_rewards = []
    for reward in rewards: 
        discounted_rewards.append(reward)

    for index in range(len(discounted_rewards) - 2, -1, -1):
        discounted_rewards[index] += discounted_rewards[index+1] * discount_factor
    
    return discounted_rewards




def generate_trajectory(env, model):
    """
    Generates lists of states, actions, and rewards for one complete episode.

    :param env: The openai gym environment
    :param model: The model used to generate the actions
    :returns: A tuple of lists (states, actions, rewards), where each list has length equal to the number of timesteps in the episode
    """
    states = []
    actions = []
    rewards = []
    state = env.reset()
    done = False
    step_count = 0
    while not done and step_count<10:
        # 1) use model to generate probability distribution over next actions
        # 2) sample from this distribution to pick the next action

        #Getting the probability distribution of action given state
        probaibility_distribution_of_action_over_states = model.call(np.expand_dims(state, axis=0))
        print("Prob dist over action = ", probaibility_distribution_of_action_over_states)
        action = np.random.choice(model.num_actions, p=np.squeeze(probaibility_distribution_of_action_over_states, axis=0))
        print("Action = ", action)

        states.append(state)
        actions.append(action)
        state, rwd, done, _ = env.step(action)
        print("State = ", np.reshape(state,(4,6)))
        print("Reward = ", rwd)
        rewards.append(rwd)
        step_count += 1

    return states, actions, rewards


def train(env, model):
    """
    This function should train your model for one episode.
    Each call to this function should generate a complete trajectory for one
    episode (lists of states, action_probs, and rewards seen/taken in the episode), and
    then train on that data to minimize your model loss.
    Make sure to return the total reward for the episode

    :param env: The openai gym environment
    :param model: The model
    :returns: The total reward for the episode
    """

    # 1) Use generate trajectory to run an episode and get states, actions, and rewards.
    # 2) Compute discounted rewards.
    # 3) Compute the loss from the model and run backpropagation on the model.
    

    with tf.GradientTape() as tape:
        states, actions, rewards = generate_trajectory(env, model)
        discounted_rewards = discount(rewards)
        loss = model.loss(np.array(states), np.array(actions), np.array(discounted_rewards))

    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return np.sum(rewards)




def main():
    env = Env()
    state_size = 24
    num_actions = 4
    model = ReinforceWithBaseline(state_size, num_actions)

    no_of_episode = 650
    total_reward = []
    for episode in range(no_of_episode):
        reward = train(env, model)
        total_reward.append(reward)
        print("reward in episode ", episode, " = " , reward) 
    
    average_of_last_50 = sum(total_reward[-50:]) / 50 

    print("Average reward of last 50 epsiode = ", average_of_last_50)
    #   Visualize your rewards.
    visualize_data(total_reward)
    visualize_episode(env,model)


if __name__ == '__main__':
    main()
