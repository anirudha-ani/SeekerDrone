import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        #actor network 
        self.linear1_actor = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.linear2_actor = tf.keras.layers.Dense(num_actions, activation='softmax')
        
        #critic network
        self.linear1_critic = tf.keras.layers.Dense(128, activation=tf.nn.relu)
        self.linear2_critic = tf.keras.layers.Dense(1)

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this!
        first_layer_output = self.linear1_actor(states)
        prediction_over_actions = self.linear2_actor(first_layer_output)

        return prediction_over_actions

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        first_layer_output = self.linear1_critic(states)
        baseline_value_of_state = self.linear2_critic(first_layer_output)

        return tf.squeeze(baseline_value_of_state)

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
         #Getting the probability distribution of action given state
        probaibility_distribution_of_action_over_states = self.call(states)

        # making a list from 0 to epsiode length
        episode = list(range(0, len(actions)))

        # zipping it with the action now the action will look like [[episode_no, action_taken]]
        action_with_episode = list(zip(episode , actions))

        # getting the probaibility of taken action from the probability distribution of action over state
        probability_of_action = tf.gather_nd(probaibility_distribution_of_action_over_states, action_with_episode)

        # log of probability
        log_probaibility = tf.math.log(probability_of_action)
        
        baseline_value = self.value_function(states)
        advantage_value = discounted_rewards - baseline_value
        
        loss_critic = tf.reduce_sum(tf.math.square(advantage_value))
        loss_actor = -1.0 * tf.reduce_sum(tf.math.multiply(log_probaibility, tf.stop_gradient(advantage_value)))

        return loss_actor + loss_critic
