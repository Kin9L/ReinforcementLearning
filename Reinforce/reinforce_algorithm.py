import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class PolicyNetwork(keras.Model):

    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = Dense(state_dim, activation='relu')
        self.fc2 = Dense(state_dim, activation='relu')
        self.fc3 = Dense(action_dim, activation='softmax')

    def call(self, state):
        x1 = self.fc1(state)
        x2 = self.fc2(x1)
        output = self.fc3(x2)
        return output


class Agent:

    def __init__(self,
                 action_dim,
                 state_dim,
                 n_episodes=100,
                 learning_rate=0.0001,
                 gama=0.99,
                 chkpt_dir='reinforce_models/') -> None:
        self.gama = gama
        self.n_episodes = n_episodes
        self.chkpt_dir = chkpt_dir

        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor.compile(optimizer=Adam(learning_rate=learning_rate))

        self.trajectory = []

    def save_models(self):
        print(6 * '#', ' start saving models ', 6 * '#')
        self.actor.save(self.chkpt_dir + 'policy')

    def load_models(self):
        print(6 * '#', ' start loading models ', 6 * '#')
        self.actor = keras.models.load_model(self.chkpt_dir + 'policy')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])

        probs = self.actor(state)
        dist = tfp.distributions.Categorical(probs)
        action = dist.sample()
        action = action.numpy()[0]

        return action

    def store_current_trajectory(self, state, action, reward):
        self.trajectory.append([state, action, reward])

    def clear_current_trajectory(self):
        self.trajectory = []

    def learn(self):
        STATE = 0
        ACTION = 1
        REWARD = 2

        with tf.GradientTape() as tape:
            loss = 0
            future_reward = 0
            # calculate the discount rewards
            discount_rewards = np.zeros(len(self.trajectory))
            trajectory_length = len(self.trajectory)
            for t in range(trajectory_length - 1, -1, -1):
                future_reward = self.trajectory[t][REWARD] + \
                    self.gama * future_reward
                discount_rewards[t] = future_reward

                state = tf.convert_to_tensor([self.trajectory[t][STATE]])
                probs = self.actor(state)
                action_probs = tfp.distributions.Categorical(probs)
                log_prob = action_probs.log_prob(self.trajectory[t][ACTION])
                loss += -discount_rewards[t] * tf.squeeze(log_prob)
        actor_params = self.actor.trainable_variables
        gradients = tape.gradient(loss, actor_params)
        self.actor.optimizer.apply_gradients(zip(gradients, actor_params))
        self.clear_current_trajectory()


def train():
    env = gym.make('CartPole-v0')
    n_episodes = 50000
    learning_rate = 0.0001
    gama = 0.99
    agent = Agent(action_dim=env.action_space.n,
                  state_dim=env.observation_space.shape[0],
                  n_episodes=n_episodes,
                  learning_rate=learning_rate,
                  gama=gama)
    score_history = []
    best_score = 0
    for i in range(n_episodes):
        print("episode: ", i)
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation=observation)
            next_obs, reward, done, info = env.step(action)
            score += reward
            agent.store_current_trajectory(observation, action, reward)
            observation = next_obs
        agent.learn()
        agent.clear_current_trajectory()  # Reinforce is on policy algorithm
        if len(score_history) > 0 and score > max(score_history):  # Save models
            agent.save_models()
            best_score = score
        score_history.append(score)
        print("current best score: ", best_score)


if __name__ == "__main__":
    train()
