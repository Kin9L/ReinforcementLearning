import gym
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


STATE = "state"
ACTION = "action"
REWARD = "reward"
PROB = "prob"
VALUE = "value"


class ActorNetwork(keras.Model):
    def __init__(self, state_dim, action_dim, epsilon):
        super(ActorNetwork, self).__init__()
        self.fc1 = Dense(state_dim, activation="relu")
        self.fc2 = Dense(state_dim, activation="relu")
        self.fc3 = Dense(action_dim, activation="softmax")
        self.epsilon = epsilon

    def call(self, state):
        x1 = self.fc1(state)
        x2 = self.fc2(x1)
        output = self.fc3(x2)
        return output

    def compute_loss(self, actions, states, old_probs, advantages):
        states = tf.convert_to_tensor(states)
        new_probs = self.call(states)
        dist = tfp.distributions.Categorical(new_probs)
        log_new_probs = dist.log_prob(actions)
        ratio = tf.exp(log_new_probs - old_probs)
        clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon)
        adv_tensor = tf.convert_to_tensor([advantages])
        adv_tensor = tf.transpose(adv_tensor)
        surrogate = -tf.minimum(ratio * adv_tensor, clipped_ratio * adv_tensor)
        return tf.reduce_mean(surrogate)

    def train(self, actions, states, old_probs, advantages):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(actions, states, old_probs, advantages)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class CriticNetwork(keras.Model):
    def __init__(self, state_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = Dense(state_dim, activation="relu")
        self.fc2 = Dense(state_dim, activation="relu")
        self.fc3 = Dense(1, None)

    def call(self, state):
        x1 = self.fc1(state)
        x2 = self.fc2(x1)
        output = self.fc3(x2)
        return output

    def compute_loss(self, states, target_values):
        states = tf.convert_to_tensor(states)
        predit_values = self.call(states)
        target_values = tf.transpose(target_values)
        mse = tf.keras.losses.MeanSquaredError()
        return mse(predit_values, target_values)

    def train(self, states, target_values):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(states, target_values)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class Agent:
    def __init__(
        self,
        action_dim,
        state_dim,
        actor_lr=0.0001,
        critic_lr=0.0001,
        gamma=0.99,
        gae_lambda=0.1,
        epsilon=0.1,
        chkpt_dir="ppo_models/",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.chkpt_dir = chkpt_dir

        # actor
        self.actor = ActorNetwork(state_dim, action_dim, epsilon)
        self.actor.compile(optimizer=Adam(learning_rate=actor_lr))

        # critic
        self.critic = CriticNetwork(state_dim)
        self.critic.compile(optimizer=Adam(learning_rate=critic_lr))

        self.trajectory = {STATE: [], ACTION: [], REWARD: [], PROB: [], VALUE: []}
        pass

    def save_models(self):
        print(6 * "#", " start saving models ", 6 * "#")
        self.actor.save(self.chkpt_dir + "actor")
        self.critic.save(self.chkpt_dir + "critic")

    def load_models(self):
        print(6 * "#", " start loading models ", 6 * "#")
        self.actor = keras.models.load_model(self.chkpt_dir + "actor")
        self.critic = keras.models.load_model(self.chkpt_dir + "critic")

    def choose_action(self, observation):
        observation = tf.convert_to_tensor([observation])
        prob = self.actor(observation)
        value = self.critic(observation)
        dist = tfp.distributions.Categorical(prob)
        action = dist.sample()
        action = action.numpy()[0]
        log_prob = dist.log_prob(action)
        log_prob = log_prob.numpy()
        value = value.numpy()
        return action, log_prob, value

    def store_trajectory(self, state, action, reward, prob, cur_val):
        self.trajectory[STATE].append(state)
        self.trajectory[ACTION].append(action)
        self.trajectory[REWARD].append(reward)
        self.trajectory[PROB].append(prob)
        self.trajectory[VALUE].append(cur_val)
        pass

    def clear_trajectory(self):
        self.trajectory = {STATE: [], ACTION: [], REWARD: [], PROB: [], VALUE: []}

    def gae_calculation(self, future_value):
        gae_target = 0
        # calculate the advantages
        advantages = np.zeros(len(self.trajectory[STATE]), dtype=np.float32)
        target_values = np.zeros_like(advantages, dtype=np.float32)
        trajectory_length = len(self.trajectory[STATE])
        for t in range(trajectory_length - 1, -1, -1):
            delta = (
                self.trajectory[REWARD][t]
                + self.gamma * future_value
                - self.trajectory[VALUE][t]
            )
            advantages[t] = self.gamma * self.gae_lambda * gae_target + delta
            gae_target = advantages[t]
            target_values[t] = advantages[t] + self.trajectory[VALUE][t]
            future_value = self.trajectory[VALUE][t]
        return advantages, target_values

    def lean(self, advantages, target_values):
        actor_loss = 0
        critic_loss = 0
        # compute actor's loss
        actor_loss = self.actor.train(
            self.trajectory[ACTION],
            self.trajectory[STATE],
            self.trajectory[PROB],
            advantages,
        )
        # compute critic's loss
        critic_loss = self.critic.train(self.trajectory[STATE], target_values)
        return actor_loss, critic_loss


def train():
    env = gym.make("CartPole-v0")
    n_episodes = 5000
    max_epochs = 3
    gamma = 0.99
    update_freq = 5
    agent = Agent(
        action_dim=env.action_space.n,
        state_dim=env.observation_space.shape[0],
        actor_lr=0.0003,
        critic_lr=0.0001,
        gamma=gamma,
        gae_lambda=0.95,
        epsilon=0.1,
    )
    score_history = []
    best_score = 0
    for ep in range(n_episodes):
        print("episode: ", ep)
        obs = env.reset()
        done = False
        score = 0
        future_value = 0
        while not done:
            action, prob, value = agent.choose_action(observation=obs)
            next_obs, reward, done, info = env.step(action)
            score += reward
            agent.store_trajectory(obs, action, reward, prob, value)
            obs = next_obs
            if len(agent.trajectory) > update_freq or done:
                if not done:
                    future_value = agent.critic(next_obs).numpy()
                else:
                    future_value = 0
                advantages, target_values = agent.gae_calculation(future_value)
                for epoch in range(max_epochs):
                    actor_loss, critic_loss = agent.lean(advantages, target_values)
                agent.clear_trajectory()
        if len(score_history) > 0 and score > max(score_history):  # Save models
            agent.save_models()
            best_score = score
        score_history.append(score)
        print("current best score: ", best_score)


if __name__ == "__main__":
    train()