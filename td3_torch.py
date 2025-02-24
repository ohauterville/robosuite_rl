import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import CriticNetwork, ActorNetwork


class Agent:
    def __init__(
        self,
        actor_learning_rate,
        critic_learning_rate,
        input_dims,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,  # should be 1000?
        n_actions=2,
        max_size=1000000,
        layer1_size=256,
        layer2_size=128,
        batch_size=100,
        noise=0.1,
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.bacth_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.update_actor_iter = update_actor_interval

        # Create the networks
        self.actor = ActorNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="actor",
            learning_rate=actor_learning_rate,
        )

        self.critic1 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="critic1",
            learning_rate=critic_learning_rate,
        )

        self.critic2 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="critic2",
            learning_rate=critic_learning_rate,
        )

        # Create the target networks
        self.target_actor = ActorNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="target_actor",
            learning_rate=actor_learning_rate,
        )

        self.target_critic1 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="target_critic1",
            learning_rate=critic_learning_rate,
        )

        self.target_critic2 = CriticNetwork(
            input_dims=input_dims,
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=n_actions,
            name="target_critic2",
            learning_rate=critic_learning_rate,
        )

        self.actor.to(self.actor.device)
        self.critic1.to(self.critic1.device)
        self.critic2.to(self.critic2.device)
        self.target_actor.to(self.target_actor.device)
        self.target_critic1.to(self.target_critic1.device)
        self.target_critic2.to(self.target_critic2.device)

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and validation is False:
            mu = T.tensor(
                np.random.normal(scale=self.noise, size=(self.n_actions,))
            ).to(self.actor.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.actor.device)
            mu = self.actor.forward(state).to(self.actor.device)

        mu_prime = mu + T.tensor(np.random.normal(scale=self.noise), dtype=T.float).to(
            self.actor.device
        )
        mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])

        self.time_step += 1

        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def learn(self):
        if self.memory.mem_ctr < self.bacth_size * 10:
            return

        state, action, reward, next_state, done = self.memory.sample_buffer(
            self.bacth_size
        )

        reward = T.tensor(reward, dtype=T.float).to(self.critic1.device)
        done = T.tensor(done).to(self.critic1.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.critic1.device)
        state = T.tensor(state, dtype=T.float).to(self.critic1.device)
        action = T.tensor(action, dtype=T.float).to(self.critic1.device)

        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(
            T.tensor(np.random.normal(scale=0.2)), -0.5, 0.5
        )
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])

        next_q1 = self.target_critic1.forward(next_state, target_actions)
        next_q2 = self.target_critic2.forward(next_state, target_actions)

        q1 = self.critic1.forward(state, action)
        q2 = self.critic2.forward(state, action)

        next_q1[done] = 0.0
        next_q2[done] = 0.0

        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)

        next_critic_value = T.min(next_q1, next_q2)

        target = (
            reward + self.gamma * next_critic_value
        )  # we make an heuristic on next future reward
        target = target.view(self.bacth_size, 1)

        self.critic1.optimizer.zero_grad()
        self.critic2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)

        critic_loss = q1_loss + q2_loss
        critic_loss.backward()

        # Deepseek suggestion
        # It is to stabilize critic stability
        T.nn.utils.clip_grad_norm_(self.critic1.parameters(), 1.0)
        T.nn.utils.clip_grad_norm_(self.critic2.parameters(), 1.0)

        self.critic1.optimizer.step()
        self.critic2.optimizer.step()

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)  # we want to maximize
        actor_loss.backward()

        self.actor.optimizer.step()
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau == None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic1_params = self.critic1.named_parameters()
        critic2_params = self.critic2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic1_params = self.target_critic1.named_parameters()
        target_critic2_params = self.target_critic2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic1_state_dict = dict(critic1_params)
        critic2_state_dict = dict(critic2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic1_state_dict = dict(target_critic1_params)
        target_critic2_state_dict = dict(target_critic2_params)

        for name in critic1_state_dict:
            critic1_state_dict[name] = (
                tau * critic1_state_dict[name].clone()
                + (1 - tau) * target_critic1_state_dict[name].clone()
            )

        for name in critic2_state_dict:
            critic2_state_dict[name] = (
                tau * critic2_state_dict[name].clone()
                + (1 - tau) * target_critic2_state_dict[name].clone()
            )

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_state_dict[name].clone()
            )

        self.target_critic1.load_state_dict(critic1_state_dict)
        self.target_critic2.load_state_dict(critic2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        try:
            self.actor.save_checkpoint()
            self.target_actor.save_checkpoint()
            self.critic1.save_checkpoint()
            self.target_critic1.save_checkpoint()
            self.critic2.save_checkpoint()
            self.target_critic2.save_checkpoint()
            print("Successfully saved the models.")
        except:
            print("Failed to save the models.")

    def load_models(self):
        try:
            self.actor.load_checkpoint()
            self.target_actor.load_checkpoint()
            self.critic1.load_checkpoint()
            self.target_critic1.load_checkpoint()
            self.critic2.load_checkpoint()
            self.target_critic2.load_checkpoint()
            print("Successfully loaded models.")
        except Exception as e:
            print("Failed to load models. Starting from scratch.")
