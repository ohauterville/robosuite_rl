import time
import os
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper

from td3_torch import Agent

if __name__ == "__main__":
    run_name = "run_0"

    checkpoint_dir = os.path.join("tmp/td3", run_name)

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"
        ),
        has_renderer=True,
        use_camera_obs=False,
        render_camera="frontview",
        has_offscreen_renderer=True,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)
    print(env.action_space.shape[0])

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
    )

    n_games = 3
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    agent.load_models(best_models=True)

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(
                observation, validation=True
            )  # important to set validation
            next_observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            observation = next_observation
            time.sleep(0.01)

        print(f"Episode: {i} score {score}")
