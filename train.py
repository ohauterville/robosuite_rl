import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == "__main__":

    run_name = "run_0"

    checkpoint_dir = os.path.join("tmp/td3", run_name)
    log_dir = os.path.join("logs", run_name)
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env_name = "Door"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(
            default_controller="JOINT_VELOCITY"
        ),
        has_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        checkpoint_dir=checkpoint_dir,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
    )

    writer = SummaryWriter(log_dir)
    n_games = 1000
    best_score = 0 # to do later

    episode_identifier = f"{run_name}=actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} layer1_size={layer1_size} layer2_size={layer2_size}"

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        writer.add_scalar(f"Run - {episode_identifier}", score, global_step=i)

        print(f"Episode: {i} score {score}")

        if i % 25 == 0 or i == n_games-1:
            agent.save_models()

    print("\nTraining complete.\n")
