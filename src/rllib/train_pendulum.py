from ray.rllib.algorithms.ppo import PPOConfig

# Create a config instance for the PPO algorithm.
config = (
    PPOConfig()
    .environment("Pendulum-v1")
)
config.env_runners(num_env_runners=2)
config.training(
    lr=0.0002,
    train_batch_size_per_learner=2000,
    num_epochs=10,
)
# Build the Algorithm (PPO).
ppo = config.build_algo()
from pprint import pprint

for _ in range(4):
    pprint(ppo.train())
    print("---------------------------------------")

# Save the trained Algorithm to a checkpoint.
checkpoint_path = ppo.save_to_path()
print(f"Checkpoint saved at: {checkpoint_path}")
# OR:
# ppo.save_to_path([a checkpoint location of your choice])

config.evaluation(
    # Run one evaluation round every iteration.
    evaluation_interval=1,

    # Create 2 eval EnvRunners in the extra EnvRunnerGroup.
    evaluation_num_env_runners=2,

    # Run evaluation for exactly 10 episodes. Note that because you have
    # 2 EnvRunners, each one runs through 5 episodes.
    evaluation_duration_unit="episodes",
    evaluation_duration=10,
)

# Rebuild the PPO, but with the extra evaluation EnvRunnerGroup
ppo_with_evaluation = config.build_algo()

for _ in range(3):
    pprint(ppo_with_evaluation.train())
    print("=======================================")
