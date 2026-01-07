import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from TicTacToeRayEnv import TicTacToeRayEnv

ray.init()

register_env("tictactoe", lambda cfg: TicTacToeRayEnv(cfg))

# Create one env to grab spaces for policy specs.
tmp_env = TicTacToeRayEnv({})
obs_space = tmp_env.observation_space
act_space = tmp_env.action_space

def policy_mapping_fn(agent_id, *args, **kwargs):
    # Two-player self-play with two separate policies:
    return "p0" if agent_id in ["player_0", 0, "X"] else "p1"

config = (
    PPOConfig()
    .environment("tictactoe", env_config={})
    .framework("torch")  # or "tf2"
    .env_runners(num_env_runners=2)  # <- OK (don’t set env_runner_cls)
    .multi_agent(
        policies={
            "p0": (None, obs_space, act_space, {}),
            "p1": (None, obs_space, act_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn,
    )
    .training(lr=2e-4, train_batch_size_per_learner=2000, num_epochs=10)
)

algo = config.build()   # if your Ray version supports build_algo(), that’s fine too
rounds = 4
for _ in range(rounds):
    print(algo.train())
    print("=======================================")
    print(f"Completed round {_+1} of {rounds}")
