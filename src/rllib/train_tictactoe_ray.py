from pprint import pprint
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from TicTacToeRayEnv import TicTacToeRayEnv

def log_result(r):
    learner = r["learners"]["p1"]
    print(
        f"iter={r['training_iteration']:>3} | "
        f"steps={int(r['num_env_steps_sampled_lifetime']):>6} | "
        f"episodes={int(r['env_runners']['num_episodes_lifetime']):>4} | "
        f"episode_return={r['env_runners']['episode_return_mean']:+7.2f} | "
        f"episode_len={r['env_runners']['episode_len_mean']:5.1f} | "
        f"policy_loss={learner['policy_loss']:+.4f} | "
        f"vf_loss={learner['vf_loss']:.2f} | "
        f"entropy={learner['entropy']:.3f}"
    )


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
    metrics = algo.train()
    print("=======================================")
    log_result(metrics)

